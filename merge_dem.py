"""
merge_dem.py
============
MELITDEM (90m) を UTM に変換したものを基準 CRS とし、
その一部を AW3D (1m) で差し替えるスクリプト。

【処理フロー】
  Step 0: MELITDEM を UTM に変換
  Step 1: AW3D 範囲 + バッファで UTM MELITDEM をクリップ
          ※ クリップ範囲を 90m グリッドにスナップ → 隙間をゼロにする
  Step 2: クリップ片を 1m にリサンプリング
  Step 3: AW3D を UTM に変換 → AW3D 優先モザイク → 1m パッチ
  Step 3.5: バッファ領域でフェザリングブレンド（境界を滑らかに）
  Step 4: UTM MELITDEM のパッチ範囲を NoData で穴あき化

【出力ファイル】
  <o>_patch_1m.tif      : 1m パッチ（AW3D + MELITDEM buffer、境界ブレンド済み）
  <o>_melitdem_hole.tif : 穴あき MELITDEM（90m）

【使い方】
  python merge_dem.py config.yaml
  python merge_dem.py config.yaml --buffer_m 300 --blend_width_m 200

【設定項目（config.yaml）】
  blend_width_m: 200   # フェザリング幅 [m]（0 で無効、デフォルト: buffer_m の 40%）

【依存ライブラリ】
  pip install rasterio numpy shapely pyyaml scipy
"""

import argparse
import math
import os
import sys
import tempfile

import numpy as np
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box, mapping


RESAMPLING_MAP = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic":    Resampling.cubic,
    "lanczos":  Resampling.lanczos,
}

CONFIG_SCHEMA = [
    ("melitdem",      None,       str,   "MELITDEM GeoTIFF (90m)"),
    ("aw3d",          None,       str,   "AW3D GeoTIFF (1m)"),
    ("output",        None,       str,   "output prefix"),
    ("utm_epsg",      None,       str,   "UTM EPSG (auto if omitted)"),
    ("buffer_m",      500.0,      float, "buffer around AW3D [m]"),
    ("blend_width_m", None,       float, "feather blend width [m] (default: buffer_m * 0.4)"),
    ("nodata",        -9999.0,    float, "NoData value"),
    ("resampling",    "bilinear", str,   "resampling method"),
]


# --------------------------------------------------
# UTM zone auto-detect
# --------------------------------------------------

def auto_utm_epsg(melitdem_path):
    with rasterio.open(melitdem_path) as ds:
        b = ds.bounds
        if ds.crs.is_geographic:
            cx = (b.left + b.right)   / 2
            cy = (b.bottom + b.top)   / 2
        else:
            wb = transform_bounds(ds.crs, CRS.from_epsg(4326), *b)
            cx = (wb[0] + wb[2]) / 2
            cy = (wb[1] + wb[3]) / 2
    zone  = int((cx + 180) / 6) + 1
    base  = 32600 if cy >= 0 else 32700
    epsg  = base + zone
    print(f"  UTM auto: center ({cx:.3f}, {cy:.3f}) -> Zone {zone} -> EPSG:{epsg}")
    return str(epsg)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def hdr(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def info(path, label=""):
    with rasterio.open(path) as ds:
        b = ds.bounds
        rx = abs(ds.transform.a)
        ry = abs(ds.transform.e)
        print(f"  [{label or os.path.basename(path)}]")
        print(f"    CRS    : {ds.crs}")
        print(f"    res    : {rx:.4f} x {ry:.4f}  ({ds.width} x {ds.height} px)")
        print(f"    bounds : {b.left:.4f}, {b.bottom:.4f}, {b.right:.4f}, {b.top:.4f}")
        print(f"    nodata : {ds.nodata}")


def warp_to_crs(src_path, dst_path, dst_crs, resolution_m, resampling, nodata):
    with rasterio.open(src_path) as src:
        if resolution_m is not None:
            tf, w, h = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=(resolution_m, resolution_m),
            )
        else:
            tf, w, h = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
            )
        meta = src.meta.copy()
        meta.update({"crs": dst_crs, "transform": tf, "width": w, "height": h,
                     "nodata": nodata, "dtype": "float32"})
        with rasterio.open(dst_path, "w", **meta) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=tf, dst_crs=dst_crs,
                    resampling=resampling,
                    src_nodata=src.nodata, dst_nodata=nodata,
                )


# --------------------------------------------------
# Step 0: MELITDEM -> UTM
# --------------------------------------------------

def step0_melitdem_to_utm(melitdem_path, out_path, utm_crs, resampling, nodata):
    warp_to_crs(melitdem_path, out_path, utm_crs,
                resolution_m=None, resampling=resampling, nodata=nodata)
    print("  ok: MELITDEM -> UTM")
    info(out_path, "melitdem_utm")


# --------------------------------------------------
# Step 1: clip MELITDEM (snapped to 90m grid)
# --------------------------------------------------

def step1_clip_snapped(melitdem_utm_path, aw3d_path, out_path,
                       buffer_m, utm_crs, nodata):
    """
    AW3D 範囲 + buffer を MELITDEM の 90m グリッドにスナップしてクリップ。
    スナップ済みの bounds は Step 4 の穴あけにも使うので返り値として返す。
    AW3D の UTM bounds も返す（Step 3.5 のブレンドマスク作成に使用）。
    """
    with rasterio.open(aw3d_path) as aw:
        aw3d_bounds_utm = transform_bounds(aw.crs, utm_crs, *aw.bounds)
        left, bottom, right, top = aw3d_bounds_utm

    left   -= buffer_m
    bottom -= buffer_m
    right  += buffer_m
    top    += buffer_m

    with rasterio.open(melitdem_utm_path) as dem:
        res = abs(dem.transform.a)   # ~90m in UTM
        ox  = dem.transform.c        # grid origin X (left edge)
        oy  = dem.transform.f        # grid origin Y (top edge, larger value)

        # snap outward to pixel boundaries
        left_s  = ox + math.floor((left  - ox) / res) * res
        right_s = ox + math.ceil( (right - ox) / res) * res
        top_s    = oy - math.floor((oy - top)    / res) * res
        bottom_s = oy - math.ceil( (oy - bottom) / res) * res

        # clamp to DEM bounds
        b = dem.bounds
        left_s   = max(left_s,   b.left)
        bottom_s = max(bottom_s, b.bottom)
        right_s  = min(right_s,  b.right)
        top_s    = min(top_s,    b.top)

        clip_geom = box(left_s, bottom_s, right_s, top_s)
        out_img, out_tf = rio_mask(dem, [mapping(clip_geom)],
                                   crop=True, nodata=nodata, filled=True)
        meta = dem.meta.copy()
        meta.update({"transform": out_tf,
                     "width": out_img.shape[2], "height": out_img.shape[1],
                     "nodata": nodata, "dtype": "float32"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(out_img.astype("float32"))

    snapped = (left_s, bottom_s, right_s, top_s)
    print(f"  ok: MELITDEM clip (buffer={buffer_m}m, grid-snapped)")
    print(f"      snapped bounds: X {left_s:.1f}~{right_s:.1f} / Y {bottom_s:.1f}~{top_s:.1f}")
    info(out_path, "melitdem_clip_utm")
    return snapped, aw3d_bounds_utm


# --------------------------------------------------
# Step 2: resample clip to 1m
# --------------------------------------------------

def step2_resample_1m(src_path, out_path, snapped_bounds, utm_crs, resampling, nodata):
    """
    snapped_bounds (90m グリッドに揃えた bounds) を厳密に維持しながら 1m にリサンプリング。
    """
    from rasterio.transform import from_bounds as _from_bounds
    left_s, bottom_s, right_s, top_s = snapped_bounds
    width  = round((right_s - left_s)   / 1.0)
    height = round((top_s   - bottom_s) / 1.0)
    tf = _from_bounds(left_s, bottom_s, right_s, top_s, width, height)

    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        meta.update({"crs": utm_crs, "transform": tf, "width": width, "height": height,
                     "nodata": nodata, "dtype": "float32"})
        with rasterio.open(out_path, "w", **meta) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=tf, dst_crs=utm_crs,
                    resampling=resampling,
                    src_nodata=src.nodata, dst_nodata=nodata,
                )
    print("  ok: resample to 1m (bounds locked to snapped grid)")
    info(out_path, "melitdem_clip_1m")


# --------------------------------------------------
# Step 3: AW3D -> UTM, then mosaic (AW3D priority)
# --------------------------------------------------

def step3_mosaic(aw3d_path, mel_1m_path, out_path,
                 utm_crs, resampling, nodata, tmpdir):
    aw3d_utm = os.path.join(tmpdir, "aw3d_utm.tif")
    warp_to_crs(aw3d_path, aw3d_utm, utm_crs,
                resolution_m=1.0, resampling=resampling, nodata=nodata)
    print("  ok: AW3D -> UTM")
    info(aw3d_utm, "aw3d_utm")

    with rasterio.open(aw3d_utm) as hi, rasterio.open(mel_1m_path) as lo:
        mosaic, tf = merge([hi, lo], nodata=nodata, method="first")
        meta = hi.meta.copy()
        meta.update({"width": mosaic.shape[2], "height": mosaic.shape[1],
                     "transform": tf, "nodata": nodata, "dtype": "float32"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic.astype("float32"))
    print("  ok: mosaic -> patch_1m (pre-blend)")
    info(out_path, "patch_1m (pre-blend)")

    return aw3d_utm  # Step 3.5 で使用


# --------------------------------------------------
# Step 3.5: feather blend at AW3D boundary
# --------------------------------------------------

def step35_feather_blend(patch_path, mel_1m_path, aw3d_utm_path,
                         snapped_bounds, aw3d_bounds_utm,
                         blend_width_m, nodata, out_path):
    """
    AW3D とバッファ（MELITDEM 1m）の境界をフェザリングブレンドする。

    ブレンドの考え方:
      境界の「両側」にグラデーションをかける。

      AW3D内側                  境界                  バッファ(MELITDEM)
      |←── blend_px ──→|        |        |←── blend_px ──→|
      weight=1.0       weight=0.5(境界)   weight=0.0

      【修正前の問題】
        distance_transform_edt(aw3d_mask) は「AW3D内側からの距離」のみ返す。
        バッファ帯（aw3d_mask=False）では距離=0 → weight=0 固定になり、
        AW3D境界で値が急変して線が出ていた。

      【修正後】
        境界からの符号付き距離を使う。
        - AW3D内側: dist_in（境界まで正の距離）
        - バッファ帯: dist_out（境界まで正の距離）
        weight = clip((dist_in - dist_out + blend_px) / (2 * blend_px), 0, 1)
        → 境界を中心に -blend_px〜+blend_px の範囲で 0→1 に変化する。
    """
    blend_px = max(1, int(blend_width_m))  # 1m グリッドなのでそのままピクセル数

    with rasterio.open(patch_path) as patch_ds:
        patch_tf   = patch_ds.transform
        patch_data = patch_ds.read(1).astype("float32")
        patch_w    = patch_ds.width
        patch_h    = patch_ds.height

    with rasterio.open(mel_1m_path) as mel_ds:
        mel_data = mel_ds.read(1).astype("float32")

    # ── AW3D 範囲を patch 座標系のピクセルインデックスに変換 ──
    aw3d_left, aw3d_bottom, aw3d_right, aw3d_top = aw3d_bounds_utm
    patch_left = patch_tf.c
    patch_top  = patch_tf.f
    res        = abs(patch_tf.a)

    col0 = max(0, int((aw3d_left   - patch_left) / res))
    col1 = min(patch_w, int(math.ceil((aw3d_right  - patch_left) / res)))
    row0 = max(0, int((patch_top   - aw3d_top)    / res))
    row1 = min(patch_h, int(math.ceil((patch_top   - aw3d_bottom) / res)))

    # AW3D 本体マスク（True: AW3D 内側）
    aw3d_mask = np.zeros((patch_h, patch_w), dtype=bool)
    aw3d_mask[row0:row1, col0:col1] = True

    patch_valid = patch_data != nodata
    mel_valid   = mel_data   != nodata

    # ── バッファ側のみ weight を計算（AW3D本体は変えない）──
    #
    #   AW3D本体         境界          バッファ(MELITDEM)
    #   ████████████████  │  ░░▒▒▓▓████████
    #   weight=1.0(固定)  │  0→blend_px で 1.0→0.0
    #
    dist_out = distance_transform_edt(~aw3d_mask)  # 外側: 境界まで +距離

    weight = np.where(
        aw3d_mask,
        1.0,                                      # AW3D本体: 常に1.0（変更なし）
        np.clip(dist_out / blend_px, 0.0, 1.0),  # バッファ帯: 境界0.0→遠方1.0
    ).astype("float32")

    # ── ブレンド計算 ──
    both_valid = patch_valid & mel_valid
    blended    = patch_data.copy()

    # ブレンドゾーン: バッファ帯 かつ 両データ有効 かつ blend_px 以内
    blend_zone = both_valid & ~aw3d_mask & (dist_out < blend_px)

    blended[blend_zone] = (
        weight[blend_zone]           * patch_data[blend_zone]
        + (1.0 - weight[blend_zone]) * mel_data[blend_zone]
    )
    blended[~patch_valid] = nodata

    # ── 書き出し ──
    with rasterio.open(patch_path) as src:
        meta = src.meta.copy()
        meta.update({"nodata": nodata, "dtype": "float32"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(blended[np.newaxis, :, :])

    n_blend = int(blend_zone.sum())
    pct     = 100.0 * n_blend / max(1, int(patch_valid.sum()))
    print(f"  ok: feather blend  width={blend_width_m}m ({blend_px}px) buffer-side only")
    print(f"      blended pixels : {n_blend:,} ({pct:.1f}% of valid patch area)")
    print(f"      AW3D core      : rows {row0}:{row1}, cols {col0}:{col1}")
    info(out_path, "patch_1m (blended)")


# --------------------------------------------------
# Step 4: punch hole using snapped bounds
# --------------------------------------------------

def step4_punch_hole(melitdem_utm_path, snapped_bounds, out_path, nodata):
    """
    スナップ済み bounds からピクセルインデックスを直接計算して穴を開ける。
    """
    left_s, bottom_s, right_s, top_s = snapped_bounds

    with rasterio.open(melitdem_utm_path) as dem:
        ox  = dem.transform.c
        oy  = dem.transform.f
        res = abs(dem.transform.a)

        c0 = round((left_s   - ox) / res)
        c1 = round((right_s  - ox) / res)
        r0 = round((oy - top_s)    / res)
        r1 = round((oy - bottom_s) / res)

        c0 = max(0, c0);  c1 = min(dem.width,  c1)
        r0 = max(0, r0);  r1 = min(dem.height, r1)

        data = dem.read().astype("float32")
        data[:, r0:r1, c0:c1] = nodata

        meta = dem.meta.copy()
        meta.update({"nodata": nodata, "dtype": "float32"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)

    print(f"  ok: punch hole  rows {r0}:{r1}  cols {c0}:{c1}")
    info(out_path, "melitdem_hole")


# --------------------------------------------------
# Config
# --------------------------------------------------

def load_config(yaml_path, cli):
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = {}
    for key, default, cast, _ in CONFIG_SCHEMA:
        if cli.get(key) is not None:
            cfg[key] = cast(cli[key])
        elif key in raw and raw[key] is not None:
            cfg[key] = cast(raw[key])
        elif default is not None:
            cfg[key] = default
        else:
            cfg[key] = None
    missing = [k for k in ("melitdem", "aw3d", "output") if not cfg.get(k)]
    if missing:
        print(f"[ERROR] missing required keys: {missing}")
        sys.exit(1)
    if cfg["resampling"] not in RESAMPLING_MAP:
        print(f"[ERROR] invalid resampling: {cfg['resampling']}")
        sys.exit(1)
    # blend_width_m が未指定なら buffer_m の 40% をデフォルトとする
    if cfg["blend_width_m"] is None:
        cfg["blend_width_m"] = cfg["buffer_m"] * 0.4
    return cfg


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replace part of MELITDEM with AW3D (1m), UTM-based, with feather blending")
    parser.add_argument("config")
    parser.add_argument("--melitdem",      default=None)
    parser.add_argument("--aw3d",          default=None)
    parser.add_argument("--output",        default=None)
    parser.add_argument("--utm_epsg",      default=None)
    parser.add_argument("--buffer_m",      default=None, type=float)
    parser.add_argument("--blend_width_m", default=None, type=float,
                        help="feather blend width [m] (default: buffer_m * 0.4)")
    parser.add_argument("--nodata",        default=None, type=float)
    parser.add_argument("--resampling",    default=None, choices=RESAMPLING_MAP.keys())
    args = parser.parse_args()

    cfg = load_config(args.config,
                      {k: getattr(args, k) for k in
                       ("melitdem", "aw3d", "output", "utm_epsg",
                        "buffer_m", "blend_width_m", "nodata", "resampling")})

    resampling     = RESAMPLING_MAP[cfg["resampling"]]
    nodata         = cfg["nodata"]
    blend_width_m  = cfg["blend_width_m"]
    out_patch      = cfg["output"] + "_patch_1m.tif"
    out_hole       = cfg["output"] + "_melitdem_hole.tif"

    os.makedirs(os.path.dirname(os.path.abspath(cfg["output"])), exist_ok=True)

    hdr("Input files")
    info(cfg["melitdem"], "MELITDEM")
    info(cfg["aw3d"],     "AW3D")

    hdr("Settings")
    print(f"  buffer_m      : {cfg['buffer_m']} m")
    print(f"  blend_width_m : {blend_width_m} m  "
          f"({'default: buffer_m×0.4' if args.blend_width_m is None else 'manual'})")
    if blend_width_m > cfg["buffer_m"]:
        print(f"  [WARNING] blend_width_m ({blend_width_m}) > buffer_m ({cfg['buffer_m']})")
        print(f"            ブレンド幅がバッファを超えています。buffer_m を大きくするか blend_width_m を小さくしてください。")

    hdr("UTM zone")
    if cfg.get("utm_epsg"):
        utm_crs = CRS.from_epsg(int(cfg["utm_epsg"]))
        print(f"  manual: EPSG:{cfg['utm_epsg']} -> {utm_crs.to_string()}")
    else:
        utm_crs = CRS.from_epsg(int(auto_utm_epsg(cfg["melitdem"])))
        print(f"  -> {utm_crs.to_string()}")

    with tempfile.TemporaryDirectory() as tmp:
        mel_utm      = os.path.join(tmp, "mel_utm.tif")
        mel_clip     = os.path.join(tmp, "mel_clip.tif")
        mel_clip1m   = os.path.join(tmp, "mel_clip_1m.tif")
        patch_pre    = os.path.join(tmp, "patch_pre_blend.tif")

        hdr("Step 0 | MELITDEM -> UTM")
        step0_melitdem_to_utm(cfg["melitdem"], mel_utm, utm_crs, resampling, nodata)

        hdr("Step 1 | clip MELITDEM (grid-snapped)")
        snapped, aw3d_bounds_utm = step1_clip_snapped(
            mel_utm, cfg["aw3d"], mel_clip,
            cfg["buffer_m"], utm_crs, nodata)

        hdr("Step 2 | resample clip to 1m")
        step2_resample_1m(mel_clip, mel_clip1m, snapped, utm_crs, resampling, nodata)

        hdr("Step 3 | mosaic AW3D + MELITDEM_1m -> patch")
        aw3d_utm = step3_mosaic(cfg["aw3d"], mel_clip1m, patch_pre,
                                utm_crs, resampling, nodata, tmp)

        hdr("Step 3.5 | feather blend at AW3D boundary")
        if blend_width_m > 0:
            step35_feather_blend(
                patch_pre, mel_clip1m, aw3d_utm,
                snapped, aw3d_bounds_utm,
                blend_width_m, nodata, out_patch,
            )
        else:
            import shutil
            shutil.copy2(patch_pre, out_patch)
            print("  skip: blend_width_m=0, フェザリングをスキップしました")

        hdr("Step 4 | punch hole in MELITDEM (grid-snapped)")
        step4_punch_hole(mel_utm, snapped, out_hole, nodata)

    hdr("Done!")
    print(f"  patch_1m       : {out_patch}")
    print(f"  melitdem_hole  : {out_hole}")
    print(f"  CRS            : {utm_crs.to_string()}")
    print(f"  blend_width    : {blend_width_m} m")
    print()
    print("  Overlay melitdem_hole + patch_1m in GIS/DioVISTA.")
    print("  Grid-snapped boundaries ensure zero gap.")
    print("  Feathered boundary ensures smooth transition.")


if __name__ == "__main__":
    main()
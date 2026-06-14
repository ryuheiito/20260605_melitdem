"""
merge_dem.py
============
MELITDEM (90m) を UTM に変換し、全域をタイル分割して指定解像度で出力する。
AW3D (1m) 範囲のタイルは AW3D と結合（フェザリングブレンド）して出力する。

【処理フロー】
  Step 0 : MELITDEM -> UTM 変換
  Step 1 : AW3D 範囲 + バッファをグリッドスナップしてクリップ
  Step 2 : クリップ片を出力解像度にリサンプリング
  Step 3 : AW3D -> UTM 変換 → AW3D 優先モザイク
  Step 3.5: バッファ帯フェザリングブレンド → AW3D パッチ完成
  Step 4 : MELITDEM 全域をタイル分割して出力解像度で書き出し
           AW3D パッチと重なるタイルはパッチで上書き

【出力ファイル】
  <output>/tile_R{row:03d}_C{col:03d}.tif  : 各タイル（解像度 = resolution_m）
  AW3D 範囲タイルは AW3D+MELITDEM 結合済み

【config.yaml】
  melitdem:      path/to/melitdem.tif
  aw3d:          path/to/aw3d.tif
  output:        path/to/output_dir        # ディレクトリ（自動作成）
  utm_epsg:      32654                     # 省略時は自動検出
  buffer_m:      500.0
  blend_width_m: 200.0                     # 省略時 buffer_m * 0.4
  tile_size_m:   5000.0                    # タイル一辺 [m]
  resolution_m:  1.0                       # 出力解像度 [m]
  nodata:        -9999.0
  resampling:    bilinear

【使い方】
  python merge_all.py config.yaml
  python merge_all.py config.yaml --tile_size_m 5000 --resolution_m 1

【依存ライブラリ】
  pip install rasterio numpy shapely pyyaml scipy
"""

import argparse
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge
from rasterio.transform import from_bounds as tf_from_bounds
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
    ("output",        None,       str,   "output directory"),
    ("utm_epsg",      None,       str,   "UTM EPSG (auto if omitted)"),
    ("buffer_m",      500.0,      float, "buffer around AW3D [m]"),
    ("blend_width_m", None,       float, "feather blend width [m] (default: buffer_m*0.4)"),
    ("tile_size_m",   None,       float, "tile size [m] (auto if omitted)"),
    ("resolution_m",  1.0,        float, "output resolution [m]"),
    ("max_file_gb",   1.0,        float, "max tile file size [GB] (used when tile_size_m omitted)"),
    ("nodata",        -9999.0,    float, "NoData value"),
    ("resampling",    "bilinear", str,   "resampling method"),
    ("compress",      "deflate",  str,   "TIFF compression (deflate, lzw, zstd, none)"),
]


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
        print(f"    bounds : {b.left:.2f}, {b.bottom:.2f}, {b.right:.2f}, {b.top:.2f}")
        print(f"    nodata : {ds.nodata}")


def auto_utm_epsg(path):
    with rasterio.open(path) as ds:
        b = ds.bounds
        if ds.crs.is_geographic:
            cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        else:
            wb = transform_bounds(ds.crs, CRS.from_epsg(4326), *b)
            cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
    zone = int((cx + 180) / 6) + 1
    epsg = (32600 if cy >= 0 else 32700) + zone
    print(f"  UTM auto: center ({cx:.3f}, {cy:.3f}) -> Zone {zone} -> EPSG:{epsg}")
    return str(epsg)


def warp_to_crs(src_path, dst_path, dst_crs, resolution_m, resampling, nodata):
    with rasterio.open(src_path) as src:
        kw = {"resolution": (resolution_m, resolution_m)} if resolution_m else {}
        tf, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, **kw)
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


def resample_region(src_path, out_path, bounds, utm_crs, resolution_m, nodata):
    """bounds 範囲を resolution_m で切り出す（MELITDEM専用: nearest固定）"""
    left, bottom, right, top = bounds
    width  = max(1, round((right - left)  / resolution_m))
    height = max(1, round((top   - bottom) / resolution_m))
    tf = tf_from_bounds(left, bottom, right, top, width, height)

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
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata, dst_nodata=nodata,
                )


# --------------------------------------------------
# Step 0: MELITDEM -> UTM
# --------------------------------------------------

def step0_melitdem_to_utm(melitdem_path, out_path, utm_crs, nodata):
    # MELITDEM は粗い格子データなので nearest で元のセル値を保持する
    warp_to_crs(melitdem_path, out_path, utm_crs,
                resolution_m=None, resampling=Resampling.nearest, nodata=nodata)
    print("  ok: MELITDEM -> UTM (nearest)")
    info(out_path, "melitdem_utm")


# --------------------------------------------------
# Step 1: clip MELITDEM around AW3D (grid-snapped)
# --------------------------------------------------

def step1_clip_snapped(melitdem_utm_path, aw3d_path, out_path,
                       buffer_m, utm_crs, nodata):
    with rasterio.open(aw3d_path) as aw:
        aw3d_bounds_utm = transform_bounds(aw.crs, utm_crs, *aw.bounds)
    left, bottom, right, top = aw3d_bounds_utm
    left -= buffer_m; bottom -= buffer_m
    right += buffer_m; top += buffer_m

    with rasterio.open(melitdem_utm_path) as dem:
        res = abs(dem.transform.a)
        ox, oy = dem.transform.c, dem.transform.f
        left_s  = ox + math.floor((left  - ox) / res) * res
        right_s = ox + math.ceil( (right - ox) / res) * res
        top_s   = oy - math.floor((oy - top)    / res) * res
        bot_s   = oy - math.ceil( (oy - bottom) / res) * res
        b = dem.bounds
        left_s = max(left_s, b.left);  bot_s  = max(bot_s,  b.bottom)
        right_s= min(right_s,b.right); top_s  = min(top_s,  b.top)

        clip_geom = box(left_s, bot_s, right_s, top_s)
        out_img, out_tf = rio_mask(dem, [mapping(clip_geom)],
                                   crop=True, nodata=nodata, filled=True)
        meta = dem.meta.copy()
        meta.update({"transform": out_tf,
                     "width": out_img.shape[2], "height": out_img.shape[1],
                     "nodata": nodata, "dtype": "float32"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(out_img.astype("float32"))

    snapped = (left_s, bot_s, right_s, top_s)
    print(f"  ok: AW3D clip (buffer={buffer_m}m, grid-snapped)")
    return snapped, aw3d_bounds_utm


# --------------------------------------------------
# Step 2: resample AW3D clip to resolution_m
# --------------------------------------------------

def step2_resample(src_path, out_path, snapped_bounds, utm_crs,
                   resolution_m, nodata):
    resample_region(src_path, out_path, snapped_bounds,
                    utm_crs, resolution_m, nodata)
    print(f"  ok: resample clip to {resolution_m}m")
    info(out_path, "melitdem_clip_resampled")


# --------------------------------------------------
# Step 3: AW3D -> UTM, mosaic (AW3D priority)
# --------------------------------------------------

def step3_mosaic(aw3d_path, mel_resampled_path, out_path,
                 utm_crs, resolution_m, resampling, nodata, tmpdir):
    aw3d_utm = os.path.join(tmpdir, "aw3d_utm.tif")
    warp_to_crs(aw3d_path, aw3d_utm, utm_crs,
                resolution_m=resolution_m, resampling=resampling, nodata=nodata)
    print(f"  ok: AW3D -> UTM ({resolution_m}m)")

    with rasterio.open(aw3d_utm) as hi, rasterio.open(mel_resampled_path) as lo:
        mosaic, tf = merge([hi, lo], nodata=nodata, method="first")
        meta = hi.meta.copy()
        meta.update({"width": mosaic.shape[2], "height": mosaic.shape[1],
                     "transform": tf, "nodata": nodata, "dtype": "float32"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic.astype("float32"))
    print("  ok: mosaic -> AW3D patch (pre-blend)")
    return aw3d_utm


# --------------------------------------------------
# Step 3.5: 距離加重ブレンド (AW3D境界から blend_width_m 幅)
# --------------------------------------------------

def step35_feather_blend(patch_path, mel_resampled_path, aw3d_utm_path,
                         snapped_bounds, aw3d_bounds_utm,
                         blend_width_m, resolution_m, nodata, out_path):
    """
    AW3D 領域はそのまま維持し、境界外側の blend_width_m [m] 幅の帯を
    距離に応じた重み付き平均でブレンドする。

    距離 d（AW3D境界からの距離）に対して：
        weight = 1 - smootherstep(d / blend_width_m)
        result = weight * nearest_aw3d_val + (1 - weight) * mel_val

    つまり：
        d = 0  (境界直後) → AW3D境界値と元値の平均に近い (weight≈0.5付近)
        d = blend_width_m (帯の外端) → 元のMELITDEM値をそのまま使用 (weight=0)
    ※ smootherstep により中間領域が滑らかに遷移する。
    """
    from scipy.ndimage import binary_dilation
    from rasterio.warp import reproject as rio_reproject, Resampling as RS

    blend_px = max(1, int(blend_width_m / resolution_m))

    # ---- patch (AW3D優先モザイク) と MELITDEM を読み込む ----
    with rasterio.open(patch_path) as ds:
        patch_tf         = ds.transform
        patch_data       = ds.read(1).astype("float32")
        patch_w, patch_h = ds.width, ds.height
        patch_crs        = ds.crs

    with rasterio.open(mel_resampled_path) as ds:
        mel_data = ds.read(1).astype("float32")

    # ---- AW3D の「真の有効域」をパッチグリッドに投影してマスク作成 ----
    import tempfile as _tf_mod
    with _tf_mod.NamedTemporaryFile(suffix=".tif", delete=False) as _tmp:
        aw3d_reproj_path = _tmp.name
    try:
        with rasterio.open(aw3d_utm_path) as aw_src:
            dst_meta = {
                "driver": "GTiff", "dtype": "float32",
                "width": patch_w, "height": patch_h,
                "count": 1, "crs": patch_crs,
                "transform": patch_tf, "nodata": nodata,
            }
            with rasterio.open(aw3d_reproj_path, "w", **dst_meta) as dst:
                rio_reproject(
                    source=rasterio.band(aw_src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=aw_src.transform, src_crs=aw_src.crs,
                    dst_transform=patch_tf, dst_crs=patch_crs,
                    resampling=RS.bilinear,
                    src_nodata=aw_src.nodata, dst_nodata=nodata,
                )
        with rasterio.open(aw3d_reproj_path) as ds:
            aw3d_data = ds.read(1).astype("float32")
    finally:
        try:
            os.unlink(aw3d_reproj_path)
        except OSError:
            pass

    aw3d_mask   = aw3d_data  != nodata   # AW3D 有効域
    patch_valid = patch_data != nodata
    mel_valid   = mel_data   != nodata

    if not aw3d_mask.any():
        shutil.copy2(patch_path, out_path)
        print("  skip: AW3D 有効ピクセルなし")
        return

    # ---- AW3D境界からの距離マップ（ピクセル単位）----
    aw3d_boundary    = aw3d_mask & patch_valid
    dist_px, nearest_idx = distance_transform_edt(~aw3d_boundary, return_indices=True)

    # ---- ブレンド帯マスク: AW3D外側 かつ blend_px 以内 ----
    blend_zone = ~aw3d_mask & mel_valid & patch_valid & (dist_px <= blend_px)

    # ---- 各ピクセルの最近傍AW3D境界値 ----
    nearest_aw3d_val = patch_data[nearest_idx[0], nearest_idx[1]]

    # ---- 距離に応じた重み (smootherstep: 境界側=1 → 外端=0) ----
    t = np.clip(dist_px[blend_zone] / blend_px, 0.0, 1.0).astype("float32")
    # smootherstep: 6t^5 - 15t^4 + 10t^3
    t_smooth = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    weight = 1.0 - t_smooth   # 境界直後=1(AW3D寄り), 外端=0(MELITDEM寄り)

    # ---- ブレンド適用 ----
    result = patch_data.copy()
    result[blend_zone] = (
        weight * nearest_aw3d_val[blend_zone]
        + (1.0 - weight) * mel_data[blend_zone]
    )
    result[~patch_valid] = nodata

    with rasterio.open(patch_path) as src:
        meta = src.meta.copy()
        meta.update({"nodata": nodata, "dtype": "float32"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(result[np.newaxis, :, :])

    n_zone = int(blend_zone.sum())
    print(f"  ok: 距離加重ブレンド  幅={blend_width_m}m ({blend_px}px)  対象ピクセル数={n_zone:,}")
    info(out_path, "aw3d_patch (blended)")


# --------------------------------------------------
# Step 4: tile MELITDEM + overwrite AW3D patch tiles
# --------------------------------------------------

def step4_tile_output(melitdem_utm_path, aw3d_patch_path,
                      out_dir, tile_size_m, resolution_m,
                      utm_crs, resampling, nodata, compress, tmpdir):
    """
    MELITDEM 全域をタイル分割して resolution_m で出力。
    AW3D パッチと重なるタイルはパッチ優先でモザイクして上書き。

    【隙間ゼロ保証】
    タイル境界を resolution_m グリッドにスナップして生成する。
    各タイルは from_bounds() で transform を直接計算するため
    浮動小数点誤差による隙間・重複が発生しない。
    隣接タイルの境界ピクセルは共有せず、左閉右開 [x0, x1) で定義する。
    """
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(melitdem_utm_path) as dem:
        b = dem.bounds

    with rasterio.open(aw3d_patch_path) as patch_ds:
        pb = patch_ds.bounds

    res = resolution_m  # 出力解像度

    # ── 圧縮オプションを共通辞書として準備 ──
    compress_opts = {}
    if compress and compress.lower() != "none":
        compress_opts = {"compress": compress, "predictor": 3}  # float32用predictor

    # ── タイルグリッドを resolution_m にスナップして生成 ──
    # 全体の左下を原点として、ピクセル数単位でタイル境界を決める
    tile_px = int(round(tile_size_m / res))   # タイル1辺のピクセル数

    # 全体ピクセル数（端数切り上げ）
    total_w = math.ceil((b.right  - b.left) / res)
    total_h = math.ceil((b.top    - b.bottom) / res)

    n_cols = math.ceil(total_w / tile_px)
    n_rows = math.ceil(total_h / tile_px)
    total  = n_rows * n_cols

    print(f"  全体サイズ: {total_w} × {total_h} px @ {res}m")
    print(f"  タイル数  : {n_rows} rows × {n_cols} cols = {total} tiles")
    print(f"  タイルサイズ: {tile_px} × {tile_px} px ({tile_size_m}m)")
    print(f"  圧縮方式  : {compress if compress and compress.lower() != 'none' else '非圧縮'}")

    done = 0
    for ri in range(n_rows):
        for ci in range(n_cols):
            # ピクセルインデックスでタイル範囲を定義（左閉右開）
            px0 = ci * tile_px
            py0 = ri * tile_px
            px1 = min(px0 + tile_px, total_w)
            py1 = min(py0 + tile_px, total_h)

            # 座標に変換（resolution_m グリッドに完全スナップ済み）
            x0 = b.left   + px0 * res
            x1 = b.left   + px1 * res
            y0 = b.bottom + py0 * res
            y1 = b.bottom + py1 * res
            tile_bounds = (x0, y0, x1, y1)

            # タイル名: 行は下から上（ri=0が南端）
            tile_name = f"tile_R{ri:03d}_C{ci:03d}.tif"
            tile_path = os.path.join(out_dir, tile_name)

            # MELITDEM タイルを resolution_m でリサンプリング
            mel_tile = os.path.join(tmpdir, f"mel_{ri}_{ci}.tif")
            resample_region(melitdem_utm_path, mel_tile, tile_bounds,
                            utm_crs, resolution_m, nodata)

            # AW3D パッチと重なるか判定
            overlap = not (pb.right <= x0 or pb.left >= x1 or
                           pb.top   <= y0 or pb.bottom >= y1)

            if overlap:
                # mel_tile の transform を基準にパッチをはめ込む
                # → merge の bounds/res は mel_tile と完全一致させる
                with rasterio.open(mel_tile) as lo_ds:
                    lo_tf  = lo_ds.transform
                    lo_w   = lo_ds.width
                    lo_h   = lo_ds.height
                    lo_meta = lo_ds.meta.copy()

                with rasterio.open(aw3d_patch_path) as hi, \
                     rasterio.open(mel_tile) as lo:
                    mosaic, tf = merge(
                        [hi, lo], nodata=nodata, method="first",
                        bounds=(lo_tf.c,
                                lo_tf.f + lo_tf.e * lo_h,  # bottom
                                lo_tf.c + lo_tf.a * lo_w,  # right
                                lo_tf.f),                   # top
                        res=res,
                    )
                lo_meta.update({"width": mosaic.shape[2], "height": mosaic.shape[1],
                                "transform": tf, "nodata": nodata, "dtype": "float32"})
                lo_meta.update(compress_opts)
                with rasterio.open(tile_path, "w", **lo_meta) as dst:
                    dst.write(mosaic.astype("float32"))
                os.remove(mel_tile)
                marker = " [AW3D]"
            else:
                if compress_opts:
                    # 圧縮設定がある場合は再エンコードして保存
                    with rasterio.open(mel_tile) as lo_ds:
                        lo_img  = lo_ds.read(1)
                        lo_meta = lo_ds.meta.copy()
                    lo_meta.update({"nodata": nodata, "dtype": "float32"})
                    lo_meta.update(compress_opts)
                    with rasterio.open(tile_path, "w", **lo_meta) as dst:
                        dst.write(lo_img.astype("float32"), 1)
                    os.remove(mel_tile)
                else:
                    shutil.move(mel_tile, tile_path)
                marker = ""

            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] {tile_name}{marker}")

    print(f"  ok: {total} タイル出力完了 -> {out_dir}")


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
        print(f"[ERROR] missing required keys: {missing}"); sys.exit(1)
    if cfg["resampling"] not in RESAMPLING_MAP:
        print(f"[ERROR] invalid resampling: {cfg['resampling']}"); sys.exit(1)
    if cfg["blend_width_m"] is None:
        cfg["blend_width_m"] = cfg["buffer_m"] * 0.4
    # tile_size_m が未指定なら max_file_gb から逆算
    if cfg["tile_size_m"] is None:
        res        = cfg["resolution_m"]
        bytes_per_px = 4  # float32
        max_bytes  = cfg["max_file_gb"] * 1024 ** 3
        max_px_side = math.floor(math.sqrt(max_bytes / bytes_per_px))
        cfg["tile_size_m"] = max_px_side * res
        print(f"  tile_size_m 自動計算: {cfg['max_file_gb']}GB上限 "
              f"-> {max_px_side}px -> {cfg['tile_size_m']:.0f}m")
    return cfg


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tile MELITDEM at target resolution, merging AW3D patch")
    parser.add_argument("config")
    parser.add_argument("--melitdem",      default=None)
    parser.add_argument("--aw3d",          default=None)
    parser.add_argument("--output",        default=None)
    parser.add_argument("--utm_epsg",      default=None)
    parser.add_argument("--buffer_m",      default=None, type=float)
    parser.add_argument("--blend_width_m", default=None, type=float)
    parser.add_argument("--tile_size_m",   default=None, type=float,
                        help="tile size [m] (auto from max_file_gb if omitted)")
    parser.add_argument("--resolution_m",  default=None, type=float)
    parser.add_argument("--max_file_gb",   default=None, type=float,
                        help="max tile file size [GB] (default: 1.0)")
    parser.add_argument("--nodata",        default=None, type=float)
    parser.add_argument("--resampling",    default=None, choices=RESAMPLING_MAP.keys())
    parser.add_argument("--compress",      default=None,
                        help="TIFF compression: deflate, lzw, zstd, none (default: deflate)")
    args = parser.parse_args()

    cfg = load_config(args.config,
                      {k: getattr(args, k) for k in
                       ("melitdem", "aw3d", "output", "utm_epsg",
                        "buffer_m", "blend_width_m", "tile_size_m",
                        "resolution_m", "max_file_gb", "nodata", "resampling",
                        "compress")})

    resampling    = RESAMPLING_MAP[cfg["resampling"]]
    nodata        = cfg["nodata"]
    blend_width_m = cfg["blend_width_m"]
    resolution_m  = cfg["resolution_m"]
    tile_size_m   = cfg["tile_size_m"]
    out_dir       = cfg["output"]
    compress      = cfg["compress"]

    hdr("Input files")
    info(cfg["melitdem"], "MELITDEM")
    info(cfg["aw3d"],     "AW3D")

    hdr("Settings")
    print(f"  resolution_m  : {resolution_m} m")
    tile_px_side = int(tile_size_m / resolution_m)
    tile_size_gb = (tile_px_side ** 2 * 4) / 1024 ** 3
    print(f"  tile_size_m   : {tile_size_m:.0f} m  "
          f"({tile_px_side}×{tile_px_side} px, ~{tile_size_gb:.2f} GB/tile 非圧縮時)")
    print(f"  max_file_gb   : {cfg['max_file_gb']} GB")
    print(f"  buffer_m      : {cfg['buffer_m']} m")
    print(f"  blend_width_m : {blend_width_m} m")
    print(f"  compress      : {compress if compress and compress.lower() != 'none' else '非圧縮'}")

    hdr("UTM zone")
    if cfg.get("utm_epsg"):
        utm_crs = CRS.from_epsg(int(cfg["utm_epsg"]))
        print(f"  manual: EPSG:{cfg['utm_epsg']}")
    else:
        utm_crs = CRS.from_epsg(int(auto_utm_epsg(cfg["melitdem"])))

    with tempfile.TemporaryDirectory() as tmp:
        mel_utm       = os.path.join(tmp, "mel_utm.tif")
        mel_clip      = os.path.join(tmp, "mel_clip.tif")
        mel_clip_res  = os.path.join(tmp, "mel_clip_resampled.tif")
        patch_pre     = os.path.join(tmp, "patch_pre_blend.tif")
        patch_final   = os.path.join(tmp, "aw3d_patch.tif")

        hdr("Step 0 | MELITDEM -> UTM")
        step0_melitdem_to_utm(cfg["melitdem"], mel_utm, utm_crs, nodata)

        hdr("Step 1 | clip MELITDEM around AW3D (grid-snapped)")
        snapped, aw3d_bounds_utm = step1_clip_snapped(
            mel_utm, cfg["aw3d"], mel_clip,
            cfg["buffer_m"], utm_crs, nodata)

        hdr(f"Step 2 | resample clip to {resolution_m}m")
        step2_resample(mel_clip, mel_clip_res, snapped,
                       utm_crs, resolution_m, nodata)

        hdr("Step 3 | mosaic AW3D + MELITDEM -> patch")
        aw3d_utm = step3_mosaic(cfg["aw3d"], mel_clip_res, patch_pre,
                                utm_crs, resolution_m, resampling, nodata, tmp)

        hdr("Step 3.5 | 隣接1メッシュ平均化 at AW3D boundary")
        step35_feather_blend(
            patch_pre, mel_clip_res, aw3d_utm,
            snapped, aw3d_bounds_utm,
            blend_width_m, resolution_m, nodata, patch_final,
        )

        hdr(f"Step 4 | tile MELITDEM + overwrite AW3D tiles -> {out_dir}")
        step4_tile_output(
            mel_utm, patch_final,
            out_dir, tile_size_m, resolution_m,
            utm_crs, resampling, nodata, compress, tmp,
        )

    hdr("Done!")
    print(f"  出力ディレクトリ : {out_dir}")
    print(f"  解像度           : {resolution_m} m")
    print(f"  タイルサイズ     : {tile_size_m} m")
    print(f"  AW3D境界         : 隣接1メッシュ平均化済み")


if __name__ == "__main__":
    main()
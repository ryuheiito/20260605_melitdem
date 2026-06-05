"""
make_pseudo_aw3d.py
===================
MERIT DEM（90m）の一部を切り取り、疑似的な 1m メッシュ DEM を生成するスクリプト。
merge_dem.py のテスト用途として使用する。

【やること】
  1. MERIT DEM から指定範囲（中心点 + サイズ）を切り出す
  2. 1m 解像度にアップサンプリング
  3. 地形らしいランダムノイズを加算（AW3D との差異を再現）
  4. 疑似 AW3D として保存

【使い方】
  python make_pseudo_aw3d.py \\
      --dem     path/to/merit_dem.tif \\
      --output  path/to/pseudo_aw3d.tif \\
      [--center_lon 134.8] \\
      [--center_lat  34.7] \\
      [--size_m    2000  ] \\
      [--noise_std  0.3  ] \\
      [--seed       42   ]

  # または YAML で指定
  python make_pseudo_aw3d.py config_pseudo.yaml

【依存ライブラリ】
  pip install rasterio numpy scipy pyyaml
"""

import argparse
import math
import os
import sys

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, mapping


# ─────────────────────────────────────────────
# 設定読み込み
# ─────────────────────────────────────────────

DEFAULTS = {
    "center_lon": None,     # 中心経度（省略時は DEM 中心）
    "center_lat": None,     # 中心緯度（省略時は DEM 中心）
    "size_m":     2000.0,   # 切り出しサイズ [m]（正方形）
    "noise_std":  0.3,      # ノイズの標準偏差 [m]（AW3D の計測誤差を模擬）
    "noise_scale": 50,      # ノイズの空間スケール [px]（地形起伏の粗さ）
    "seed":       42,       # 乱数シード
    "nodata":     -9999.0,
    "resampling": "bilinear",
}

RESAMPLING_MAP = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic":    Resampling.cubic,
}


def load_config(yaml_path: str, cli_overrides: dict) -> dict:
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in raw.items() if k in DEFAULTS or k in ("dem", "output")})
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})
    for key in ("dem", "output"):
        if not cfg.get(key):
            print(f"[ERROR] 必須項目 '{key}' が設定されていません。")
            sys.exit(1)
    return cfg


def print_header(text: str) -> None:
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


# ─────────────────────────────────────────────
# Step 1: 中心点 + サイズで切り出し
# ─────────────────────────────────────────────

def clip_region(dem_path: str, center_lon: float, center_lat: float,
                size_m: float, nodata: float) -> tuple:
    """
    指定した中心点から size_m × size_m の範囲を切り出す。
    (array, transform, crs) を返す。
    """
    with rasterio.open(dem_path) as ds:
        crs = ds.crs
        bounds = ds.bounds

        # center がなければ DEM 中心を使用
        if center_lon is None:
            center_lon = (bounds.left + bounds.right) / 2
        if center_lat is None:
            center_lat = (bounds.bottom + bounds.top) / 2

        # サイズを度に変換（地理座標系の場合）
        if crs.is_geographic:
            half_lat = (size_m / 2) / 111_320
            half_lon = (size_m / 2) / (111_320 * math.cos(math.radians(center_lat)))
        else:
            half_lat = half_lon = size_m / 2

        clip_geom = box(
            center_lon - half_lon, center_lat - half_lat,
            center_lon + half_lon, center_lat + half_lat,
        ).intersection(box(*bounds))

        out_image, out_transform = rio_mask(
            ds, [mapping(clip_geom)],
            crop=True, nodata=nodata, filled=True,
        )

        print(f"  切り出し完了: {out_image.shape[2]} × {out_image.shape[1]} px "
              f"@ {abs(ds.transform.a)*111320:.0f}m 解像度")
        return out_image.astype("float32"), out_transform, crs


# ─────────────────────────────────────────────
# Step 2: 1m にリサンプリング
# ─────────────────────────────────────────────

def resample_to_1m(src_array: np.ndarray, src_transform, src_crs,
                   target_res_m: float, resampling: Resampling,
                   nodata: float) -> tuple:
    """
    src_array を target_res_m [m] の解像度にリサンプリングする。
    (array, transform) を返す。
    """
    # 現在の解像度を取得
    if src_crs.is_geographic:
        # 度 → メートル概算
        src_res_m = abs(src_transform.a) * 111_320
        target_res_deg = target_res_m / 111_320
    else:
        src_res_m = abs(src_transform.a)
        target_res_deg = target_res_m

    scale = src_res_m / target_res_m
    h, w  = src_array.shape[1], src_array.shape[2]
    new_h = int(h * scale)
    new_w = int(w * scale)

    # 新しい transform（左上隅はそのまま、ピクセルサイズのみ変更）
    new_transform = src_transform * src_transform.scale(
        w / new_w, h / new_h
    )

    dst = np.empty((src_array.shape[0], new_h, new_w), dtype="float32")

    reproject(
        source        = src_array,
        destination   = dst,
        src_transform = src_transform,
        src_crs       = src_crs,
        dst_transform = new_transform,
        dst_crs       = src_crs,
        resampling    = resampling,
        src_nodata    = nodata,
        dst_nodata    = nodata,
    )

    print(f"  リサンプリング完了: {new_w} × {new_h} px @ 1m 解像度")
    return dst, new_transform, src_crs


# ─────────────────────────────────────────────
# Step 3: 地形ノイズを加算
# ─────────────────────────────────────────────

def add_terrain_noise(array: np.ndarray, nodata: float,
                      noise_std: float, noise_scale: int,
                      seed: int) -> np.ndarray:
    """
    地形らしいノイズを加える。
    ・大スケール成分 (gaussian_filter) で地形起伏を模擬
    ・小スケール成分でセンサーノイズを模擬
    有効値ピクセルにのみ適用。
    """
    rng   = np.random.default_rng(seed)
    result = array.copy()

    for b in range(array.shape[0]):
        band     = result[b]
        valid    = band != nodata

        raw_noise = rng.normal(0, noise_std, band.shape).astype("float32")

        # 空間的に相関したノイズ（地形起伏の凸凹感）
        smooth = gaussian_filter(raw_noise, sigma=noise_scale).astype("float32")
        # 正規化して std を揃える
        if smooth.std() > 0:
            smooth = smooth / smooth.std() * noise_std * 0.7
        # 高周波ノイズ（センサーノイズ）
        hf_noise = (raw_noise * 0.3).astype("float32")

        band[valid] += smooth[valid] + hf_noise[valid]

    print(f"  ノイズ加算完了（std={noise_std}m, scale={noise_scale}px, seed={seed}）")
    return result


# ─────────────────────────────────────────────
# Step 4: 保存
# ─────────────────────────────────────────────

def save_raster(array: np.ndarray, transform, crs,
                output_path: str, nodata: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     array.shape[2],
        "height":    array.shape[1],
        "count":     array.shape[0],
        "crs":       crs,
        "transform": transform,
        "nodata":    nodata,
        "compress":  "lzw",
    }
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array)

    with rasterio.open(output_path) as ds:
        b = ds.bounds
        print(f"  保存完了: {output_path}")
        print(f"    サイズ : {ds.width} × {ds.height} px")
        print(f"    Bounds : {b.left:.6f}, {b.bottom:.6f}, {b.right:.6f}, {b.top:.6f}")
        print(f"    解像度 : {abs(ds.transform.a):.8f} deg ({abs(ds.transform.a)*111320:.2f} m)")


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MERIT DEM から疑似 1m AW3D データを生成"
    )
    parser.add_argument("config",        nargs="?", default=None,
                        help="YAML 設定ファイル（省略時は CLI 引数のみ）")
    parser.add_argument("--dem",         default=None, help="入力 MERIT DEM パス")
    parser.add_argument("--output",      default=None, help="出力 GeoTIFF パス")
    parser.add_argument("--center_lon",  default=None, type=float, help="中心経度")
    parser.add_argument("--center_lat",  default=None, type=float, help="中心緯度")
    parser.add_argument("--size_m",      default=None, type=float,
                        help="切り出しサイズ [m]（デフォルト: 2000）")
    parser.add_argument("--noise_std",   default=None, type=float,
                        help="ノイズ標準偏差 [m]（デフォルト: 0.3）")
    parser.add_argument("--noise_scale", default=None, type=int,
                        help="ノイズ空間スケール [px]（デフォルト: 50）")
    parser.add_argument("--seed",        default=None, type=int,
                        help="乱数シード（デフォルト: 42）")
    parser.add_argument("--nodata",      default=None, type=float)
    parser.add_argument("--resampling",  default=None,
                        choices=RESAMPLING_MAP.keys())
    args = parser.parse_args()

    cli = {k: getattr(args, k) for k in
           ("dem", "output", "center_lon", "center_lat", "size_m",
            "noise_std", "noise_scale", "seed", "nodata", "resampling")}

    if args.config:
        cfg = load_config(args.config, cli)
    else:
        cfg = dict(DEFAULTS)
        cfg.update({k: v for k, v in cli.items() if v is not None})
        for key in ("dem", "output"):
            if not cfg.get(key):
                parser.error(f"--{key} が必要です（または YAML ファイルを指定）")

    resampling = RESAMPLING_MAP[cfg["resampling"]]

    print_header("疑似 AW3D（1m）生成")
    print(f"  入力 DEM  : {cfg['dem']}")
    print(f"  出力      : {cfg['output']}")
    print(f"  中心座標  : lon={cfg['center_lon']}, lat={cfg['center_lat']}")
    print(f"  サイズ    : {cfg['size_m']} m × {cfg['size_m']} m")
    print(f"  ノイズ    : std={cfg['noise_std']}m, scale={cfg['noise_scale']}px")

    print_header("Step 1 | 範囲切り出し")
    arr, tf, crs = clip_region(
        cfg["dem"],
        cfg["center_lon"], cfg["center_lat"],
        cfg["size_m"], cfg["nodata"],
    )

    print_header("Step 2 | 1m リサンプリング")
    arr_1m, tf_1m, crs = resample_to_1m(
        arr, tf, crs, 1.0, resampling, cfg["nodata"]
    )

    print_header("Step 3 | 地形ノイズ加算")
    arr_noisy = add_terrain_noise(
        arr_1m, cfg["nodata"],
        cfg["noise_std"], cfg["noise_scale"], cfg["seed"],
    )

    print_header("Step 4 | 保存")
    save_raster(arr_noisy, tf_1m, crs, cfg["output"], cfg["nodata"])

    print_header("完了！")
    print(f"  疑似 AW3D: {cfg['output']}")
    print("  → merge_dem.py の --aw3d に指定してテストできます。")


if __name__ == "__main__":
    main()

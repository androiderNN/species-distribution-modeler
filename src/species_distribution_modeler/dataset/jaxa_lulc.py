import os
from pathlib import Path
import math
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point


LULC_CATEGORIES_EN = {
    1: 'Water',
    2: 'Built-up',
    3: 'Paddy field',
    4: 'Cropland',
    5: 'Grassland',
    6: 'DBF (Deciduous broad-leaf)',
    7: 'DNF (Deciduous needle-leaf)',
    8: 'EBF (Evergreen broad-leaf)',
    9: 'ENF (Evergreen needle-leaf)',
    10: 'Bare',
    11: 'Bamboo forest',
    12: 'Solar panel',
    13: 'Wetland',
    14: 'Greenhouse',
    15: 'Rock reef / Tidal flat',
}

LULC_CATEGORIES_JP = {
    1: '水域',
    2: '市街地',
    3: '水田',
    4: '畑地',
    5: '草地',
    6: '落葉広葉樹',
    7: '落葉針葉樹',
    8: '常緑広葉樹',
    9: '常緑針葉樹',
    10: '裸地',
    11: '竹林',
    12: '太陽光パネル',
    13: '湿地',
    14: '温室',
    15: '岩礁・干潟',
}

LULC_SUMMARY_NAMES = [f"lulc_{n}" for n in LULC_CATEGORIES_EN.keys()]


def coord_to_tile_path(raster_dir: Path, lat: float, lon: float) -> Path:
    """
    (lat, lon) から対応するタイルファイルのパスを返す。
    ファイルが存在しない場合はエラー"""
    lat_floor = int(np.floor(lat))
    lon_floor = int(np.floor(lon))
    ns = 'N' if lat_floor >= 0 else 'S'
    ew = 'E' if lon_floor >= 0 else 'W'

    fname = f'LC_{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}.tif'
    tif_path = raster_dir / fname

    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"{tif_path} not found.")

    return tif_path


def sample_at_point(center: tuple[float, float], raster_dir: Path) -> float:
    """
    指定した座標の土地利用クラスIDを返す

    Args:
        center: 抽出地点の座標
        raster_dir: tifの保存先

    Returns:
        土地利用クラスID（1-15）。NoDataの場合は0を返す
    """
    lat, lon = center
    tif_path = coord_to_tile_path(raster_dir, lat, lon)

    with rasterio.open(tif_path) as src:
        row, col = src.index(lon, lat)
        val = int(src.read(1)[row, col])

    return val if val != 0 else 0.0


def summarize_in_circle(
        center: tuple[float, float],
        radius_meters: float,
        raster_dir: Path,
        crs: str = "EPSG:4326"
    ) -> npt.NDArray:
    """
    円形範囲で土地利用クラスを集計する（タイルまたぎ対応）

    Args:
        center: 中心座標（緯度, 経度）
        radius_meters: 半径（メートル）
        raster_dir: tifの保存先
        crs: 座標系
    Returns:
        result: 集計結果
    """
    lat, lon = center

    # 円ポリゴン作成（Point(x=lon, y=lat)）
    point = gpd.GeoSeries([Point((lon, lat))], crs=crs)
    utm_crs = point.estimate_utm_crs()
    circle = point.to_crs(utm_crs).buffer(radius_meters).to_crs(crs)
    geom = [circle.iloc[0]]

    # 円のバウンディングボックス（度）
    radius_deg = radius_meters / 80000
    lat_min = max(-90, lat - radius_deg)
    lat_max = min(90, lat + radius_deg)
    lon_min = lon - radius_deg
    lon_max = lon + radius_deg

    # 該当する全タイルを列挙してカウント
    counts = np.zeros(16, dtype=np.int64)

    for lat_idx, lon_idx in itertools.product(
        range(int(math.floor(lat_min)), int(math.floor(lat_max)) + 1),
        range(int(math.floor(lon_min)), int(math.floor(lon_max)) + 1)
    ):
        tif_path = coord_to_tile_path(raster_dir, lat_idx + 0.5, lon_idx + 0.5)

        with rasterio.open(tif_path) as src:
            out_image, _ = mask(src, geom, crop=True, all_touched=True)
            data = out_image[0]
            nodata = src.nodata

            if nodata is not None:
                valid = data[data != nodata]
            else:
                valid = data[data != 0]

            if valid.size > 0:
                counts += np.bincount(valid, minlength=16)

    # 割合に変換
    counts = counts / counts[1:].sum()
    return counts


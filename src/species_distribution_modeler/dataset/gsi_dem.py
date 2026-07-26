import zipfile
from pathlib import Path
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point
import xml.etree.ElementTree as ET


_NS = {"gml": "http://www.opengis.net/gml/3.2"}
# カバレッジインデックス（lazy build）
_coverage_index: list | None = None


def _build_coverage_index(raster_dir: Path) -> list:
    """
    全zipのカバレッジをスキャンしてインデックスを構築する
    各zipの先頭サブメッシュから地理範囲を読み取る

    Args:
        raster_dir: DSM zip ファイルのあるディレクトリ

    Returns:
        (zip_path, lat_min, lat_max, lon_min, lon_max, dlat, dlon) のリスト
    """
    global _coverage_index
    if _coverage_index is not None:
        return _coverage_index

    _coverage_index = []
    for zp in sorted(Path(raster_dir).glob("*.zip")):
        # 先頭サブメッシュを読んで地理範囲を取得
        with zipfile.ZipFile(zp) as z:
            root = ET.fromstring(z.read(sorted(z.namelist())[0]))

        env = root.find(".//gml:Envelope", _NS)
        lo = env.find("gml:lowerCorner", _NS).text.split()
        hi = env.find("gml:upperCorner", _NS).text.split()

        lat0, lon0 = float(lo[0]), float(lo[1])  # 南西端
        lat1, lon1 = float(hi[0]), float(hi[1])  # 北東端（1サブメッシュ分）
        dlat = lat1 - lat0  # 1サブメッシュの緯度幅（≈ 0.00833°）
        dlon = lon1 - lon0  # 1サブメッシュの経度幅（≈ 0.0125°）

        # zip全体の範囲 = サブメッシュ幅 × 10（10×10 のサブメッシュ grid）
        _coverage_index.append((
            zp, lat0, lat0 + dlat * 10, lon0, lon0 + dlon * 10, dlat, dlon
        ))

    return _coverage_index


def _read_submesh(zip_path, sub_idx: int) -> tuple[npt.NDArray, rasterio.Affine]:
    """
    zip内の1サブメッシュを読む
    1つのzipには 10x10=100 のサブメッシュ（XML）が含まれる
    sub_idx = row * 10 + col で特定する

    Args:
        zip_path: zip ファイルのパス
        sub_idx: サブメッシュインデックス（0-99）

    Returns:
        (elevation, transform)。elevation は (150, 225) の2次元配列
    """
    with zipfile.ZipFile(zip_path) as z:
        root = ET.fromstring(z.read(sorted(z.namelist())[sub_idx]))

    # 地理範囲を取得
    env = root.find(".//gml:Envelope", _NS)
    lo = env.find("gml:lowerCorner", _NS).text.split()
    hi = env.find("gml:upperCorner", _NS).text.split()
    lat_min, lon_min = float(lo[0]), float(lo[1])  # 南西端
    lat_max, lon_max = float(hi[0]), float(hi[1])  # 北東端

    # グリッドサイズ（225列 × 150行）
    high = root.find(".//gml:high", _NS).text.split()
    n_cols, n_rows = int(high[0]) + 1, int(high[1]) + 1

    # 標高値を"地表面,標高値"のリストからパース
    tuples = root.find(".//gml:tupleList", _NS)
    lines = tuples.text.strip().splitlines()
    elev = np.array([float(line.split(",")[1]) for line in lines])
    elev = elev.reshape(n_rows, n_cols)

    # rasterio transformを作成
    # Affine(セル経度幅, 0, 左上経度, 0, -セル緯度幅, 左上緯度)
    pw = (lon_max - lon_min) / n_cols
    ph = (lat_max - lat_min) / n_rows
    tr = rasterio.Affine(pw, 0, lon_min, 0, -ph, lat_max)

    return elev, tr


def sample_at_point(raster_dir: Path, lat: float, lon: float) -> float:
    """
    指定した座標の標高値を返す
    該当するzip → サブメッシュ → ピクセル と解決する

    Args:
        raster_dir: DSM zipファイルのあるディレクトリ
        lat: 緯度
        lon: 経度

    Returns:
        標高値（m）。値-9999は欠損（水域等）でNaNを返す
    """
    idx = _build_coverage_index(raster_dir)

    for zip_path, lat_min, lat_max, lon_min, lon_max, dlat, dlon in idx:
        # このzipが (lat, lon) をカバーするか判定
        if not (lat_min <= lat < lat_max and lon_min <= lon < lon_max):
            continue

        # サブメッシュ位置を計算（10×10 grid 内の i, j）
        si = int((lat - lat_min) / dlat)
        sj = int((lon - lon_min) / dlon)
        elev, tr = _read_submesh(zip_path, si * 10 + sj)

        # 標高値のピクセル座標に変換
        col = int((lon - tr.c) / tr.a)
        row = int((tr.f - lat) / -tr.e)
        row = min(max(row, 0), elev.shape[0] - 1)
        col = min(max(col, 0), elev.shape[1] - 1)

        val = float(elev[row, col])
        return val if val != -9999 else float("nan")

    raise ValueError(f"No DEM data at ({lat}, {lon})")


def summarize_in_circle(
    center: tuple[float, float],
    radius_meters: float,
    raster_dir: Path,
    crs: str = "EPSG:4326",
) -> list[float]:
    """
    円形範囲内の標高を集計する（タイルまたぎ対応）
    円ポリゴンを作成し、交差する全サブメッシュを収集してマスク
    有効ピクセルの統計量を計算する

    Args:
        center: 中心座標（緯度, 経度）
        radius_meters: 半径（メートル）
        raster_dir: DSM zip ファイルのあるディレクトリ
        crs: 入力座標のCRS。デフォルトは EPSG:4326

    Returns:
        [mean, min, max, std] の順のリスト
        有効データがない場合は全要素が NaN
    """
    lat, lon = center

    # 円ポリゴン作成
    point = gpd.GeoSeries([Point((lon, lat))], crs=crs)
    circle = point.to_crs(point.estimate_utm_crs()).buffer(radius_meters).to_crs(crs)
    geom = [circle.iloc[0]]
    bbox = circle.total_bounds  # [lon_min, lat_min, lon_max, lat_max]

    # 該当する全サブメッシュを収集
    idx = _build_coverage_index(raster_dir)
    tiles = []

    for zip_path, lat_min, lat_max, lon_min, lon_max, dlat, dlon in idx:
        # bboxとzipの範囲が交差しない場合はスキップ
        if bbox[2] < lon_min or bbox[0] > lon_max:
            continue
        if bbox[3] < lat_min or bbox[1] > lat_max:
            continue

        # bbox にかかるサブメッシュ範囲を計算（0〜9 に clamp）
        si0 = max(0, int((bbox[1] - lat_min) / dlat))
        si1 = min(9, int((bbox[3] - lat_min) / dlat))
        sj0 = max(0, int((bbox[0] - lon_min) / dlon))
        sj1 = min(9, int((bbox[2] - lon_min) / dlon))

        for si in range(si0, si1 + 1):
            for sj in range(sj0, sj1 + 1):
                tiles.append((zip_path, si * 10 + sj))

    # マスク処理
    all_valid = []

    for zip_path, sidx in tiles:
        elev, tr = _read_submesh(zip_path, sidx)

        # メモリ上でGeoTIFFとして開き、円マスクを適用
        with rasterio.MemoryFile() as mem:
            with mem.open(
                driver="GTiff", height=elev.shape[0], width=elev.shape[1],
                count=1, dtype=elev.dtype, crs="EPSG:4326", transform=tr,
            ) as ds:
                ds.write(elev, 1)
                out_img, _ = mask(ds, geom, crop=True, all_touched=True, filled=False)

            data = out_img[0]

            # filled=False でマスク外は欠損になる→ 標高0mと区別できる
            if hasattr(data, 'mask') and data.mask.any():
                valid = data.data[(~data.mask) & (data.data != -9999)]
            else:
                valid = data[(data != -9999) & (data != 0)]

            if valid.size > 0:
                all_valid.append(valid)

    # 集計
    if not all_valid:
        return [float("nan"), float("nan"), float("nan"), float("nan")]

    combined = np.concatenate(all_valid)

    return [
        float(combined.mean()),
        float(combined.min()),
        float(combined.max()),
        float(combined.std()),
    ]

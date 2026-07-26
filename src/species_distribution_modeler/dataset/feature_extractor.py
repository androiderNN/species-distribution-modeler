import numpy as np
import numpy.typing as npt
import pandas as pd

from species_distribution_modeler.config.schemas import PathConfig
from species_distribution_modeler.dataset import gsi_dem, jaxa_lulc


def extract(
        center: tuple[float, float],
        path_cfg: PathConfig,
        radius_list: list[int],
    ) -> dict[str, float]:
    """
    採集地点の環境特徴量を抽出する
    地点のピクセル値と、指定した半径の円領域集計結果を結合して返す

    Args:
        center: 採集地点（緯度, 経度）
        path_cfg: データパス設定（PathConfig）
        radius_list: 集計に使う半径リスト（m）

    Returns:
        特徴量名→値のdict
    """
    features = dict()

    # 採集地点の指標抽出
    value = jaxa_lulc.sample_at_point(center, path_cfg.lulc_dir)
    features.update({"lulc_point": value})

    value = gsi_dem.sample_at_point(center, path_cfg.dem_dir)
    features.update({"dem_point": value})

    for radius in radius_list:
        # DSMの円領域集計
        names = [f"{n}_{radius}m" for n in gsi_dem.DEM_SUMMARY_NAMES]
        values = gsi_dem.summarize_in_circle(center, radius, path_cfg.dem_dir)
        features.update({n: f for n, f in zip(names, values)})

        # LULCの円領域集計
        names = [f"{n}_{radius}m" for n in jaxa_lulc.LULC_SUMMARY_NAMES]
        values = jaxa_lulc.summarize_in_circle(center, radius, path_cfg.lulc_dir)
        features.update({n: f for n, f in zip(names, values)})

    return features


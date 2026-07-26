from pathlib import Path
from pydantic import BaseModel


class PathConfig(BaseModel):
    """各種パス"""
    dsm_dir: Path
    lulc_dir: Path


class AppConfig(BaseModel):
    """すべての設定を保持するクラス"""
    paths : PathConfig


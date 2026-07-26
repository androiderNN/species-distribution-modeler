from pathlib import Path
import yaml

from .schemas import AppConfig, PathConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "configs"
_PATH_PREFIXES = ("data/", "output/")


def _resolve_paths(cfg: dict, root: Path) -> None:
    """
    設定内の相対パスを絶対パスに変換する（in-place）

    Args:
        cfg: 設定dict
        root: プロジェクトルートパス
    """
    for key, val in cfg.items():
        if isinstance(val, dict):
            _resolve_paths(val, root)
        elif isinstance(val, str) and val.startswith(_PATH_PREFIXES):
            cfg[key] = str(root / val)


def _load_yaml(config_dir: Path, filename: str) -> dict:
    """
    configs/配下のYAMLを読み込み、相対パスを解決して返す

    Args:
        config_dir: configの保存ディレクトリ
        filename: YAMLファイル名（config_dirからの相対パス）

    Returns:
        パス解決済みの dict
    """
    path = config_dir / filename

    with open(path) as f:
        cfg = yaml.safe_load(f)

    _resolve_paths(cfg, _PROJECT_ROOT)
    return cfg


def load_config(config_dir: Path = _CONFIG_DIR) -> AppConfig:
    """
    設定を読み込む

    Args:
        config_dir: config保存先ディレクトリ

    Returns:
        AppConfigインスタンス
    """
    cfg =  AppConfig(
        paths=PathConfig.model_validate(_load_yaml(config_dir, "path.yaml")),
    )

    return cfg


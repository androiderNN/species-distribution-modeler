"""Test that the package can be imported."""

from prediction_model_builder import __version__


def test_version() -> None:
    """Verify the version string is present."""
    assert __version__ == "0.1.0"

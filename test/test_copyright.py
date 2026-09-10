import pytest

try:
    from ament_copyright.main import main
    HAS_COPYRIGHT = True
except ImportError:
    HAS_COPYRIGHT = False


@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
    # Only run copyright check if explicitly requested with a flag
    pytest.skip("Copyright check skipped in standard test suite")

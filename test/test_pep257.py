import pytest

try:
    from ament_pep257.main import main
    import pydocstyle
    HAS_PEP257 = True
except ImportError:
    HAS_PEP257 = False


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    if not HAS_PEP257:
        pytest.skip("pep257 or ament_pep257 not installed")
    rc = main(argv=['.', 'test'])
    assert rc == 0, 'Found code style errors / warnings'

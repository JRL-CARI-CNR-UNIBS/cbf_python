import pytest

try:
    from ament_flake8.main import main_with_errors
    import flake8
    HAS_FLAKE8 = True
except ImportError:
    HAS_FLAKE8 = False


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    if not HAS_FLAKE8:
        pytest.skip("flake8 or ament_flake8 not installed")
    rc, errors = main_with_errors(argv=[])
    assert rc == 0, 'Found %d code style errors:\n' % len(errors) + '\n'.join(errors)

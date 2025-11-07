"""
Test basic project setup and imports.
"""

import pytest


def test_moonboard_core_import():
    """Test that moonboard_core can be imported."""
    try:
        import moonboard_core
        assert moonboard_core is not None
    except ImportError as e:
        pytest.fail(f"Failed to import moonboard_core: {e}")


def test_moonboard_core_modules():
    """Test that required moonboard_core modules are available."""
    from moonboard_core import grade_encoder
    from moonboard_core import position_parser
    from moonboard_core import grid_builder
    from moonboard_core import data_processor
    
    # Verify key functions exist
    assert hasattr(grade_encoder, 'encode_grade')
    assert hasattr(grade_encoder, 'decode_grade')
    assert hasattr(position_parser, 'parse_position')
    assert hasattr(grid_builder, 'create_grid_tensor')
    assert hasattr(data_processor, 'process_problem')
    assert hasattr(data_processor, 'load_dataset')


def test_src_package_import():
    """Test that the src package can be imported."""
    try:
        import src
        assert hasattr(src, '__version__')
        assert src.__version__ == "0.1.0"
    except ImportError as e:
        pytest.fail(f"Failed to import src package: {e}")


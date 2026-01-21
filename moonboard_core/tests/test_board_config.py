"""
Unit tests for board_config module
"""

import json
import tempfile
from pathlib import Path

import pytest

from moonboard_core.board_config import (
    AngleConfig,
    HoldSetup,
    BoardConfigRegistry,
    get_registry,
    reset_registry,
)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "holdSetups": [
            {
                "id": "masters-2017",
                "name": "MoonBoard Masters 2017",
                "angles": [
                    {
                        "angle": 40,
                        "dataFile": "data/problems.json",
                        "modelFile": "models/model.pth",
                        "isDefault": True
                    }
                ]
            }
        ]
    }


@pytest.fixture
def multi_setup_config_data():
    """Configuration with multiple setups and angles."""
    return {
        "holdSetups": [
            {
                "id": "masters-2017",
                "name": "MoonBoard Masters 2017",
                "angles": [
                    {
                        "angle": 25,
                        "dataFile": "data/2017-25/problems.json",
                        "modelFile": "models/2017-25.pth"
                    },
                    {
                        "angle": 40,
                        "dataFile": "data/2017-40/problems.json",
                        "modelFile": "models/2017-40.pth",
                        "isDefault": True
                    }
                ]
            },
            {
                "id": "masters-2019",
                "name": "MoonBoard Masters 2019",
                "angles": [
                    {
                        "angle": 40,
                        "dataFile": "data/2019-40/problems.json",
                        "modelFile": None
                    }
                ]
            }
        ]
    }


@pytest.fixture
def config_file(sample_config_data):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config_data, f)
        return Path(f.name)


@pytest.fixture
def multi_setup_config_file(multi_setup_config_data):
    """Create a temporary config file with multiple setups."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(multi_setup_config_data, f)
        return Path(f.name)


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset the global registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


class TestAngleConfig:
    """Test AngleConfig dataclass."""

    def test_create_angle_config(self):
        """Test creating an AngleConfig."""
        config = AngleConfig(
            angle=40,
            data_file="data/problems.json",
            model_file="models/model.pth",
            is_default=True
        )
        assert config.angle == 40
        assert config.data_file == "data/problems.json"
        assert config.model_file == "models/model.pth"
        assert config.is_default is True

    def test_angle_config_default_is_default(self):
        """Test that is_default defaults to False."""
        config = AngleConfig(
            angle=40,
            data_file="data/problems.json",
            model_file=None
        )
        assert config.is_default is False

    def test_angle_config_immutable(self):
        """Test that AngleConfig is immutable (frozen)."""
        config = AngleConfig(angle=40, data_file="test.json", model_file=None)
        with pytest.raises(AttributeError):
            config.angle = 25


class TestHoldSetup:
    """Test HoldSetup dataclass."""

    def test_create_hold_setup(self):
        """Test creating a HoldSetup."""
        angle = AngleConfig(angle=40, data_file="test.json", model_file=None)
        setup = HoldSetup(
            id="test-setup",
            name="Test Setup",
            angles=(angle,)
        )
        assert setup.id == "test-setup"
        assert setup.name == "Test Setup"
        assert len(setup.angles) == 1
        assert setup.angles[0].angle == 40

    def test_hold_setup_immutable(self):
        """Test that HoldSetup is immutable (frozen)."""
        angle = AngleConfig(angle=40, data_file="test.json", model_file=None)
        setup = HoldSetup(id="test", name="Test", angles=(angle,))
        with pytest.raises(AttributeError):
            setup.id = "new-id"


class TestBoardConfigRegistry:
    """Test BoardConfigRegistry class."""

    def test_load_config_from_file(self, config_file):
        """Test loading configuration from a file."""
        registry = BoardConfigRegistry(config_file)
        setups = registry.get_hold_setups()
        assert len(setups) == 1
        assert setups[0].id == "masters-2017"

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            BoardConfigRegistry("/nonexistent/path/config.json")

    def test_missing_hold_setups_key(self):
        """Test that ValueError is raised when holdSetups key is missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"other": "data"}, f)
            config_path = Path(f.name)

        with pytest.raises(ValueError, match="must contain 'holdSetups'"):
            BoardConfigRegistry(config_path)

    def test_missing_setup_id(self):
        """Test that ValueError is raised when setup has no id."""
        data = {
            "holdSetups": [
                {"name": "Test", "angles": [{"angle": 40, "dataFile": "test.json", "isDefault": True}]}
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            config_path = Path(f.name)

        with pytest.raises(ValueError, match="must have an 'id' field"):
            BoardConfigRegistry(config_path)

    def test_missing_angles(self):
        """Test that ValueError is raised when setup has no angles."""
        data = {
            "holdSetups": [
                {"id": "test", "name": "Test", "angles": []}
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            config_path = Path(f.name)

        with pytest.raises(ValueError, match="must have at least one angle"):
            BoardConfigRegistry(config_path)

    def test_no_default_configuration(self):
        """Test that ValueError is raised when no default is set."""
        data = {
            "holdSetups": [
                {"id": "test", "name": "Test", "angles": [{"angle": 40, "dataFile": "test.json"}]}
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            config_path = Path(f.name)

        with pytest.raises(ValueError, match="No default configuration found"):
            BoardConfigRegistry(config_path)

    def test_multiple_defaults(self):
        """Test that ValueError is raised when multiple defaults are set."""
        data = {
            "holdSetups": [
                {
                    "id": "test1",
                    "name": "Test 1",
                    "angles": [{"angle": 40, "dataFile": "test.json", "isDefault": True}]
                },
                {
                    "id": "test2",
                    "name": "Test 2",
                    "angles": [{"angle": 40, "dataFile": "test.json", "isDefault": True}]
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            config_path = Path(f.name)

        with pytest.raises(ValueError, match="Multiple default configurations"):
            BoardConfigRegistry(config_path)


class TestGetHoldSetups:
    """Test get_hold_setups method."""

    def test_returns_all_setups(self, multi_setup_config_file):
        """Test that all setups are returned."""
        registry = BoardConfigRegistry(multi_setup_config_file)
        setups = registry.get_hold_setups()
        assert len(setups) == 2

        ids = {s.id for s in setups}
        assert ids == {"masters-2017", "masters-2019"}

    def test_returns_list(self, config_file):
        """Test that a list is returned."""
        registry = BoardConfigRegistry(config_file)
        setups = registry.get_hold_setups()
        assert isinstance(setups, list)


class TestGetHoldSetup:
    """Test get_hold_setup method."""

    def test_get_existing_setup(self, config_file):
        """Test getting an existing setup by ID."""
        registry = BoardConfigRegistry(config_file)
        setup = registry.get_hold_setup("masters-2017")
        assert setup.id == "masters-2017"
        assert setup.name == "MoonBoard Masters 2017"

    def test_get_nonexistent_setup(self, config_file):
        """Test that KeyError is raised for nonexistent setup."""
        registry = BoardConfigRegistry(config_file)
        with pytest.raises(KeyError, match="not found"):
            registry.get_hold_setup("nonexistent")


class TestGetAngleConfig:
    """Test get_angle_config method."""

    def test_get_existing_angle(self, multi_setup_config_file):
        """Test getting an existing angle configuration."""
        registry = BoardConfigRegistry(multi_setup_config_file)
        angle_config = registry.get_angle_config("masters-2017", 40)
        assert angle_config.angle == 40
        assert angle_config.data_file == "data/2017-40/problems.json"
        assert angle_config.is_default is True

    def test_get_nonexistent_angle(self, config_file):
        """Test that ValueError is raised for nonexistent angle."""
        registry = BoardConfigRegistry(config_file)
        with pytest.raises(ValueError, match="Angle 25 not found"):
            registry.get_angle_config("masters-2017", 25)

    def test_get_angle_nonexistent_setup(self, config_file):
        """Test that KeyError is raised for nonexistent setup."""
        registry = BoardConfigRegistry(config_file)
        with pytest.raises(KeyError, match="not found"):
            registry.get_angle_config("nonexistent", 40)

    def test_null_model_file(self, multi_setup_config_file):
        """Test that null model file is handled correctly."""
        registry = BoardConfigRegistry(multi_setup_config_file)
        angle_config = registry.get_angle_config("masters-2019", 40)
        assert angle_config.model_file is None


class TestGetDefault:
    """Test get_default method."""

    def test_get_default(self, config_file):
        """Test getting the default configuration."""
        registry = BoardConfigRegistry(config_file)
        setup, angle_config = registry.get_default()
        assert setup.id == "masters-2017"
        assert angle_config.angle == 40
        assert angle_config.is_default is True

    def test_get_default_multi_setup(self, multi_setup_config_file):
        """Test getting default from multiple setups."""
        registry = BoardConfigRegistry(multi_setup_config_file)
        setup, angle_config = registry.get_default()
        assert setup.id == "masters-2017"
        assert angle_config.angle == 40


class TestGlobalRegistry:
    """Test global registry functions."""

    def test_get_registry_creates_instance(self, config_file):
        """Test that get_registry creates a singleton instance."""
        registry = get_registry(config_file)
        assert registry is not None
        assert isinstance(registry, BoardConfigRegistry)

    def test_get_registry_returns_same_instance(self, config_file):
        """Test that get_registry returns the same instance."""
        registry1 = get_registry(config_file)
        registry2 = get_registry()
        assert registry1 is registry2

    def test_reset_registry(self, config_file):
        """Test that reset_registry clears the instance."""
        registry1 = get_registry(config_file)
        reset_registry()
        registry2 = get_registry(config_file)
        assert registry1 is not registry2


class TestProductionConfig:
    """Test loading the actual production configuration."""

    def test_load_production_config(self):
        """Test loading the production board_setups.json file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "board_setups.json"
        if not config_path.exists():
            pytest.skip("Production config file not found")

        registry = BoardConfigRegistry(config_path)
        setups = registry.get_hold_setups()
        assert len(setups) >= 1

        # Verify default exists
        default_setup, default_angle = registry.get_default()
        assert default_setup is not None
        assert default_angle is not None
        assert default_angle.is_default is True

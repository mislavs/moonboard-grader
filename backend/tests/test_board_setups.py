"""
Tests for board setups API endpoint.
"""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import create_application
from app.core.config import settings


@pytest.fixture
def sample_board_config():
    """Sample board configuration for testing."""
    return {
        "holdSetups": [
            {
                "id": "masters-2017",
                "name": "MoonBoard Masters 2017",
                "angles": [
                    {
                        "angle": 40,
                        "dataFile": "data/problems.json",
                        "modelFile": "models/model_for_inference.pth",
                        "isDefault": True
                    }
                ]
            }
        ]
    }


@pytest.fixture
def multi_setup_board_config():
    """Board configuration with multiple setups for testing."""
    return {
        "holdSetups": [
            {
                "id": "masters-2017",
                "name": "MoonBoard Masters 2017",
                "angles": [
                    {
                        "angle": 25,
                        "dataFile": "data/2017-25/problems.json",
                        "modelFile": None
                    },
                    {
                        "angle": 40,
                        "dataFile": "data/problems.json",
                        "modelFile": "models/model_for_inference.pth",
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
def temp_board_config(tmp_path: Path, sample_board_config):
    """Create a temporary board config file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "board_setups.json"
    config_file.write_text(json.dumps(sample_board_config, indent=2))
    return config_file


@pytest.fixture
def temp_multi_setup_config(tmp_path: Path, multi_setup_board_config):
    """Create a temporary board config file with multiple setups."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "board_setups.json"
    config_file.write_text(json.dumps(multi_setup_board_config, indent=2))
    return config_file


@pytest.fixture
def client_with_board_config(temp_board_config, monkeypatch):
    """Test client with temporary board config."""
    monkeypatch.setattr(settings, 'board_config_path', temp_board_config)
    app = create_application()
    return TestClient(app)


@pytest.fixture
def client_with_multi_setup(temp_multi_setup_config, monkeypatch):
    """Test client with multiple board setups."""
    monkeypatch.setattr(settings, 'board_config_path', temp_multi_setup_config)
    app = create_application()
    return TestClient(app)


class TestBoardSetupsEndpoint:
    """Test GET /board-setups endpoint."""

    def test_get_board_setups_returns_200(self, client_with_board_config):
        """Test that endpoint returns 200 OK."""
        response = client_with_board_config.get("/board-setups")
        assert response.status_code == 200

    def test_get_board_setups_returns_correct_structure(self, client_with_board_config):
        """Test that response has correct structure."""
        response = client_with_board_config.get("/board-setups")
        data = response.json()

        assert "holdSetups" in data
        assert isinstance(data["holdSetups"], list)
        assert len(data["holdSetups"]) == 1

    def test_get_board_setups_returns_setup_details(self, client_with_board_config):
        """Test that setup details are correct."""
        response = client_with_board_config.get("/board-setups")
        data = response.json()

        setup = data["holdSetups"][0]
        assert setup["id"] == "masters-2017"
        assert setup["name"] == "MoonBoard Masters 2017"
        assert "angles" in setup
        assert len(setup["angles"]) == 1

    def test_get_board_setups_returns_angle_details(self, client_with_board_config):
        """Test that angle details are correct."""
        response = client_with_board_config.get("/board-setups")
        data = response.json()

        angle = data["holdSetups"][0]["angles"][0]
        assert angle["angle"] == 40
        assert angle["isDefault"] is True
        assert "hasModel" in angle

    def test_get_board_setups_multiple_setups(self, client_with_multi_setup):
        """Test response with multiple setups."""
        response = client_with_multi_setup.get("/board-setups")
        data = response.json()

        assert len(data["holdSetups"]) == 2
        ids = {s["id"] for s in data["holdSetups"]}
        assert ids == {"masters-2017", "masters-2019"}

    def test_get_board_setups_multiple_angles(self, client_with_multi_setup):
        """Test response with multiple angles per setup."""
        response = client_with_multi_setup.get("/board-setups")
        data = response.json()

        # Find masters-2017 setup
        setup_2017 = next(s for s in data["holdSetups"] if s["id"] == "masters-2017")
        assert len(setup_2017["angles"]) == 2

        angles = {a["angle"] for a in setup_2017["angles"]}
        assert angles == {25, 40}

    def test_get_board_setups_only_one_default(self, client_with_multi_setup):
        """Test that only one angle is marked as default."""
        response = client_with_multi_setup.get("/board-setups")
        data = response.json()

        defaults = []
        for setup in data["holdSetups"]:
            for angle in setup["angles"]:
                if angle["isDefault"]:
                    defaults.append((setup["id"], angle["angle"]))

        assert len(defaults) == 1
        assert defaults[0] == ("masters-2017", 40)


class TestBoardSetupsWithMissingConfig:
    """Test behavior when config file is missing."""

    def test_missing_config_returns_empty_list(self, tmp_path, monkeypatch):
        """Test that missing config returns empty list."""
        nonexistent_path = tmp_path / "nonexistent" / "board_setups.json"
        monkeypatch.setattr(settings, 'board_config_path', nonexistent_path)

        app = create_application()
        client = TestClient(app)

        response = client.get("/board-setups")
        assert response.status_code == 200

        data = response.json()
        assert data["holdSetups"] == []


class TestProductionConfig:
    """Test with actual production configuration."""

    def test_load_production_config(self, monkeypatch):
        """Test loading the actual production board_setups.json."""
        config_path = Path(__file__).parent.parent.parent / "config" / "board_setups.json"
        if not config_path.exists():
            pytest.skip("Production config file not found")

        # Set the config path to the actual file
        monkeypatch.setattr(settings, 'board_config_path', config_path)
        app = create_application()
        client = TestClient(app)

        response = client.get("/board-setups")
        assert response.status_code == 200

        data = response.json()
        assert len(data["holdSetups"]) >= 1

        # Verify at least one default exists
        has_default = False
        for setup in data["holdSetups"]:
            for angle in setup["angles"]:
                if angle["isDefault"]:
                    has_default = True
                    break
            if has_default:
                break

        assert has_default, "No default configuration found"


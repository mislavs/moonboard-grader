"""
Board Configuration Module

This module provides configuration management for different Moonboard setups
and wall angles. It loads configuration from a central JSON file and provides
access to hold setups and their associated angles, data files, and models.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AngleConfig:
    """Configuration for a specific wall angle within a hold setup."""
    angle: int
    data_file: str
    model_file: Optional[str]
    is_default: bool = False


@dataclass(frozen=True)
class HoldSetup:
    """Configuration for a hold setup (e.g., MoonBoard Masters 2017)."""
    id: str
    name: str
    angles: tuple[AngleConfig, ...]


class BoardConfigRegistry:
    """
    Registry for board configurations loaded from a JSON file.

    The configuration file should have the following structure:
    {
        "holdSetups": [
            {
                "id": "masters-2017",
                "name": "MoonBoard Masters 2017",
                "angles": [
                    {
                        "angle": 40,
                        "dataFile": "data/problems.json",
                        "modelFile": "models/model_for_inference.pth",
                        "isDefault": true
                    }
                ]
            }
        ]
    }
    """

    def __init__(self, config_path: Path | str):
        """
        Initialize the registry by loading configuration from a JSON file.

        Args:
            config_path: Path to the board_setups.json configuration file

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        self._config_path = Path(config_path)
        self._hold_setups: dict[str, HoldSetup] = {}
        self._default_setup_id: Optional[str] = None
        self._default_angle: Optional[int] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load and parse the configuration file."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self._config_path}")

        with open(self._config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'holdSetups' not in data:
            raise ValueError("Configuration must contain 'holdSetups' array")

        for setup_data in data['holdSetups']:
            self._parse_hold_setup(setup_data)

        if self._default_setup_id is None:
            raise ValueError("No default configuration found. One angle must have 'isDefault': true")

    def _parse_hold_setup(self, setup_data: dict) -> None:
        """Parse a single hold setup from the configuration data."""
        setup_id = setup_data.get('id')
        if not setup_id:
            raise ValueError("Hold setup must have an 'id' field")

        name = setup_data.get('name', setup_id)
        angles_data = setup_data.get('angles', [])

        if not angles_data:
            raise ValueError(f"Hold setup '{setup_id}' must have at least one angle configuration")

        angles = []
        for angle_data in angles_data:
            angle_config = AngleConfig(
                angle=angle_data['angle'],
                data_file=angle_data['dataFile'],
                model_file=angle_data.get('modelFile'),
                is_default=angle_data.get('isDefault', False)
            )
            angles.append(angle_config)

            if angle_config.is_default:
                if self._default_setup_id is not None:
                    raise ValueError(
                        f"Multiple default configurations found. "
                        f"Only one angle should have 'isDefault': true"
                    )
                self._default_setup_id = setup_id
                self._default_angle = angle_config.angle

        self._hold_setups[setup_id] = HoldSetup(
            id=setup_id,
            name=name,
            angles=tuple(angles)
        )

    def get_hold_setups(self) -> list[HoldSetup]:
        """
        Get all available hold setups.

        Returns:
            List of all HoldSetup configurations
        """
        return list(self._hold_setups.values())

    def get_hold_setup(self, setup_id: str) -> HoldSetup:
        """
        Get a specific hold setup by ID.

        Args:
            setup_id: The ID of the hold setup

        Returns:
            The HoldSetup configuration

        Raises:
            KeyError: If the setup ID doesn't exist
        """
        if setup_id not in self._hold_setups:
            raise KeyError(f"Hold setup '{setup_id}' not found")
        return self._hold_setups[setup_id]

    def get_angle_config(self, setup_id: str, angle: int) -> AngleConfig:
        """
        Get the configuration for a specific setup and angle combination.

        Args:
            setup_id: The ID of the hold setup
            angle: The wall angle in degrees

        Returns:
            The AngleConfig for the specified setup and angle

        Raises:
            KeyError: If the setup ID doesn't exist
            ValueError: If the angle doesn't exist for the setup
        """
        setup = self.get_hold_setup(setup_id)
        for angle_config in setup.angles:
            if angle_config.angle == angle:
                return angle_config
        raise ValueError(f"Angle {angle} not found for setup '{setup_id}'")

    def get_default(self) -> tuple[HoldSetup, AngleConfig]:
        """
        Get the default hold setup and angle configuration.

        Returns:
            Tuple of (HoldSetup, AngleConfig) for the default configuration
        """
        setup = self._hold_setups[self._default_setup_id]
        angle_config = self.get_angle_config(self._default_setup_id, self._default_angle)
        return setup, angle_config


# Default configuration path relative to project root
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "board_setups.json"

# Global registry instance (lazy-loaded)
_registry: Optional[BoardConfigRegistry] = None


def get_registry(config_path: Path | str | None = None) -> BoardConfigRegistry:
    """
    Get the global board configuration registry.

    Args:
        config_path: Optional path to configuration file. If not provided,
                    uses the default path. Only used on first call.

    Returns:
        The global BoardConfigRegistry instance
    """
    global _registry
    if _registry is None:
        path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH
        _registry = BoardConfigRegistry(path)
    return _registry


def reset_registry() -> None:
    """Reset the global registry. Useful for testing."""
    global _registry
    _registry = None

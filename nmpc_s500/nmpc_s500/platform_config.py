"""Platform configuration loader for NMPC S500 package.

Defines PlatformConfig dataclass and loads preset configurations from YAML.
"""

from dataclasses import dataclass
import yaml
import os


@dataclass
class PlatformConfig:
    """Platform-specific parameters for NMPC solver and controller."""

    name: str
    mass_kg: float
    max_thrust_n: float
    Ixx: float
    Iyy: float
    Izz: float
    max_body_rate: float
    hover_thrust_normalised: float

    def __post_init__(self):
        """Validate platform config on instantiation."""
        if self.mass_kg <= 0:
            raise ValueError(f"mass_kg must be > 0, got {self.mass_kg}")
        if self.max_thrust_n <= 0:
            raise ValueError(f"max_thrust_n must be > 0, got {self.max_thrust_n}")
        if not (0 < self.hover_thrust_normalised <= 1.0):
            raise ValueError(f"hover_thrust_normalised must be in (0, 1], got {self.hover_thrust_normalised}")
        if self.max_body_rate <= 0:
            raise ValueError(f"max_body_rate must be > 0, got {self.max_body_rate}")


def load_platform_config(platform_name: str) -> PlatformConfig:
    """Load platform configuration from YAML.

    Args:
        platform_name: Name of the platform preset (e.g. 'sim_iris', 'real_s500')

    Returns:
        PlatformConfig dataclass instance

    Raises:
        FileNotFoundError: If platforms.yaml not found
        KeyError: If platform_name not in YAML
        ValueError: If configuration validation fails
    """
    # Find platforms.yaml relative to this module
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    platforms_file = os.path.join(config_dir, 'config', 'platforms.yaml')

    if not os.path.isfile(platforms_file):
        raise FileNotFoundError(f"platforms.yaml not found at {platforms_file}")

    with open(platforms_file, 'r') as f:
        platforms = yaml.safe_load(f)

    if platform_name not in platforms:
        available = ', '.join(platforms.keys())
        raise KeyError(f"Platform '{platform_name}' not found. Available: {available}")

    cfg_dict = platforms[platform_name]
    cfg_dict['name'] = platform_name

    return PlatformConfig(**cfg_dict)

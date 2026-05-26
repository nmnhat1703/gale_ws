"""Platform-aware solver setup and code generation.

This module wraps the NMPC solver generation with platform-specific configuration.
"""

from .platform_config import PlatformConfig
from .generate_nmpc import QuadrotorEulerErrMPC
from .acados_model import QuadrotorEulerModel


def create_solver(
    platform_config: PlatformConfig,
    horizon: float = 2.0,
    num_steps: int = 50,
    generate_c_code: bool = False,
) -> QuadrotorEulerErrMPC:
    """Create an NMPC solver for the given platform configuration.

    Args:
        platform_config: PlatformConfig instance with mass, thrust bounds, etc.
        horizon: MPC horizon in seconds
        num_steps: Number of horizon discretisation steps
        generate_c_code: If True, regenerate C code; else load cached

    Returns:
        QuadrotorEulerErrMPC solver instance
    """
    model = QuadrotorEulerModel(
        mass=platform_config.mass_kg,
        Ixx=platform_config.Ixx,
        Iyy=platform_config.Iyy,
        Izz=platform_config.Izz,
    )

    solver = QuadrotorEulerErrMPC(
        generate_c_code=generate_c_code,
        quadrotor=model,
        horizon=horizon,
        num_steps=num_steps,
        max_thrust_n=platform_config.max_thrust_n,
        max_body_rate=platform_config.max_body_rate,
    )

    return solver

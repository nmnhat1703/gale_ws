#!/usr/bin/env python3
"""Standalone script to generate the acados NMPC solver C code for a given platform.

Usage:
  python3 generate_solver.py --platform sim_iris
  python3 generate_solver.py --platform real_s500

The generated C code is written to:
  nmpc_s500/acados_generated_files/holybro_euler_err_mpc_c_generated_code/
"""

import argparse
import sys
import os

# Make sure we can import nmpc_s500 modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmpc_s500.platform_config import load_platform_config
from nmpc_s500.solver_setup import create_solver


def main():
    parser = argparse.ArgumentParser(
        description="Generate acados NMPC solver C code for a platform"
    )
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        choices=["sim_iris", "real_s500"],
        help="Platform configuration (required)",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=2.0,
        help="MPC horizon in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of horizon steps (default: 50)",
    )

    args = parser.parse_args()

    print(f"\n[generate_solver] Loading platform config: {args.platform}")
    platform_cfg = load_platform_config(args.platform)

    print(f"[generate_solver] Platform: {platform_cfg.name}")
    print(f"  mass: {platform_cfg.mass_kg} kg")
    print(f"  max_thrust: {platform_cfg.max_thrust_n} N")
    print(f"  horizon: {args.horizon} s, num_steps: {args.num_steps}")

    print(f"\n[generate_solver] Generating solver...")
    solver = create_solver(
        platform_config=platform_cfg,
        horizon=args.horizon,
        num_steps=args.num_steps,
        generate_c_code=True,  # Force generation
    )

    print(f"[generate_solver] Done!")
    print(f"[generate_solver] Solver C code at: {solver.code_export_directory}")


if __name__ == "__main__":
    main()

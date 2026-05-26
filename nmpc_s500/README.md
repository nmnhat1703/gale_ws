# nmpc_s500 — ROS 1 NMPC Controller for S500 Quadrotor

**Status**: Step 2 — Package skeleton created. Steps 3–9 in progress.

A ROS 1 Noetic NMPC (Nonlinear Model Predictive Control) package for autonomous flight of an S500 quadrotor with LiDAR-inertial SLAM, using MAVROS to interface with PX4 v1.16.0.

## Overview

This package adapts the upstream [evannsmc/nmpc_acados_px4](https://github.com/evannsmc/nmpc_acados_px4) from ROS 2 to ROS 1, replacing the ROS 2-specific PX4 bridge with MAVROS, and targeting the S500 + Hesai LiDAR platform with ENU-based state estimation.

### Key Features

- **NMPC Solver**: Acados-based finite-horizon OCP with wrapped yaw error
- **State Vector**: 9D `[x, y, z, vx, vy, vz, roll, pitch, yaw]` (ENU frame)
- **Control Input**: 4D `[thrust_N, p, q, r]` (thrust in Newtons, body rates in rad/s)
- **Platform Abstraction**: Easy switching between sim_iris (Gazebo) and real_s500
- **ROS 1 / MAVROS**: Subscribers to `/mavros/local_position/` for state, publisher to `/mavros/setpoint_raw/attitude` for control

## Package Structure

```
nmpc_s500/
├── nmpc_s500/                          # Python package
│   ├── __init__.py
│   ├── platform_config.py              # Load platform configs from YAML
│   ├── frames.py                       # Quaternion → ZYX Euler conversion
│   ├── acados_model.py                 # Quadrotor dynamics (upstream)
│   ├── generate_nmpc.py                # NMPC OCP formulation (patched)
│   ├── yaw_utils.py                    # Yaw wrapping utilities (upstream)
│   ├── solver_setup.py                 # Solver factory
│   └── nmpc_node.py                    # ROS 1 node (Step 6)
├── scripts/
│   └── generate_solver.py              # CLI to generate solver for a platform
├── launch/
│   ├── sitl_iris.launch                # SITL simulation launch
│   └── real_s500.launch                # Real hardware launch
├── config/
│   └── platforms.yaml                  # Platform presets (sim_iris, real_s500)
├── tests/
│   ├── __init__.py
│   └── test_frames.py                  # Frame conversion tests (Step 5)
├── CMakeLists.txt
├── package.xml
├── setup.py
└── README.md
```

## Installation & Build

### Prerequisites

- ROS 1 Noetic on Ubuntu 20.04
- MAVROS (installed)
- acados (installed and `ACADOS_SOURCE_DIR` environment variable set)
- PX4-Autopilot v1.16.0 (for SITL simulation)

### Build

```bash
cd ~/catkin_ws
catkin build nmpc_s500
source devel/setup.bash
```

## Known Assumptions (Placeholders)

These assumptions are documented in the code and will be refined during validation flights:

### sim_iris (Gazebo Classic iris simulation)
- **Maximum thrust**: 22.8 N (unverified estimate: 4 motors × ~5.7 N each)
  - Status: Stock Gazebo iris model; thrust characteristics not validated against actual iris hardware
  - Symptom if wrong: If the drone won't climb in SITL or thrust saturates near 1.0, revisit this value
  - Impact: Hard constraint in the OCP; incorrect value can mask tuning issues

### real_s500 (Custom S500 hardware)
- **Mass**: 2.7 kg (measured: LiDAR + battery + Microstrain + airframe + props)
  - Status: Verified by scale
  - Impact: Affects hover thrust calculation and OCP scaling

- **Inertia values**: Ixx=0.029, Iyy=0.029, Izz=0.055 kg·m² (CAD estimates for S500-class quad)
  - Status: Placeholder; to be refined via system ID or auto-tune
  - Impact: Currently NOT used in the OCP dynamics; kept for future rate damping / advanced control

- **Maximum thrust**: 135.0 N (derived from measured HTE = 0.196 at flight 29)
  - Calculation: mass × g / HTE = 2.7 × 9.81 / 0.196 ≈ 135.0 N
  - Avenger 3120-700KV motors with current battery/props
  - Status: Based on empirical HTE; valid only if battery cell count and props remain unchanged
  - Caveat: Re-validate HTE on non-saturated range before trusting for aggressive trajectories. Set MPC_THR_HOVER=0.4, HTE_THR_RANGE=0.4 before validation flight.
  - Impact: Hard constraint in the OCP; if battery changed (e.g. 6S→8S), must be recomputed

- **Hover thrust normalised**: 0.196 (measured from flight 29 HTE convergence)
  - Status: Verified empirically
  - Impact: Used at runtime to scale setpoint thrust [0,1] ↔ [0, max_thrust_n] N

## Frame Convention

**ENU (East-North-Up), z-up world frame**:
- Position: `[x, y, z]` where z points upward
- Velocity: `[vx, vy, vz]` in world frame
- Gravity: `[0, 0, 9.8]` (positive upward)
- Euler angles: `[roll, pitch, yaw]` in ZYX intrinsic order

No ENU↔NED conversions; the acados model uses ENU natively, matching MAVROS.

## Usage

### SITL Simulation

```bash
# Terminal 1: PX4 SITL
cd ~/PX4-Autopilot
make px4_sitl gazebo-classic_iris

# Terminal 2: ROS + MAVROS + NMPC
roslaunch nmpc_s500 sitl_iris.launch
```

Expected behavior:
- Node initializes, generates/loads solver
- Streams attitude setpoints at 50 Hz
- Drone takes off to 1.5 m and holds hover

### Real Hardware

Assumes MAVROS is already up and running (e.g. via a separate launch file).

```bash
roslaunch nmpc_s500 real_s500.launch
```

**Safety protocol** (manual RC control required):
1. Arm via RC in STAB mode
2. Drone takes off to 1 m (controlled by PX4 auto-takeoff)
3. Flip RC switch to OFFBOARD (position 6) to engage NMPC
4. RC kill switch (position 5) for emergency exit

## Deviations from Upstream

| Aspect | Upstream | This Package | Reason |
|--------|----------|--------------|--------|
| ROS Version | ROS 2 Humble | ROS 1 Noetic | Target platform |
| PX4 Bridge | `px4_msgs` + uXRCE-DDS | MAVROS | ROS 1 integration |
| State Topics | `/fmu/out/vehicle_odometry` | `/mavros/local_position/pose` + `velocity_local` | MAVROS standard |
| Control Topic | `/fmu/in/vehicle_rates_setpoint` | `/mavros/setpoint_raw/attitude` | MAVROS standard |
| Trajectory Generator | `quad_trajectories` (dependency) | None yet | Defer to Phase 3; only hover implemented |
| Platform Config | Runtime via CLI args | YAML file + Python module | Simplify reuse across scripts |
| Feedforward Control | JAX-based DF (upstream) | Not ported | Defer to Phase 3; test with feedback-only first |

## Build Plan Completion

- [x] Step 1 — Read upstream repo
- [x] Step 2 — Create ROS 1 package skeleton
- [ ] Step 3 — Copy OCP files and patch
- [ ] Step 4 — Write platform config and solver setup
- [ ] Step 5 — Write frames.py and tests
- [ ] Step 6 — Write the ROS 1 node
- [ ] Step 7 — Write launch files
- [ ] Step 8 — End-to-end SITL run
- [ ] Step 9 — Write README (this file)

## License

MIT (inherited from upstream evannsmc/nmpc_acados_px4)

## References

- **Upstream**: https://github.com/evannsmc/nmpc_acados_px4
- **Acados**: https://github.com/acados/acados
- **MAVROS**: http://wiki.ros.org/mavros
- **PX4**: https://px4.io/

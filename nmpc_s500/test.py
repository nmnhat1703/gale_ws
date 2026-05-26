import sys
sys.path.insert(0, 'nmpc_s500')
from platform_config import load_platform_config
from solver_setup import create_solver
import numpy as np

cfg = load_platform_config('sim_iris')
print(f'Platform mass: {cfg.mass_kg} kg')
print(f'Expected hover thrust: {cfg.mass_kg * 9.81:.4f} N')

solver = create_solver(platform_config=cfg, horizon=2.0, num_steps=50, generate_c_code=False)

# Inspect the solver's stored hover_ctrl
print(f'solver.hover_ctrl: {solver.hover_ctrl}')
print(f'solver.quad.m: {solver.quad.m}')
print(f'solver.quad.g: {solver.quad.g}')

# Simulate first RUN solve: drone on ground, target 1.5 m up
x_now = np.array([0.0, 0.0, 0.0,    # position (on ground)
                  0.0, 0.0, 0.0,    # velocity
                  0.0, 0.0, 0.0])   # euler
x_ref = np.tile(np.array([0.0, 0.0, 1.5,    # position target
                          0.0, 0.0, 0.0,    # velocity
                          0.0, 0.0, 0.0]),  # euler
                (50, 1))

last_u = np.array([cfg.mass_kg * 9.81, 0.0, 0.0, 0.0])
u, x_pred, status = solver.solve_mpc_control(
    x_now, x_ref, last_u, nx=9, nu=4, verbose=True
)
print(f'Solver status: {status}')
print(f'First control u[0]: {u[0, :]}')
print(f'Thrust commanded: {u[0, 0]:.4f} N')
print(f'Expected (hover): {cfg.mass_kg * 9.81:.4f} N')
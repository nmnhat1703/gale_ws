"""
Quadrotor NMPC with Error-Based Cost using Euler Angles
========================================================

This MPC uses error-based cost formulation where:
- Stage cost output: [p_err, v_err, euler_err, u_err] = 13D
- Terminal cost output: [p_err, v_err, euler_err] = 9D
- References are passed as stage-wise parameters
- yref is ZERO because cost expression outputs errors

Key improvement: Wrapped yaw error using atan2(sin, cos) for proper angle wrapping.

PATCHES vs upstream evannsmc/nmpc_acados_px4:
  1. max_thrust_n: passed as __init__ parameter (was hardcoded 27.0)
  2. max_body_rate: passed as __init__ parameter (was hardcoded 0.8)
  3. QuadrotorEulerModel now accepts Ixx, Iyy, Izz parameters (reserved for future use)
"""

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from .acados_model import QuadrotorEulerModel
import importlib, sys, os
from scipy.linalg import block_diag
import time


class QuadrotorEulerErrMPC:
    def __init__(
        self,
        generate_c_code: bool,
        quadrotor: QuadrotorEulerModel,
        horizon: float,
        num_steps: int,
        max_thrust_n: float = 27.0,
        max_body_rate: float = 0.8,
    ):
        """Initialize NMPC solver with platform-specific constraints.

        Args:
            generate_c_code: If True, regenerate C code; else load cached solver
            quadrotor: QuadrotorEulerModel instance (holds mass, inertia)
            horizon: MPC horizon in seconds
            num_steps: Number of horizon steps
            max_thrust_n: Maximum total thrust in Newtons (platform-specific)
            max_body_rate: Maximum body angular rate in rad/s
        """
        self.model = AcadosModel()
        self.quad = quadrotor
        self.model_name = 'holybro_euler_err'
        self.horizon = horizon
        self.num_steps = num_steps
        self.max_thrust_n = max_thrust_n
        self.max_body_rate = max_body_rate

        self.hover_ctrl = np.array([self.quad.m * self.quad.g, 0., 0., 0.])

        self.ocp_solver = None
        self.generate_c_code = generate_c_code

        # === Put all generated artifacts under <this package>/acados_generated_files ===
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        parts = pkg_dir.split(os.sep)
        parts = ["src" if part in ("build", "install") else part for part in parts]
        pkg_dir = os.sep.join(parts)
        gen_root = os.path.join(pkg_dir, "acados_generated_files")
        os.makedirs(gen_root, exist_ok=True)

        # C code export dir and JSON path
        self.code_export_directory = os.path.join(gen_root, f"{self.model_name}_mpc_c_generated_code")
        self.json_path = os.path.join(gen_root, f"{self.model_name}_mpc_acados_ocp.json")

        if self.generate_c_code:
            print("\n[acados] Generating/compiling fresh MPC (euler_err)...\n")
            self.generate_mpc()
            print("[acados] Done.")
        else:
            try:
                print("\n[acados] Trying to load compiled MPC (euler_err); will generate if not found...\n")
                sys.path.append(self.code_export_directory)
                acados_ocp_solver_pyx = importlib.import_module('acados_ocp_solver_pyx')
                self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(
                    self.model_name, 'SQP', self.num_steps
                )
            except ImportError:
                print("[acados] Compiled MPC not found. Generating now...")
                self.generate_mpc()
                print("[acados] Done! Control stack should begin in two seconds...")
                time.sleep(2)

    def generate_mpc(self):
        """Generate the MPC with error-based NONLINEAR_LS cost."""
        # Get dynamics and cost expressions from model
        f_expl, f_impl, x, xdot, u, full_ref, cost_y_expr, cost_y_expr_e = self.quad.dynamics()

        model = self.model
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = full_ref  # Stage-wise reference values as a param named 'p'
        model.name = self.model_name

        # Define Acados OCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.code_export_directory = self.code_export_directory

        # Dimensions
        nx = model.x.size()[0]  # 9 (position, velocity, euler)
        nu = model.u.size()[0]  # 4 (thrust, p, q, r)
        np_param = model.p.size()[0]  # 13 (p_ref, v_ref, euler_ref, u_ref)
        ny = cost_y_expr.size()[0]    # 13 (p_err, v_err, euler_err, u_err)
        ny_e = cost_y_expr_e.size()[0]  # 9 (p_err, v_err, euler_err)

        # Temporal parameters
        Tf = self.horizon
        N = self.num_steps
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = Tf
        ocp.solver_options.qp_solver_cond_N = N

        # --- Cost function (NONLINEAR_LS with error-based expressions) ---
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # Axis-specific weights let us damp XY hover oscillation without
        # disturbing the vertical hover calibration.
        pos_xy_cost = 3000
        pos_z_cost = 5e2
        vel_xy_cost = 100
        vel_z_cost = 2e1
        ang_cost = 1e1
        Q_mat = 2 * np.diag([pos_xy_cost, pos_xy_cost, pos_z_cost,
                             vel_xy_cost, vel_xy_cost, vel_z_cost,
                             ang_cost, ang_cost, 1e2])       # euler error (yaw weighted higher)

        pos_xy_cost_e = 7e2
        pos_z_cost_e = 5e2
        vel_xy_cost_e = 5e1
        vel_z_cost_e = 2e1
        ang_cost_e = 1e1
        Q_e = np.diag([pos_xy_cost_e, pos_xy_cost_e, pos_z_cost_e,
                       vel_xy_cost_e, vel_xy_cost_e, vel_z_cost_e,
                       ang_cost_e, ang_cost_e, 1e2])

        R_mat = np.diag([1e1, 1e3, 1e3, 1e2])  # thrust, p, q, r rates

        ocp.cost.W = block_diag(Q_mat, R_mat)  # 13x13
        ocp.cost.W_e = Q_e  # 9x9

        # Error-based cost expressions (output errors, not absolute states)
        ocp.model.cost_y_expr = cost_y_expr      # [p_err, v_err, euler_err, u_err]
        ocp.model.cost_y_expr_e = cost_y_expr_e  # [p_err, v_err, euler_err]

        # yref is ZERO because cost expression already outputs errors
        ocp.cost.yref = np.zeros(ny)    # 13D zeros
        ocp.cost.yref_e = np.zeros(ny_e)  # 9D zeros

        # Initialize parameters (will be updated at runtime)
        ocp.parameter_values = np.zeros(np_param)  # 13D: [p_ref, v_ref, euler_ref, u_ref]

        # Solver and integrator options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'

        use_RTI = True
        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 5
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        # --- Constraints ---
        # Control bounds (using platform-specific parameters)
        min_thrust = 0.0
        ocp.constraints.lbu = np.array([min_thrust, -self.max_body_rate, -self.max_body_rate, -self.max_body_rate])
        ocp.constraints.ubu = np.array([self.max_thrust_n, self.max_body_rate, self.max_body_rate, self.max_body_rate])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Initial state constraint (will be updated at runtime)
        pos = np.array([0.1, 0.1, 0.1])
        vel = np.array([0.2, 0.01, 0.1])
        euler0 = np.array([0.1, -0.1, 0.04])
        x0 = np.hstack((pos, vel, euler0))
        ocp.constraints.x0 = x0

        # --- Generate solver ---
        AcadosOcpSolver.generate(ocp, json_file=self.json_path)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)

        # Import compiled module
        mod_path = os.path.join(self.code_export_directory, "acados_ocp_solver_pyx.so")
        spec = importlib.util.spec_from_file_location("acados_ocp_solver_pyx", mod_path)  # type: ignore
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)
        acados_ocp_solver_pyx = mod

        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(
            self.model_name, 'SQP', self.num_steps
        )

    def solve_mpc_control(self, x0, xd, last_u, nx, nu, verbose=False, u_ref_traj=None):
        """
        Solve the MPC optimization problem with error-based cost.

        Args:
            x0: Current state (9D) [p, v, euler]
            xd: Reference trajectory (N, 9) for each stage [p_ref, v_ref, euler_ref]
            last_u: Last control input for warm-start
            nx: State dimension (9)
            nu: Control dimension (4)
            verbose: Print solver statistics
            u_ref_traj: Optional feedforward control reference (N, 4) [f_N, roll_rate, pitch_rate, yaw_rate].
                        If None, hover control is used as reference for all stages.

        Returns:
            simU: Optimal control sequence (N, 4)
            simX: Predicted state trajectory (N+1, 9)
            status: Solver status (0 = success)
        """
        N = self.num_steps
        if xd.shape[0] != N:
            raise ValueError(f"Reference trajectory length {xd.shape[0]} != horizon steps {N}")

        ocp_solver = self.ocp_solver

        # Set initial state constraints
        ocp_solver.set(0, "lbx", x0)  # type: ignore
        ocp_solver.set(0, "ubx", x0)  # type: ignore

        # Warm start with last control
        ocp_solver.set(0, "u", last_u)  # type: ignore

        # Pack stage-wise reference parameters.
        #
        # The Acados model exposes a 13D parameter vector p at every stage:
        #   p = [p_ref(3), v_ref(3), euler_ref(3), u_ref(4)]
        #
        # The cost function then computes errors relative to these values:
        #   p_err    = p    - p_ref       (position error)
        #   v_err    = v    - v_ref       (velocity error)
        #   euler_err= euler- euler_ref   (attitude error, yaw is wrapped)
        #   u_err    = u    - u_ref       (control error)
        #
        # Stage cost = weighted sum of squared errors over all 13 dimensions.
        # Terminal cost = same but without the u_err term (9D).
        #
        # Feedforward enters through two of these four terms:
        #
        #  1. euler_ref (columns 6:9 of xd) — when feedforward is active these
        #     hold the physically correct roll/pitch/yaw that flat-output inversion
        #     predicts the trajectory needs.  This makes euler_err penalise
        #     deviation from the correct attitude rather than from flat hover.
        #
        #  2. u_ref — when u_ref_traj is provided it carries the feedforward
        #     control [df, dphi, dth, dpsi] at each stage.  Without feedforward
        #     u_ref = hover_ctrl = [m*g, 0, 0, 0], which biases the optimiser
        #     toward rest.  With feedforward it is biased toward the nominal
        #     control the trajectory demands, so the optimiser only needs to find
        #     the residual correction rather than the full control from scratch.
        for i in range(N):
            p_ref     = xd[i, 0:3]
            v_ref     = xd[i, 3:6]
            euler_ref = xd[i, 6:9]
            u_ref = np.array(u_ref_traj[i]) if u_ref_traj is not None else self.hover_ctrl

            param_i = np.hstack((p_ref, v_ref, euler_ref, u_ref))
            ocp_solver.set(i, 'p', param_i)  # type: ignore

        # Terminal stage: same structure but Acados uses it for the terminal cost
        # (no u_err term, but u_ref is still required in the parameter vector).
        p_ref_e     = xd[-1, 0:3]
        v_ref_e     = xd[-1, 3:6]
        euler_ref_e = xd[-1, 6:9]
        u_ref_e = np.array(u_ref_traj[-1]) if u_ref_traj is not None else self.hover_ctrl
        param_e = np.hstack((p_ref_e, v_ref_e, euler_ref_e, u_ref_e))
        ocp_solver.set(N, 'p', param_e)  # type: ignore

        # Solve the OCP
        status = ocp_solver.solve()  # type: ignore
        if verbose:
            self.ocp_solver.print_statistics()  # type: ignore

        if status != 0:
            raise Exception(f'acados returned status {status}.')

        # Extract solution
        simX = np.ndarray((N + 1, nx))
        simU = np.ndarray((N, nu))

        for i in range(N):
            simX[i, :] = self.ocp_solver.get(i, "x")  # type: ignore
            simU[i, :] = self.ocp_solver.get(i, "u")  # type: ignore
        simX[N, :] = self.ocp_solver.get(N, "x")  # type: ignore

        return simU, simX, status

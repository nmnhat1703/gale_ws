# TODO — Known Issues & Blockers

## Before real_s500 build

**BLOCKER**: Platform-specific solver directory naming

The generated solver C code is currently placed in a directory named
`holybro_euler_err_mpc_c_generated_code` (inherited from upstream).
If both sim_iris and real_s500 are built, they will write to the same
directory and clobber each other.

**Action**:
1. Rename/reorganize so each platform gets its own solver directory:
   - `c_generated_code_sim_iris/`
   - `c_generated_code_real_s500/`
2. Update `solver_setup.py` to compute the platform-specific path
3. Update `nmpc_node.py` to load the correct platform-specific solver on startup
4. Ensure no hardcoded directory names in the ROS node

**Status**: Not yet fixed. Blocks real_s500 solver generation / hardware testing.

---

## Phase 3 (future)

- [ ] Trajectory generator (figure-8, circle, etc.) — currently only hover
- [ ] Feedforward control via JAX differential-flatness (ported from upstream)
- [ ] System ID / auto-tune for inertia refinement
- [ ] Observability cost (visual servoing / external perception enhancement)

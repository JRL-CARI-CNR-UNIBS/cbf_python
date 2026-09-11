"""
Low-level accelerated computational kernels for CBF distance calculation and QP assembly.
"""

from cbf_python.controllers.kernels.ssm_cbf_acc import (
    dmin_and_jacobian_numba,
    h_and_jacobian_numba,
    jacobian_psi_numba,
    jacobian_psi_times_fg_fast_numba,
    compute_h_and_lie_numba,
    compute_h_and_constraints_numba,
)
from cbf_python.controllers.kernels.numba_kernels import (
    build_free_forced_one_step,
    fill_scaling_rows,
    fill_pos_rows,
    fill_tube_rows,
    fill_vel_rows,
    fill_acc_rows,
    append_cbf_rows_loop,
    assemble_objective_parts_inplace,
    assemble_qp_inplace,
)
from cbf_python.controllers.kernels.cbf_numba_lib import (
    assemble_qp_PID_problem,
    compute_q_ref_from_goal,
    damped_pinv_svd,
)

__all__ = [
    "dmin_and_jacobian_numba",
    "h_and_jacobian_numba",
    "jacobian_psi_numba",
    "jacobian_psi_times_fg_fast_numba",
    "compute_h_and_lie_numba",
    "compute_h_and_constraints_numba",
    "build_free_forced_one_step",
    "fill_scaling_rows",
    "fill_pos_rows",
    "fill_tube_rows",
    "fill_vel_rows",
    "fill_acc_rows",
    "append_cbf_rows_loop",
    "assemble_objective_parts_inplace",
    "assemble_qp_inplace",
    "assemble_qp_PID_problem",
    "compute_q_ref_from_goal",
    "damped_pinv_svd",
]

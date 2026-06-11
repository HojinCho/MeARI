ctypedef enum LIA_ORDER:
    RowMajor
    ColMajor

ctypedef enum LIA_TRANS:
    NoTrans
    Trans
    ConjTrans
    ConjNoTrans

ctypedef enum LIA_UPLO:
    Up
    Lo

ctypedef enum LIA_DIAG:
    NonUnit
    Unit

ctypedef enum LIA_SIDE:
    Left
    Right

from ..utils.types cimport complex_double

################################################################################
#
#     LiA: Linear Algebra 
#             Hojin Cho, 2025-01-15.
#  Common interface for linear algebra routines necessary for the computation
#    of the likelihood functions, including PRH-Q.
#  Reference backend is implemented with CBLAS/LAPACKE (typically provided by 
#    OpenBLAS via conda-forge) for portability. However, targeted backends 
#    such as cuBLAS and MKL are planned to be implemented for the simulation.
#
################################################################################

# from .lia.common cimport (
#     index_packed, index_loop,
#     lia_idx_tri_row_u, lia_idx_tri_row_l, lia_idx_tri_col_u, lia_idx_tri_col_l,
#     lia_idx_tri_u_i, lia_idx_tri_u_f, lia_idx_tri_l_i, lia_idx_tri_l_f,
# )
#### Index helper for packed storage of symmetric matrices
ctypedef size_t (*index_packed)(size_t i, size_t j, size_t n) noexcept nogil
cdef size_t lia_idx_tri_row_u(size_t i, size_t j, size_t n) noexcept nogil
cdef size_t lia_idx_tri_row_l(size_t i, size_t j, size_t n) noexcept nogil
cdef size_t lia_idx_tri_col_u(size_t i, size_t j, size_t n) noexcept nogil
cdef size_t lia_idx_tri_col_l(size_t i, size_t j, size_t n) noexcept nogil
    #### "Safe" version of the same index functions; in case if it is not clear what to use.
cdef size_t lia_safeidx_tri_row_u(size_t i, size_t j, size_t n) noexcept nogil
cdef size_t lia_safeidx_tri_col_u(size_t i, size_t j, size_t n) noexcept nogil
cdef size_t lia_safeidx_tri_row_l(size_t i, size_t j, size_t n) noexcept nogil
cdef size_t lia_safeidx_tri_col_l(size_t i, size_t j, size_t n) noexcept nogil
#### Index helper for the inner loop [i]nitial and [f]inal for packed storage.
ctypedef size_t (*index_loop)(size_t i, size_t n) noexcept nogil
cdef size_t lia_idx_tri_u_i(size_t i, size_t n) noexcept nogil
cdef size_t lia_idx_tri_u_f(size_t i, size_t n) noexcept nogil
cdef size_t lia_idx_tri_l_i(size_t i, size_t n) noexcept nogil
cdef size_t lia_idx_tri_l_f(size_t i, size_t n) noexcept nogil

################################################################################
#
#  Operator Naming Conventions.
# 
#  Each function is named to identify the kind of operation quickly.
#    The name always starts with lia. If the function is specific to certain
#      dimensions, (n)d is appended right after lia. Then an _ is followed.
#          lia_   : Operations that can be applied to any dimensions
#          lia2d_ : 2-dimensional operations    
#          lia3d_ : 3-dimensional operations
#    After an _, the domain of the function is presented in terms of 
#      the following symbols:
#              s: Scalar          v: Vector          m: Matrix
#    Then, an _ is appended followed by the name describing the function.
#      E.g., 
#        1. For a scalar a and a vector X, 
#           Y = a*X has a domain sign of _sv_
#        2. For scalars a and b, and vector X and Y, 
#           Z = a*X+b*Y has a domain sign of _svsv_
#      E.g., _vv_ means binary operation between two vectors. But this does
#        not specify the output domain.
#      E.g., _mv_ means binary operation between a matrix and a vector.
#      E.g., _mvv_ would usually mean affine transformation Ax+y, 
#        while _vmv_ would usually mean a inner product between two vectors
#        with m being a positive-definite matrix (e.g., covariance matrix).
#    At the end of the function name, _A_ is appended if the function can be 
#      applied to array of operands. Followed by it is how it is applied to
#      each member of the array, similar to the einsum convention.
#        0: Has only one element   i: iterated along i-th axis
#      E.g., ii means two operands has the same number of operands, follow the 
#        same axis and result in the same number of elements as the inputs.
#      E.g., 0i means the first operand is fixed and applied to each element of
#        the second operand.
#      E.g., ij means two operand may have different number of elements, 
#        and it will return a tensor that is one-rank higher than the result 
#        of the non-array variant of the function.
#      E.g, lia_vv_*_A_ij will result in a vector or matrix, depending on 
#        the nature of the operation. 
#        - If lia_vv_* is V x V -> S, then lia_vv_*_A_ij is V x V -> W.
#        - If lia_vv_* is V x V -> V, then lia_vv_*_A_ij is V x V -> M.
#      E.g, For an array of N scalars *a and an array of N vectors *X, 
#        Y[i] = a[i]*X[i] for i in range(N) can either be identified as 
#        _sv_ with _A_ii with ndim=*, or _vv_ without _A signature. 
#        Therefore, sometimes a same function can be achieved with different
#        approaches. However, the implementation behind each approach may
#        be different, affecting the performance of the operation.
#    
#    Unary operations and other n-ary operations have _inplace variants,
#      but this only makes sense when the operation can obviously defined. 
#      For n-ary operations, it is always the first operand that is modified.
# 
################################################################################
################################################################################
# Unary Operations 
################################################################################
##### Vector Unary #####
cdef double lia_v_norm(double * x, size_t n, int stride_x=*) noexcept nogil
cdef void lia_v_norm_A_i(double * out, double * x, size_t ndim, size_t nsize, int stride_x=*) noexcept nogil
cdef void lia_v_comp(double * dst, double * src, int axis, size_t ndim, size_t nsize, int stride_dst=*, int stride_src=*) noexcept nogil
##### Matrix Unary #####
cdef void lia_m_packed_to_full_sym(double * out, double * symmat, size_t ndim, bint upper=*, int lda=*) noexcept nogil
cdef void lia_m_full_to_packed_sym(double * out, double * full, size_t ndim, bint upper=*, int lda=*) noexcept nogil
##### LAPACKE routines.
#### Bunch-Kaufman diagonalization (L^TDL, U^TDU)
cdef int lia_m_BK_sym_factorize_inplace(double * full, int * ipiv, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_BK_sym_factorize(double * out, int * ipiv, double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef double lia_m_logdet_sym_BK(double * a, int * ipiv, size_t ndim, int * sign, size_t lda=*) noexcept nogil
cdef double lia_m_det_sym_BK(double * a, int * ipiv, size_t ndim, size_t lda=*) noexcept nogil
cdef int lia_m_inv_sym_BK_inplace(double * full, int * ipiv, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_inv_sym_BK(double * out, int * ipiv, double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef long lia_m_BK_sym_query_lwork(double * full, int * ipiv, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_inv_sym_BK_bf_inplace(double * full, int * ipiv, double * work, int lwork, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_inv_sym_BK_bf(double * out, int * ipiv, double * full, double * work, int lwork, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
### Full storage
cdef int lia_m_cholesky_sym_inplace(double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_cholesky_sym_norm_inplace(double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_cholesky_sym(double * out, double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_cholesky_sym_norm(double * out, double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef double lia_m_logsqrtdet_sym_cholesky(double * chol, size_t ndim) noexcept nogil
cdef double lia_m_sqrtdet_sym_cholesky(double * chol, size_t ndim) noexcept nogil
cdef double lia_m_det_sym_cholesky(double * chol, size_t ndim) noexcept nogil
# cdef double lia_m_det_sym_bf(double * full, size_t ndim, double * buffer, size_t lda=*, bint upper=*) noexcept nogil
# cdef double lia_m_det_sym(double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_inv_sym_cholesky_inplace(double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
cdef int lia_m_inv_sym_cholesky(double * out, double * chol, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
# cdef int lia_m_inv_sym_inplace(double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
# cdef int lia_m_inv_sym(double * out, double * full, size_t ndim, size_t lda=*, bint upper=*) noexcept nogil
### Packed storage
cdef int lia_m_cholesky_sympack_inplace(double * packed, size_t ndim, bint upper=*) noexcept nogil
cdef int lia_m_cholesky_sympack(double * out, double * sym, size_t ndim, bint upper=*) noexcept nogil
cdef double lia_m_sqrtdet_sympack_cholesky(double * chol, size_t ndim, bint upper=*) noexcept nogil
cdef double lia_m_det_sympack_cholesky(double * chol, size_t ndim, bint upper=*) noexcept nogil
# cdef double lia_m_det_sympack_bf(double * sym, size_t ndim, double * buffer, bint upper=*) noexcept nogil
# cdef double lia_m_det_sympack(double * sym, size_t ndim, bint upper=*) noexcept nogil
cdef int lia_m_inv_sympack_cholesky_inplace(double * packed, size_t ndim, bint upper=*) noexcept nogil
cdef int lia_m_inv_sympack_cholesky(double * out, double * chol, size_t ndim, bint upper=*) noexcept nogil
# cdef int lia_m_inv_sympack_inplace(double * packed, size_t ndim, bint upper=*) noexcept nogil
# cdef int lia_m_inv_sympack(double * out, double * sym, size_t ndim, bint upper=*) noexcept nogil    
##### Dimension-specific Unary #####
cdef void lia3d_v_rho_R(double * rho, double * R, double * xyz, size_t nsize) noexcept nogil
################################################################################
# Binary Operations 
################################################################################
##### Scalar - Vector Binary #####
cdef void lia_sv_scalar_inplace(double alpha, double * x, size_t n, int stride_x=*) noexcept nogil
cdef void lia_sv_scalar_A_ii_inplace(double alpha, double * x, size_t ndim, size_t nsize, int stride_x=*) noexcept nogil
cdef void lia_sv_scalar(double * out, double alpha, double * x, size_t n, int stride_x=*, int stride_out=*) noexcept nogil
cdef void lia_sv_scalar_A_ii(double * out, double * alpha, double * x, size_t ndim, size_t nsize, int stride_x=*, int stride_a=*, int stride_out=*) noexcept nogil
##### Vector - Vector Binary #####
ctypedef double complex lia_complex
cdef double lia_vv_dot(double * x, double * y, size_t n, int stride_x=*, int stride_y=*) noexcept nogil
cdef complex_double lia_vv_dotc(complex_double * z, complex_double * w, size_t n, int stride_z=*, int stride_w=*) noexcept nogil
cdef void lia_vv_dot_A_0i(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_dot_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_cpy(double * dst, double * src, size_t n, int stride_dst=*, int stride_src=*) noexcept nogil
cdef void lia_vv_cpy_A_ii(double * dst, double * src, size_t ndim, size_t nsize, int stride_dst=*, int stride_src=*) noexcept nogil
cdef void lia_vv_swp(double * x, double * y, size_t n, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_swp_A_ii(double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
############# Basic Binary Operations #############
############################## Addition
cdef void lia_vv_add_inplace(double * dst, double * src, size_t ndim, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_add(double * out, double * x, double * y, size_t ndim, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_add_A_i0_inplace(double * dst, double * src, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_add_A_i0(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_add_A_ii_inplace(double * dst, double * src, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_add_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
############################## Subtraction
cdef void lia_vv_sub_inplace(double * dst, double * sub, size_t ndim, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_sub(double * out, double * x, double * y, size_t ndim, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_sub_A_i0_inplace(double * dst, double * sub, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_sub_A_i0(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_sub_A_ii_inplace(double * dst, double * sub, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_sub_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
############################## Multiplication
cdef void lia_vv_mul_inplace(double * dst, double * mul, size_t ndim, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_mul(double * out, double * x, double * y, size_t ndim, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_mul_A_i0_inplace(double * dst, double * src, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_mul_A_i0(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_mul_A_ii_inplace(double * dst, double * src, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_mul_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
############################## Division
cdef void lia_vv_div_inplace(double * dst, double * div, size_t ndim, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_div(double * out, double * x, double * y, size_t ndim, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_div_A_i0_inplace(double * dst, double * div, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_div_A_i0(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vv_div_A_ii_inplace(double * dst, double * div, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*) noexcept nogil
cdef void lia_vv_div_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
##### Matrix - Vector Binary #####
cdef void lia_mv_mul(double * out, double * mat, double * vec, size_t m, size_t n, bint T=*, int lda=*, int stride_vec=*, int stride_out=*) noexcept nogil
cdef void lia_mv_mul_A_0i(double * out, double * mat, double * vec, size_t m, size_t ndim, size_t nsize, bint T=*, int lda=*, int stride_vec=*, int stride_out=*) noexcept nogil
cdef void lia_mv_mul_A_ii(double * out, double * mat, double * vec, size_t m, size_t ndim, size_t nsize, bint T=*, int lda=*, int stride_vec=*, int stride_out=*) noexcept nogil
# Special case of symmetric matrix
cdef void lia_mv_mul_sym(double * out, double * mat, double * vec, size_t n, bint upper=*, int stride_vec=*, int stride_out=*) noexcept nogil
##### LAPACKE routines.
cdef void lia_mv_solve_bf_inplace(double * mat, double * vec, size_t n, int * ipiv, size_t nvec=*, int lda=*, bint upper=*) noexcept nogil
cdef void lia_mv_solve_inplace(double * mat, double * vec, size_t n, size_t nvec=*, int lda=*, bint upper=*) noexcept nogil
cdef void lia_mv_solve_bf(double * out, double * mat, double * vec, size_t n, int * ipiv, size_t nvec=*, int lda=*, bint upper=*) noexcept nogil
cdef void lia_mv_solve(double * out, double * mat, double * vec, size_t n, size_t nvec=*, int lda=*, bint upper=*) noexcept nogil
##### Dimension-specific Binary #####
cdef void lia2d_mv_rot_inplace(double * vec, double c, double s) noexcept nogil
cdef void lia2d_mv_rot_A_0i_inplace(double * vec, double c, double s, size_t n) noexcept nogil
cdef void lia2d_mv_rot(double *out, double * vec, double c, double s) noexcept nogil
cdef void lia2d_mv_rot_A_0i(double * out, double * vec, double c, double s, size_t n) noexcept nogil
cdef void lia3d_mv_rot_inplace(double * vec, double c, double s, int axis) noexcept nogil
cdef void lia3d_mv_rot_A_0i_inplace(double * vec, double c, double s, int axis, size_t n) noexcept nogil
cdef void lia3d_mv_rot(double *out, double * vec, double c, double s, int axis) noexcept nogil
cdef void lia3d_mv_rot_A_0i(double * out, double * vec, double c, double s, int axis, size_t n) noexcept nogil
################################################################################
# Ternary Operations 
################################################################################
##### Vector - Matrix - Vector Ternary #####
# Requires a buffer for efficient computation!
cdef double lia_vmv_quadform_bf(double * x, double * A, double * y, size_t m, size_t n, double * buffer, bint T=*, int lda=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vmv_quadform_A_i0i_bf(double * out, double * x, double * A, double * y, size_t m, size_t n, size_t ndata, double * buffer, bint T=*, int lda=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vmv_quadform_A_iii_bf(double * out, double * x, double * A, double * y, size_t m, size_t n, size_t ndata, double * buffer, bint T=*, int lda=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef double lia_vmv_quadform(double * x, double * A, double * y, size_t m, size_t n, bint T=*, int lda=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vmv_quadform_A_i0i(double * out, double * x, double * A, double * y, size_t m, size_t n, size_t ndata, bint T=*, int lda=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vmv_quadform_A_iii(double * out, double * x, double * A, double * y, size_t m, size_t n, size_t ndata, bint T=*, int lda=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
# Special case of symmetric matrix
cdef double lia_vmv_quadform_sym_bf(double * x, double * A, double * y, size_t n, double * buffer, bint upper=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef double lia_vmv_quadform_sym(double * x, double * A, double * y, size_t n, bint upper=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vmv_quadform_sym_A_i0i_bf(double * out, double * x, double * A, double * y, size_t n, size_t ndata, double * buffer, bint upper=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vmv_quadform_sym_A_i0i(double * out, double * x, double * A, double * y, size_t n, size_t ndata, bint upper=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vmv_quadform_sym_A_iii_bf(double * out, double * x, double * A, double * y, size_t n, size_t ndata, double * buffer, bint upper=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_vmv_quadform_sym_A_iii(double * out, double * x, double * A, double * y, size_t n, size_t ndata, bint upper=*, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
################################################################################
# Quarternary Operations 
################################################################################
##### 2 Scalars - 2 Vector Quarternary #####
cdef void lia_svsv_lincomb(double * out, double alpha, double * x, double beta, double * y, size_t n, int stride_x=*, int stride_y=*, int stride_out=*) noexcept nogil
cdef void lia_svsv_lincomb_A_0i0i(double * out, double alpha, double * x, double beta, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_a=*, int stride_b=*, int stride_out=*) noexcept nogil
cdef void lia_svsv_lincomb_A_iiii(double * out, double * alpha, double * x, double * beta, double * y, size_t ndim, size_t nsize, int stride_x=*, int stride_y=*, int stride_a=*, int stride_b=*, int stride_out=*) noexcept nogil
################################################################################
# Low-level (0) Direct BLAS interface
################################################################################
# This may be different if it uses non-standard BLAS implementations.
cdef void lia_0_dsymm(LIA_ORDER Order, LIA_SIDE Side, LIA_UPLO Uplo, int M, int N, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) noexcept nogil
cdef void lia_0_dgemm(LIA_ORDER Order, LIA_TRANS TransA, LIA_TRANS TransB, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) noexcept nogil
cdef void lia_0_dcopy(int n, double * x, int incx, double * y, int incy) noexcept nogil

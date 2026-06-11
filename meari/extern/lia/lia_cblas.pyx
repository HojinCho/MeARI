# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

################################################################################
#
#   LiA: Linear Algebra 
#        CBLAS and LAPACKE implementation
#     Hojin Cho, 2025-01-15.
#
################################################################################

cimport cython
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport uint64_t
from libc.math cimport sqrt, log, exp, fabs

from cython.parallel import prange

from ..utils.types cimport complex_double
from ..utils.algorithms cimport kahan_sum_iterator

from .lia.cblas cimport (
    # Really, no need for single precision routines.
    CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO, CBLAS_DIAG, CBLAS_SIDE,
    # Level 1
    cblas_dcopy, cblas_dswap, cblas_ddot, cblas_dnrm2, cblas_dscal, cblas_daxpy, cblas_drot, 
    cblas_zdotc, # For FFT correlation.
    # Level 2
    # cblas_dsbmv, # For element-wise vector multiplication
    cblas_dgemv, cblas_dsymv,
    # Level 3
    cblas_dgemm, cblas_dsymm,
)
from .lia.lapacke cimport (
    LAPACK_LAYOUT,  # LapackRowMajor = 101, LapackColMajor = 102
    LAPACKE_dpptrf, # Cholesky decomposition for symmetric packed positive-definite real matrix.
    LAPACKE_dpptri, # Inversion of symmetric packed positive-definite real matrix given its Cholesky decomposition.
    LAPACKE_dpotrf, # Cholesky decomposition for symmetric positive-definite real matrix.
    LAPACKE_dpotri, # Inversion of symmetric positive-definite real matrix given its Cholesky decomposition.
    LAPACKE_dsytrf,
    LAPACKE_dsytri,
    LAPACKE_dsytri2_work,
    LAPACKE_dgesv,
    lapack_int,
)
# https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pptrf.html
# https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pptri.html
# lapack_int LAPACKE_dpptrf(int matrix_layout, char uplo, lapack_int n, double* ap)
# lapack_int LAPACKE_dpptri(int matrix_layout, char uplo, lapack_int n, double* ap)
# lapack_int LAPACKE_dpotrf(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda)
# lapack_int LAPACKE_dpotri(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda)
# lapack_int LAPACKE_dsytrf(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda, lapack_int * ipiv) noexcept nogil
# lapack_int LAPACKE_dsytri(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda, lapack_int * ipiv) noexcept nogil
# lapack_int LAPACKE_dsytri2_work(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda, const lapack_int * ipiv, double * work, lapack_int lwork) noexcept nogil

# from ..utils.numpy_interface cimport numpy_dbl_1d

from .lia cimport (
    index_packed, index_loop,
    # For a shared interface
    LIA_ORDER, LIA_TRANS, LIA_UPLO, LIA_DIAG, LIA_SIDE, 
)
include "lia/lia_common.pyx"
# #### Index helper for packed storage of symmetric matrices
# cdef size_t lia_idx_tri_row_u(size_t i, size_t j, size_t n) noexcept nogil:
#     return j+i*(2*n-i-1)//2
# cdef size_t lia_idx_tri_col_u(size_t i, size_t j, size_t n) noexcept nogil:
#     return i+j*(j+1)//2
# cdef size_t lia_idx_tri_row_l(size_t i, size_t j, size_t n) noexcept nogil:
#     return j+i*(i+1)//2
# cdef size_t lia_idx_tri_col_l(size_t i, size_t j, size_t n) noexcept nogil:
#     return i+j*(2*n-j-1)//2
# #### Index helper for the inner loop [i]nitial and [f]inal for packed storage.
# cdef size_t lia_idx_tri_u_i(size_t i, size_t n) noexcept nogil:
#     return i
# cdef size_t lia_idx_tri_u_f(size_t i, size_t n) noexcept nogil:
#     return n
# cdef size_t lia_idx_tri_l_i(size_t i, size_t n) noexcept nogil:
#     return 0
# cdef size_t lia_idx_tri_l_f(size_t i, size_t n) noexcept nogil:
#     return i+1

################################################################################
# Unary Operations 
################################################################################

##### Vector Unary #####

cdef double lia_v_norm(double * x, size_t n, int stride_x=1) noexcept nogil:
    return cblas_dnrm2(n, x, stride_x) # ||x||

cdef void lia_v_norm_A_i(double * out, double * x, size_t ndim, size_t nsize, int stride_x=1) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        out[i] = cblas_dnrm2(ndim, x + i*ndim*stride_x, stride_x)
    
cdef void lia_v_comp(double * dst, double * src, int axis, size_t ndim, size_t nsize, int stride_dst=1, int stride_src=1) noexcept nogil:
    cblas_dcopy(nsize, src+axis%ndim, ndim*stride_src, dst, stride_dst) # dst = src[:, axis]

##### Matrix Unary #####

# cdef void lia_m_packed_to_full_sym_blas_id(double * out, double * mat, size_t ndim, double * identity_n, bint upper=True, int ldout=1) noexcept nogil:
#     if ldout<ndim:
#         ldout = ndim
#     cdef CBLAS_UPLO uplo = CBLAS_UPLO.CblasUpper
#     if not upper:
#         uplo = CBLAS_UPLO.CblasLower
#     cblas_dsymm(CBLAS_ORDER.CblasRowMajor, CBLAS_SIDE.CblasLeft, uplo, ndim, ndim,
#                 1.0, mat, ndim, identity_n, ndim, 0.0, out, ldout)

# cdef void lia_m_packed_to_full_sym_blas(double * out, double * mat, size_t ndim, bint upper=True, int ldout=1) noexcept nogil:
#     cdef double * identity_n = <double *> calloc(ndim*ndim,sizeof(double))
#     cdef size_t i
#     for i in prange(ndim, nogil=True):
#         identity_n[i*(ndim+1)] = 1.0
#     lia_m_packed_to_full_sym_blas_id(out, mat, ndim, identity_n, upper=upper, ldout=ldout)
#     free(identity_n)

cdef void lia_m_packed_to_full_sym(double * out, double * symmat, size_t ndim, bint upper=True, int lda=1) noexcept nogil:
    cdef size_t i, j
    cdef index_packed idf
    cdef index_loop start, end
    if lda<ndim:
        lda = ndim
    if upper:
        idf = lia_idx_tri_row_u
        start = lia_idx_tri_u_i # i
        end   = lia_idx_tri_u_f # ndim
    else:
        idf = lia_idx_tri_row_l
        start = lia_idx_tri_l_i # 0
        end   = lia_idx_tri_l_f # i+1
    for i in prange(ndim, nogil=True):
        for j in range(start(i, ndim), end(i, ndim)):
            out[i*lda+j] = symmat[idf(i,j,ndim)]
            out[j*lda+i] = out[i*lda+j]

cdef void lia_m_full_to_packed_sym(double * out, double * full, size_t ndim, bint upper=True, int lda=1) noexcept nogil:
    cdef size_t i, j
    cdef index_packed idf
    cdef index_loop start, end
    if lda<ndim:
        lda = ndim
    if upper:
        idf = lia_idx_tri_row_u
        start = lia_idx_tri_u_i # i
        end   = lia_idx_tri_u_f # ndim
    else:
        idf = lia_idx_tri_row_l
        start = lia_idx_tri_l_i # 0
        end   = lia_idx_tri_l_f # i+1
    for i in prange(ndim, nogil=True):
        for j in range(start(i, ndim), end(i, ndim)):
            out[idf(i,j,ndim)] = full[i*lda+j]

cdef void lia_m_packed_to_full_tri(double * out, double * trimat, size_t ndim, bint upper=True, int lda=1) noexcept nogil:
    cdef size_t i, j
    cdef index_packed idf
    cdef index_loop start, end
    if lda<ndim:
        lda = ndim
    if upper:
        idf = lia_idx_tri_row_u
        start = lia_idx_tri_u_i # i
        end   = lia_idx_tri_u_f # ndim
    else:
        idf = lia_idx_tri_row_l
        start = lia_idx_tri_l_i # 0
        end   = lia_idx_tri_l_f # i+1
    for i in prange(ndim, nogil=True):
        for j in range(start(i, ndim), end(i, ndim)):
            out[i*lda+j] = trimat[idf(i,j,ndim)]
            out[j*lda+i] = 0

cdef void lia_m_full_to_packed_tri(double * out, double * full, size_t ndim, bint upper=True, int lda=1) noexcept nogil:
    cdef size_t i, j
    cdef index_packed idf
    cdef index_loop start, end
    if lda<ndim:
        lda = ndim
    if upper:
        idf = lia_idx_tri_row_u
        start = lia_idx_tri_u_i # i
        end   = lia_idx_tri_u_f # ndim
    else:
        idf = lia_idx_tri_row_l
        start = lia_idx_tri_l_i # 0
        end   = lia_idx_tri_l_f # i+1
    for i in prange(ndim, nogil=True):
        for j in range(start(i, ndim), end(i, ndim)):
            out[idf(i,j,ndim)] = full[i*lda+j]

##### LAPACKE routines.
#### Bunch-Kaufman diagonalization (L^TDL, U^TDU)
cdef int lia_m_BK_sym_factorize_inplace(double * full, int * ipiv, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    cdef char uplo=b'U'
    if not upper:
        uplo=b'L'
    if lda<ndim:
        lda=ndim
    return LAPACKE_dsytrf(LAPACK_LAYOUT.LapackRowMajor, uplo, ndim, full, lda, <lapack_int *> ipiv)

cdef int lia_m_BK_sym_factorize(double * out, int * ipiv, double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    if lda<ndim:
        lda=ndim
    cblas_dcopy(ndim*lda, full, 1, out, 1)
    return lia_m_BK_sym_factorize_inplace(out, ipiv, ndim, lda=lda, upper=upper)

cdef double lia_m_logdet_sym_BK(double * a, int * ipiv, size_t ndim, int * sign, size_t lda=1) noexcept nogil:
    if lda<ndim:
        lda=ndim
    cdef size_t i
    cdef double logdet = 0.
    cdef double buffer
    sign[0] = 1
    # for i in range(ndim):
    i=0
    while True:
        if i>=ndim:
            break
        if ipiv[i]>0:
            buffer = a[i*(lda+1)]
            if buffer<0:
                sign[0] *=-1
            logdet += log(fabs(buffer))
            i += 1
        else:
            buffer = a[i*(lda+1)]*a[(i+1)*(lda+1)] - a[i*(lda+1)+1]*a[i*(lda+1)+1]
            if buffer<0:
                sign[0] *=-1
            logdet += log(fabs(buffer))
            i += 2
    return logdet

cdef double lia_m_det_sym_BK(double * a, int * ipiv, size_t ndim, size_t lda=1) noexcept nogil:
    cdef int sign
    cdef double exponent = lia_m_logdet_sym_BK(a, ipiv, ndim, &sign, lda=lda)
    return sign*exp(exponent)

cdef int lia_m_inv_sym_BK_inplace(double * full, int * ipiv, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    cdef char uplo=b'U'
    if not upper:
        uplo=b'L'
    if lda<ndim:
        lda=ndim
    return LAPACKE_dsytri(LAPACK_LAYOUT.LapackRowMajor, uplo, ndim, full, lda, <lapack_int *> ipiv)

cdef int lia_m_inv_sym_BK(double * out, int * ipiv, double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    if lda<ndim:
        lda=ndim
    cblas_dcopy(ndim*lda, full, 1, out, 1)
    return lia_m_inv_sym_BK_inplace(out, ipiv, ndim, lda=lda, upper=upper)

cdef long lia_m_BK_sym_query_lwork(double * full, int * ipiv, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    cdef char uplo=b'U'
    if not upper:
        uplo=b'L'
    if lda<ndim:
        lda=ndim
    cdef double [1] work_query
    LAPACKE_dsytri2_work(LAPACK_LAYOUT.LapackRowMajor, uplo, ndim, full, lda, <lapack_int *> ipiv, &work_query[0], -1)
    return <long>(work_query[0])

cdef int lia_m_inv_sym_BK_bf_inplace(double * full, int * ipiv, double * work, int lwork, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    cdef char uplo=b'U'
    if not upper:
        uplo=b'L'
    if lda<ndim:
        lda=ndim
    return LAPACKE_dsytri(LAPACK_LAYOUT.LapackRowMajor, uplo, ndim, full, lda, <lapack_int *> ipiv)

cdef int lia_m_inv_sym_BK_bf(double * out, int * ipiv, double * full, double * work, int lwork, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    if lda<ndim:
        lda=ndim
    cblas_dcopy(ndim*lda, full, 1, out, 1)
    return lia_m_inv_sym_BK_bf_inplace(out, ipiv, work, lwork, ndim, lda=lda, upper=upper)

#### Cholesky decomposition
### Full storage
cdef int lia_m_cholesky_sym_inplace(double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    # packed: 
    #   in:  sym
    #   out: cholesky
    cdef char uplo=b'U'
    if not upper:
        uplo=b'L'
    if lda<ndim:
        lda=ndim
    # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pptrf.html
    return LAPACKE_dpotrf(LAPACK_LAYOUT.LapackRowMajor, uplo, ndim, full, lda)

cdef int lia_m_cholesky_sym_norm_inplace(double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    # Currently don't implement if out and full has different ld
    if lda<ndim:
        lda=ndim
    cdef size_t i
    cdef double scale = 0
    for i in range(ndim):
        scale += log(fabs(full[i*(ndim+1)]))
    cblas_dscal(ndim*lda, exp(-scale/ndim), full, 1) # x = alpha * x
    cdef int outval = lia_m_cholesky_sym_inplace(full, ndim, lda=lda, upper=upper)
    cblas_dscal(ndim*lda, exp(+0.5*scale/ndim), full, 1) # x = alpha * x
    return outval

cdef int lia_m_cholesky_sym(double * out, double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    # Currently don't implement if out and full has different ld
    if lda<ndim:
        lda=ndim
    cblas_dcopy(ndim*lda, full, 1, out, 1)
    return lia_m_cholesky_sym_inplace(out, ndim, lda=lda, upper=upper)

cdef int lia_m_cholesky_sym_norm(double * out, double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    # Currently don't implement if out and full has different ld
    if lda<ndim:
        lda=ndim
    cblas_dcopy(ndim*lda, full, 1, out, 1)
    cdef size_t i
    cdef double scale = 0
    for i in range(ndim):
        scale += log(fabs(out[i*(ndim+1)]))
    cblas_dscal(ndim*lda, exp(-scale/ndim), out, 1) # x = alpha * x
    cdef int outval = lia_m_cholesky_sym_inplace(out, ndim, lda=lda, upper=upper)
    cblas_dscal(ndim*lda, exp(+0.5*scale/ndim), out, 1) # x = alpha * x
    return outval

cdef double lia_m_logsqrtdet_sym_cholesky(double * chol, size_t ndim) noexcept nogil:
    cdef double logsqrtdet = 0
    # Replace with Kahan_Sum function?
    cdef double c = 0
    cdef size_t i
    for i in range(ndim):
        kahan_sum_iterator(&logsqrtdet, log(chol[i*(ndim+1)]), &c) 
        # logsqrtdet += log(chol[i*(ndim+1)])
    return logsqrtdet

cdef double lia_m_sqrtdet_sym_cholesky(double * chol, size_t ndim) noexcept nogil:
    # cdef double sqrtdet = 1.
    # cdef size_t i
    # for i in range(ndim):
    #     sqrtdet *= chol[i*(ndim+1)]
    # return sqrtdet
    return exp(lia_m_logsqrtdet_sym_cholesky(chol, ndim)) # safer from FP overflow

cdef double lia_m_det_sym_cholesky(double * chol, size_t ndim) noexcept nogil:
    # cdef double sqrtdet = lia_m_sqrtdet_sym_cholesky(chol, ndim)
    # return sqrtdet*sqrtdet
    return exp(2*lia_m_logsqrtdet_sym_cholesky(chol, ndim)) # safer from FP overflow

# cdef double lia_m_det_sym_bf(double * full, size_t ndim, double * buffer, size_t lda=1, bint upper=True) noexcept nogil:
#     lia_m_cholesky_sym(buffer, full, ndim, lda=lda, upper=upper)
#     return lia_m_det_sym_cholesky(buffer, ndim)

# cdef double lia_m_det_sym(double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
#     if lda<ndim:
#         lda=ndim
#     cdef double det
#     cdef double * chol = <double *> malloc(ndim*lda*sizeof(double))
#     det = lia_m_det_sym_bf(full, ndim, chol, lda=lda, upper=upper)
#     free(chol)
#     return det

cdef int lia_m_inv_sym_cholesky_inplace(double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    # packed: 
    #   in:  cholesky
    #   out: inverse of sym
    cdef char uplo=b'U'
    if not upper:
        uplo=b'L'
    if lda<ndim:
        lda=ndim
    # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pptri.html
    return LAPACKE_dpotri(LAPACK_LAYOUT.LapackRowMajor, uplo, ndim, full, lda)

cdef int lia_m_inv_sym_cholesky(double * out, double * chol, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
    # Currently don't implement if out and full has different ld
    if lda<ndim:
        lda=ndim
    cblas_dcopy(ndim*lda, chol, 1, out, 1)
    return lia_m_inv_sym_cholesky_inplace(out, ndim, lda=lda, upper=upper)

# cdef int lia_m_inv_sym_inplace(double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
#     # packed: 
#     #   in:  sym
#     #   out: inverse of sym
#     lia_m_cholesky_sym_inplace(full, ndim, lda=lda, upper=upper)
#     return lia_m_inv_sym_cholesky_inplace(full, ndim, lda=lda, upper=upper)

# cdef int lia_m_inv_sym(double * out, double * full, size_t ndim, size_t lda=1, bint upper=True) noexcept nogil:
#     if lda<ndim:
#         lda=ndim
#     cblas_dcopy(ndim*lda, full, 1, out, 1)
#     return lia_m_inv_sym_inplace(out, ndim, lda=lda, upper=upper)

### Packed storage
# This is by default slower than unpacked 
# https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/dpptrf-performance/m-p/986053
cdef int lia_m_cholesky_sympack_inplace(double * packed, size_t ndim, bint upper=True) noexcept nogil:
    # packed: 
    #   in:  sym
    #   out: cholesky
    cdef char uplo=b'U'
    if not upper:
        uplo=b'L'
    # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pptrf.html
    return LAPACKE_dpptrf(LAPACK_LAYOUT.LapackRowMajor, uplo, ndim, packed)

cdef int lia_m_cholesky_sympack(double * out, double * sym, size_t ndim, bint upper=True) noexcept nogil:
    cblas_dcopy(ndim*(ndim+1)//2, sym, 1, out, 1)
    return lia_m_cholesky_sympack_inplace(out, ndim, upper=upper)

cdef double lia_m_sqrtdet_sympack_cholesky(double * chol, size_t ndim, bint upper=True) noexcept nogil:
    cdef index_packed idf
    cdef double sqrtdet = 1.
    if upper:
        idf = lia_idx_tri_row_u
    else:
        idf = lia_idx_tri_row_l
    for i in range(ndim):
        sqrtdet *= chol[idf(i,i,ndim)]
    return sqrtdet

cdef double lia_m_det_sympack_cholesky(double * chol, size_t ndim, bint upper=True) noexcept nogil:
    cdef double sqrtdet = lia_m_sqrtdet_sympack_cholesky(chol, ndim, upper=upper)
    return sqrtdet*sqrtdet

# cdef double lia_m_det_sympack_bf(double * sym, size_t ndim, double * buffer, bint upper=True) noexcept nogil:
#     lia_m_cholesky_sympack(buffer, sym, ndim, upper=upper)
#     return lia_m_det_sympack_cholesky(buffer, ndim, upper=upper)

# cdef double lia_m_det_sympack(double * sym, size_t ndim, bint upper=True) noexcept nogil:
#     cdef double det
#     cdef double * chol = <double *> malloc(ndim*(ndim+1)//2*sizeof(double))
#     det = lia_m_det_sympack_bf(sym, ndim, chol, upper=upper)
#     free(chol)
#     return det

cdef int lia_m_inv_sympack_cholesky_inplace(double * packed, size_t ndim, bint upper=True) noexcept nogil:
    # packed: 
    #   in:  cholesky
    #   out: inverse of sym
    cdef char uplo=b'U'
    if not upper:
        uplo=b'L'
    # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pptri.html
    return LAPACKE_dpptri(LAPACK_LAYOUT.LapackRowMajor, uplo, ndim, packed)

cdef int lia_m_inv_sympack_cholesky(double * out, double * chol, size_t ndim, bint upper=True) noexcept nogil:
    cblas_dcopy(ndim*(ndim+1)//2, chol, 1, out, 1)
    return lia_m_inv_sympack_cholesky_inplace(out, ndim, upper=upper)

# cdef int lia_m_inv_sympack_inplace(double * packed, size_t ndim, bint upper=True) noexcept nogil:
#     # packed: 
#     #   in:  sym
#     #   out: inverse of sym
#     lia_m_cholesky_sympack_inplace(packed, ndim, upper=upper)
#     return lia_m_inv_sympack_cholesky_inplace(packed, ndim, upper=upper)

# cdef int lia_m_inv_sympack(double * out, double * sym, size_t ndim, bint upper=True) noexcept nogil:
#     cblas_dcopy(ndim*(ndim+1)//2, sym, 1, out, 1)
#     return lia_m_inv_sympack_inplace(out, ndim, upper=upper)
    
##### Dimension-specific Unary #####

cdef void lia3d_v_rho_R(double * rho, double * R, double * xyz, size_t nsize) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        rho[i] = cblas_dnrm2(2, xyz + i*3, 1)
        R[i]   = cblas_dnrm2(3, xyz + i*3, 1)
    # cdef long i, I
    # cdef double buffer
    # for i in range(nsize):
    #     I = 3*i
    #     buffer = xyz[I  ]*xyz[I  ] + xyz[I+1]*xyz[I+1]
    #     rho[i] = sqrt(buffer)
    #     R[i] = sqrt(buffer + xyz[I+2]*xyz[I+2])

################################################################################
# Binary Operations 
################################################################################

##### Scalar - Vector Binary #####

cdef void lia_sv_scalar_inplace(double alpha, double * x, size_t n, int stride_x=1) noexcept nogil:
    cblas_dscal(n, alpha, x, stride_x) # x = alpha * x

cdef void lia_sv_scalar_A_ii_inplace(double alpha, double * x, size_t ndim, size_t nsize, int stride_x=1) noexcept nogil:
    lia_sv_scalar_inplace(alpha, x, ndim*nsize, stride_x)

cdef void lia_sv_scalar(double * out, double alpha, double * x, size_t n, int stride_x=1, int stride_out=1) noexcept nogil:
    cblas_dcopy(n, x, stride_x, out, stride_out)     # out = (x)
    lia_sv_scalar_inplace(alpha, out, n, stride_out) # out = alpha * x

cdef void lia_sv_scalar_A_ii(double * out, double * alpha, double * x, size_t ndim, size_t nsize, int stride_x=1, int stride_a=1, int stride_out=1) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        lia_sv_scalar(out + i*ndim*stride_out, 
            alpha[i*stride_a], x + i*ndim*stride_x, ndim, stride_x, stride_out)

##### Vector - Vector Binary #####

cdef double lia_vv_dot(double * x, double * y, size_t n, int stride_x=1, int stride_y=1) noexcept nogil:
    return cblas_ddot(n, x, stride_x, y, stride_y) # x . y

cdef complex_double lia_vv_dotc(complex_double * z, complex_double * w, size_t n, int stride_z=1, int stride_w=1) noexcept nogil:
    return cblas_zdotc(n, <void *> z, stride_z, <void *> w, stride_w) # <x|y> = x^H.y

cdef void lia_vv_dot_A_0i(
    double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1,
) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        out[stride_out*i] = cblas_ddot(ndim, x, stride_x, y + i*ndim*stride_y, stride_y)

cdef void lia_vv_dot_A_ii(
    double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1,
) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        out[stride_out*i] = cblas_ddot(ndim, x + i*ndim*stride_x, stride_x, y + i*ndim*stride_y, stride_y)
        
cdef void lia_vv_cpy(double * dst, double * src, size_t n, int stride_dst=1, int stride_src=1) noexcept nogil:
    cblas_dcopy(n, src, stride_src, dst, stride_dst) # dst = src

cdef void lia_vv_cpy_A_ii(double * dst, double * src, size_t ndim, size_t nsize, int stride_dst=1, int stride_src=1) noexcept nogil:
    lia_vv_cpy(dst, src, ndim*nsize, stride_dst, stride_src)

cdef void lia_vv_swp(double * x, double * y, size_t n, int stride_x=1, int stride_y=1) noexcept nogil:
    cblas_dswap(n, x, stride_x, y, stride_y) # x, y = y, x

cdef void lia_vv_swp_A_ii(double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    lia_vv_swp(x, y, ndim*nsize, stride_x, stride_y)


############# Basic Binary Operations #############
# These have no gain when implemented with BLAS, or even slower.
############################## Addition
cdef void lia_vv_add_inplace(double * dst, double * src, size_t ndim, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t i
    for i in range(ndim):
        dst[i*stride_x] += src[i*stride_y]
cdef void lia_vv_add(double * out, double * x, double * y, size_t ndim, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t i
    for i in range(ndim):
        out[i*stride_out] = x[i*stride_x] + y[i*stride_y]
cdef void lia_vv_add_A_i0_inplace(double * dst, double * src, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            dst[(J+i)*stride_x] += src[i*stride_y]
cdef void lia_vv_add_A_i0(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            out[(J+i)*stride_out] = x[(J+i)*stride_x] + y[i*stride_y]
cdef void lia_vv_add_A_ii_inplace(double * dst, double * src, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            dst[(J+i)*stride_x] += src[(J+i)*stride_y]
cdef void lia_vv_add_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            out[(J+i)*stride_out] = x[(J+i)*stride_x] + y[(J+i)*stride_y]
############################## Subtraction
cdef void lia_vv_sub_inplace(double * dst, double * sub, size_t ndim, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t i
    for i in range(ndim):
        dst[i*stride_x] -= sub[i*stride_y]
cdef void lia_vv_sub(double * out, double * x, double * y, size_t ndim, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t i
    for i in range(ndim):
        out[i*stride_out] = x[i*stride_x] - y[i*stride_y]
cdef void lia_vv_sub_A_i0_inplace(double * dst, double * sub, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            dst[(J+i)*stride_x] -= sub[i*stride_y]
cdef void lia_vv_sub_A_i0(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            out[(J+i)*stride_out] = x[(J+i)*stride_x] - y[i*stride_y]
cdef void lia_vv_sub_A_ii_inplace(double * dst, double * sub, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            dst[(J+i)*stride_x] -= sub[(J+i)*stride_y]
cdef void lia_vv_sub_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            out[(J+i)*stride_out] = x[(J+i)*stride_x] - y[(J+i)*stride_y]
############################## Multiplication
# https://stackoverflow.com/a/13433038/4755229, but BLAS is slower.
cdef void lia_vv_mul_inplace(double * dst, double * src, size_t ndim, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t i
    for i in range(ndim):
        dst[i*stride_x] *= src[i*stride_y]
cdef void lia_vv_mul(double * out, double * x, double * y, size_t ndim, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t i
    for i in range(ndim):
        out[i*stride_out] = x[i*stride_x] * y[i*stride_y]
cdef void lia_vv_mul_A_i0_inplace(double * dst, double * src, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            dst[(J+i)*stride_x] *= src[i*stride_y]
cdef void lia_vv_mul_A_i0(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            out[(J+i)*stride_out] = x[(J+i)*stride_x] * y[i*stride_y]
cdef void lia_vv_mul_A_ii_inplace(double * dst, double * src, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            dst[(J+i)*stride_x] *= src[(J+i)*stride_y]
cdef void lia_vv_mul_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            out[(J+i)*stride_out] = x[(J+i)*stride_x] * y[(J+i)*stride_y]
############################## Division
cdef void lia_vv_div_inplace(double * dst, double * div, size_t ndim, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t i
    for i in range(ndim):
        dst[i*stride_x] /= div[i*stride_y]
cdef void lia_vv_div(double * out, double * x, double * y, size_t ndim, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t i
    for i in range(ndim):
        out[i*stride_out] = x[i*stride_x] / y[i*stride_y]
cdef void lia_vv_div_A_i0_inplace(double * dst, double * div, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            dst[(J+i)*stride_x] /= div[i*stride_y]
cdef void lia_vv_div_A_i0(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            out[(J+i)*stride_out] = x[(J+i)*stride_x] / y[i*stride_y]
cdef void lia_vv_div_A_ii_inplace(double * dst, double * div, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            dst[(J+i)*stride_x] /= div[(J+i)*stride_y]
cdef void lia_vv_div_A_ii(double * out, double * x, double * y, size_t ndim, size_t nsize, int stride_x=1, int stride_y=1, int stride_out=1) noexcept nogil:
    cdef size_t j, J, i
    for j in range(nsize):
        J = j*ndim
        for i in range(ndim):
            out[(J+i)*stride_out] = x[(J+i)*stride_x] / y[(J+i)*stride_y]

##### Matrix - Vector Binary #####

cdef void lia_mv_mul(double * out, double * mat, double * vec, size_t m, size_t n, bint T=False, int lda=1, int stride_vec=1, int stride_out=1) noexcept nogil:
    if lda<n:
        lda = n
    cdef CBLAS_TRANSPOSE transposed = CBLAS_TRANSPOSE.CblasNoTrans
    if T:
        transposed = CBLAS_TRANSPOSE.CblasTrans
    cblas_dgemv(CBLAS_ORDER.CblasRowMajor, transposed, m, n,
                1.0, mat, lda, vec, stride_vec, 0.0, out, stride_out)

cdef void lia_mv_mul_A_0i(double * out, double * mat, double * vec, size_t m, size_t ndim, size_t nsize, bint T=False, int lda=1, int stride_vec=1, int stride_out=1) noexcept nogil:
    if lda<ndim:
        lda = ndim
    cdef size_t i
    cdef CBLAS_TRANSPOSE transposed = CBLAS_TRANSPOSE.CblasNoTrans
    if T:
        transposed = CBLAS_TRANSPOSE.CblasTrans
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        cblas_dgemv(CBLAS_ORDER.CblasRowMajor, transposed, m, ndim,
                    1.0, mat, lda, vec + i*ndim*stride_vec, stride_vec, 0.0, out + i*ndim*stride_out, stride_out)

cdef void lia_mv_mul_A_ii(double * out, double * mat, double * vec, size_t m, size_t ndim, size_t nsize, bint T=False, int lda=1, int stride_vec=1, int stride_out=1) noexcept nogil:
    if lda<ndim:
        lda = ndim
    cdef size_t i
    cdef CBLAS_TRANSPOSE transposed = CBLAS_TRANSPOSE.CblasNoTrans
    if T:
        transposed = CBLAS_TRANSPOSE.CblasTrans
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        cblas_dgemv(CBLAS_ORDER.CblasRowMajor, transposed, m, ndim,
                    1.0, mat + i*m*lda, lda, vec + i*ndim*stride_vec, stride_vec, 
                    0.0, out + i*ndim*stride_out, stride_out)

# Special case of symmetric matrix
cdef void lia_mv_mul_sym(double * out, double * mat, double * vec, size_t n, bint upper=True, int stride_vec=1, int stride_out=1) noexcept nogil:
    cdef CBLAS_UPLO uplo = CBLAS_UPLO.CblasUpper
    if not upper:
        uplo = CBLAS_UPLO.CblasLower
    cblas_dsymv(CBLAS_ORDER.CblasRowMajor, uplo, n, 
                1.0, mat, n, vec, stride_vec, 0.0, out, stride_out)

##### LAPACKE routines.

cdef void lia_mv_solve_bf_inplace(
    double * mat, double * vec, size_t n, int * ipiv,
    size_t nvec=1, int lda=1, bint upper=True, 
) noexcept nogil:
    # Find x in Ax = y.
    # mat: A
    # vec: 
    #     in : y (n*nvec)
    #    out : x (n*nvec)
    # ipiv: int type perm index, no need to concern. <lapack_int *> ipiv
    if lda<n:
        lda = n
    LAPACKE_dgesv(
        LAPACK_LAYOUT.LapackRowMajor, 
        n, nvec, 
        mat, lda, <lapack_int *> ipiv,
        vec, nvec,
    )

cdef void lia_mv_solve_inplace(
    double * mat, double * vec, size_t n, 
    size_t nvec=1, int lda=1, bint upper=True, 
) noexcept nogil:
    cdef int * ipiv = <int *> malloc(n*sizeof(int))
    lia_mv_solve_bf_inplace(mat, vec, n, ipiv, nvec=nvec, lda=lda, upper=upper)
    free(ipiv)

cdef void lia_mv_solve_bf(
    double * out, double * mat, double * vec, size_t n, int * ipiv,
    size_t nvec=1, int lda=1, bint upper=True, 
) noexcept nogil:
    cblas_dcopy(n*nvec, vec, 1, out, 1)
    lia_mv_solve_bf_inplace(mat, out, n, ipiv, nvec=nvec, lda=lda, upper=upper)

cdef void lia_mv_solve(
    double * out, double * mat, double * vec, size_t n, 
    size_t nvec=1, int lda=1, bint upper=True, 
) noexcept nogil:
    cblas_dcopy(n*nvec, vec, 1, out, 1)
    lia_mv_solve_inplace(mat, out, n, nvec=nvec, lda=lda, upper=upper)

##### Dimension-specific Binary #####

cdef void lia2d_mv_rot_inplace(double * vec, double c, double s) noexcept nogil:
    cblas_drot(1, vec, 2, vec+1, 2, c, s)

cdef void lia2d_mv_rot_A_0i_inplace(double * vec, double c, double s, size_t n) noexcept nogil:
    cblas_drot(n, vec, 2, vec+1, 2, c, s)

cdef void lia2d_mv_rot(double *out, double * vec, double c, double s) noexcept nogil:
    lia_vv_cpy(out, vec, 2)
    lia2d_mv_rot_inplace(out, c, s)

cdef void lia2d_mv_rot_A_0i(double * out, double * vec, double c, double s, size_t n) noexcept nogil:
    lia_vv_cpy_A_ii(out, vec, 2, n)
    lia2d_mv_rot_A_0i_inplace(out, c, s, n)

cdef void lia3d_mv_rot_inplace(double * vec, double c, double s, int axis) noexcept nogil:
    cdef int axis1, axis2
    if axis==0:
        axis1 = 2
        axis2 = 1
    elif axis==1:
        axis1 = 0
        axis2 = 2
    else: # axis==2
        axis1 = 1
        axis2 = 0
    cblas_drot(1, vec+axis1, 3, vec+axis2, 3, c, s)

cdef void lia3d_mv_rot_A_0i_inplace(double * vec, double c, double s, int axis, size_t n) noexcept nogil:
    cdef int axis1, axis2
    if axis==0:
        axis1 = 2
        axis2 = 1
    elif axis==1:
        axis1 = 0
        axis2 = 2
    else: # axis==2
        axis1 = 1
        axis2 = 0
    cblas_drot(n, vec+axis1, 3, vec+axis2, 3, c, s)

cdef void lia3d_mv_rot(double *out, double * vec, double c, double s, int axis) noexcept nogil:
    lia_vv_cpy(out, vec, 3)
    lia3d_mv_rot_inplace(out, c, s, axis)

cdef void lia3d_mv_rot_A_0i(double * out, double * vec, double c, double s, int axis, size_t n) noexcept nogil:
    lia_vv_cpy_A_ii(out, vec, 3, n)
    lia3d_mv_rot_A_0i_inplace(out, c, s, axis, n)

################################################################################
# Ternary Operations 
################################################################################

##### Vector - Matrix - Vector Ternary #####
# Requires a buffer for efficient computation!
cdef double lia_vmv_quadform_bf(# Inner product
    double * x, double * A, double * y, 
    size_t m, size_t n, 
    double * buffer, # size of m, should not be strided.
    bint T=False, int lda=1, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    if lda<n:
        lda = n
    cdef CBLAS_TRANSPOSE transposed = CBLAS_TRANSPOSE.CblasNoTrans
    if T:
        transposed = CBLAS_TRANSPOSE.CblasTrans
    # (A.y)[m] = A[mxn].y[n]
    cblas_dgemv(CBLAS_ORDER.CblasRowMajor, transposed, m, n,
                1.0, A, lda, y, stride_y, 0.0, buffer, 1)
    # x^T.A.y = x[m]^T.(A.y)[m] 
    return cblas_ddot(m, x, stride_x, buffer, 1)

cdef double lia_vmv_quadform(# Inner product
    double * x, double * A, double * y, 
    size_t m, size_t n, 
    bint T=False, int lda=1, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef double * buffer = <double *> malloc(m*sizeof(double))
    cdef double out = lia_vmv_quadform_bf(x, A, y, m, n, buffer, 
        T=T, lda=lda, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out)
    free(buffer)
    return out

cdef void lia_vmv_quadform_A_i0i_bf(double * out, # Inner product
    double * x, double * A, double * y, 
    size_t m, size_t n, size_t ndata, 
    double * buffer, # Shared! size of m, should not be strided.
    bint T=False, int lda=1, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef size_t i
    for i in range(ndata): # Must be serial because of shared buffer!
        out[i] = lia_vmv_quadform_bf(x+i*m*stride_x, A, y+i*n*stride_y, m, n, buffer, 
            T=T, lda=lda, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out)

cdef void lia_vmv_quadform_A_i0i(double * out, # Inner product
    double * x, double * A, double * y, 
    size_t m, size_t n, size_t ndata, 
    bint T=False, int lda=1, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef double * buffer = <double *> malloc(m*sizeof(double)) # Shared buffer
    lia_vmv_quadform_A_i0i_bf(
        out,
        x, A, y, 
        m, n, ndata, buffer, 
        T=T, lda=lda, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out
    )
    free(buffer)

cdef void lia_vmv_quadform_A_iii_bf(double * out, # Inner product
    double * x, double * A, double * y, 
    size_t m, size_t n, size_t ndata, 
    double * buffer, # Shared! size of m, should not be strided.
    bint T=False, int lda=1, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef size_t i
    for i in range(ndata): # Must be serial because of shared buffer!
        out[i] = lia_vmv_quadform_bf(x+i*m*stride_x, A+i*m*lda, y+i*n*stride_y, m, n, buffer, 
            T=T, lda=lda, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out)

cdef void lia_vmv_quadform_A_iii(double * out, # Inner product
    double * x, double * A, double * y, 
    size_t m, size_t n, size_t ndata, 
    bint T=False, int lda=1, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef double * buffer = <double *> malloc(m*sizeof(double)) # Shared buffer
    lia_vmv_quadform_A_iii_bf(
        out,
        x, A, y, 
        m, n, ndata, buffer, 
        T=T, lda=lda, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out
    )
    free(buffer)

# Special case of symmetric matrix (not packed)
cdef double lia_vmv_quadform_sym_bf(# Inner product
    double * x, double * A, double * y, 
    size_t n, 
    double * buffer, # size of n, should not be strided.
    bint upper=True, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef CBLAS_UPLO uplo = CBLAS_UPLO.CblasUpper
    if not upper:
        uplo = CBLAS_UPLO.CblasLower
    # (A.y)[n] = A[nxn].y[n]
    cblas_dsymv(CBLAS_ORDER.CblasRowMajor, uplo, n, 
                1.0, A, n, y, stride_y, 0.0, buffer, 1)
    # x^T.A.y = x[n]^T.(A.y)[n] 
    return cblas_ddot(n, x, stride_x, buffer, 1)

cdef double lia_vmv_quadform_sym(# Inner product
    double * x, double * A, double * y, 
    size_t n, 
    bint upper=True, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef double * buffer = <double *> malloc(n*sizeof(double))
    cdef double out = lia_vmv_quadform_sym_bf(x, A, y, n, buffer, 
        upper=upper, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out)
    free(buffer)
    return out

cdef void lia_vmv_quadform_sym_A_i0i_bf(double * out, # Inner product
    double * x, double * A, double * y, 
    size_t n, size_t ndata, 
    double * buffer, # Shared! size of n, should not be strided.
    bint upper=True, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef size_t i
    for i in range(ndata): # Must be serial because of shared buffer!
        out[i] = lia_vmv_quadform_sym_bf(x+i*n*stride_x, A, y+i*n*stride_y, n, buffer,
            upper=upper, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out)

cdef void lia_vmv_quadform_sym_A_i0i(double * out, # Inner product
    double * x, double * A, double * y, 
    size_t n, size_t ndata, 
    bint upper=True, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef double * buffer = <double *> malloc(n*sizeof(double)) # Shared buffer
    lia_vmv_quadform_sym_A_i0i_bf(
        out,
        x, A, y, 
        n, ndata, buffer, 
        upper=upper, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out
    )
    free(buffer)

cdef void lia_vmv_quadform_sym_A_iii_bf(double * out, # Inner product
    double * x, double * A, double * y, 
    size_t n, size_t ndata, 
    double * buffer, # Shared! size of n, should not be strided.
    bint upper=True, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef size_t i
    for i in range(ndata): # Must be serial because of shared buffer!
        
        out[i] = lia_vmv_quadform_sym_bf(x+i*n*stride_x, A+i*(n*(n+1)//2), y+i*n*stride_y, n, buffer, 
            upper=upper, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out)

cdef void lia_vmv_quadform_sym_A_iii(double * out, # Inner product
    double * x, double * A, double * y, 
    size_t n, size_t ndata, 
    bint upper=True, int stride_x=1, int stride_y=1, int stride_out=1
) noexcept nogil:
    cdef double * buffer = <double *> malloc(n*sizeof(double)) # Shared buffer
    lia_vmv_quadform_sym_A_iii_bf(
        out,
        x, A, y, 
        n, ndata, buffer, 
        upper, stride_x=stride_x, stride_y=stride_y, stride_out=stride_out
    )
    free(buffer)

################################################################################
# Quarternary Operations 
################################################################################

##### 2 Scalars - 2 Vector Quarternary #####

cdef void lia_svsv_lincomb(
    double * out, double alpha, double * x, double beta, double * y, size_t n, 
    int stride_x=1, int stride_y=1, int stride_out=1,
) noexcept nogil:
    cblas_dcopy(n, y, stride_y, out, stride_out)        # out = (y)
    lia_sv_scalar_inplace(beta, out, n, stride_out)         # out = (beta * y)
    cblas_daxpy(n, alpha, x, stride_x, out, stride_out) # out = alpha * x + (beta * y)

cdef void lia_svsv_lincomb_A_0i0i(
    double * out, 
    double alpha, double * x, double beta, double * y, 
    size_t ndim, size_t nsize,
    int stride_x=1, int stride_y=1, int stride_a=1, int stride_b=1, int stride_out=1,
) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        lia_svsv_lincomb(out + i*ndim*stride_out, 
            alpha, x + i*ndim*stride_x, 
            beta , y + i*ndim*stride_y, ndim, stride_x, stride_y, stride_out)

cdef void lia_svsv_lincomb_A_iiii(
    double * out, 
    double * alpha, double * x, double * beta, double * y, 
    size_t ndim, size_t nsize,
    int stride_x=1, int stride_y=1, int stride_a=1, int stride_b=1, int stride_out=1,
) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
    # for i in range(nsize):
        lia_svsv_lincomb(out + i*ndim*stride_out, 
            alpha[i*stride_a], x + i*ndim*stride_x, 
            beta[ i*stride_b], y + i*ndim*stride_y, ndim, stride_x, stride_y, stride_out)

################################################################################
# Low-level (0) Direct BLAS interface
################################################################################
# This may be different if it uses non-standard BLAS implementations. (e.g., cuBLAS)

cdef CBLAS_ORDER __lia_order(LIA_ORDER x) noexcept nogil:
    if x == LIA_ORDER.RowMajor:
        return CBLAS_ORDER.CblasRowMajor
    return CBLAS_ORDER.CblasColMajor

cdef CBLAS_TRANSPOSE __lia_trans(LIA_TRANS x) noexcept nogil:
    if x == LIA_TRANS.NoTrans:
        return CBLAS_TRANSPOSE.CblasNoTrans
    elif x == LIA_TRANS.Trans:
        return CBLAS_TRANSPOSE.CblasTrans
    elif x == LIA_TRANS.ConjTrans:
        return CBLAS_TRANSPOSE.CblasConjTrans
    return CBLAS_TRANSPOSE.CblasConjNoTrans

cdef CBLAS_UPLO __lia_uplo(LIA_UPLO x) noexcept nogil:
    if x == LIA_UPLO.Up:
        return CBLAS_UPLO.CblasUpper
    return CBLAS_UPLO.CblasLower

cdef CBLAS_DIAG __lia_diag(LIA_DIAG x) noexcept nogil:
    if x == LIA_DIAG.NonUnit:
        return CBLAS_DIAG.CblasNonUnit
    return CBLAS_DIAG.CblasUnit

cdef CBLAS_SIDE __lia_side(LIA_SIDE x) noexcept nogil:
    if x == LIA_SIDE.Left:
        return CBLAS_SIDE.CblasLeft
    return CBLAS_SIDE.CblasRight

cdef void lia_0_dsymm(LIA_ORDER Order, LIA_SIDE Side, LIA_UPLO Uplo, int M, int N,
    double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) noexcept nogil:
    cblas_dsymm(__lia_order(Order), __lia_side(Side), __lia_uplo(Uplo), M, N, 
                alpha, A, lda, B, ldb, beta, C, ldc)

cdef void lia_0_dgemm(LIA_ORDER Order, LIA_TRANS TransA, LIA_TRANS TransB, int M, int N, int K,
    double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) noexcept nogil:
    cblas_dgemm(__lia_order(Order), __lia_trans(TransA), __lia_trans(TransB), M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)

cdef void lia_0_dcopy(int n, double * x, int incx, double * y, int incy) noexcept nogil:
    cblas_dcopy(n, x, incx, y, incy)

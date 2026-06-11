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

#### Index helper for packed storage of symmetric matrices
cdef size_t lia_idx_tri_row_u(size_t i, size_t j, size_t n) noexcept nogil:
    return j+i*(2*n-i-1)//2
cdef size_t lia_idx_tri_col_u(size_t i, size_t j, size_t n) noexcept nogil:
    return i+j*(j+1)//2
cdef size_t lia_idx_tri_row_l(size_t i, size_t j, size_t n) noexcept nogil:
    return j+i*(i+1)//2
cdef size_t lia_idx_tri_col_l(size_t i, size_t j, size_t n) noexcept nogil:
    return i+j*(2*n-j-1)//2

#### "Safe" version of the same index functions; in case if it is not clear what to use.
cdef size_t lia_safeidx_tri_row_u(size_t i, size_t j, size_t n) noexcept nogil:
    if j<i: return lia_idx_tri_row_u(j, i, n)
    else:   return lia_idx_tri_row_u(i, j, n)
cdef size_t lia_safeidx_tri_col_u(size_t i, size_t j, size_t n) noexcept nogil:
    if j<i: return lia_idx_tri_col_u(j, i, n)
    else:   return lia_idx_tri_col_u(i, j, n)
cdef size_t lia_safeidx_tri_row_l(size_t i, size_t j, size_t n) noexcept nogil:
    if i<j: return lia_idx_tri_row_l(j, i, n)
    else:   return lia_idx_tri_row_l(i, j, n)
cdef size_t lia_safeidx_tri_col_l(size_t i, size_t j, size_t n) noexcept nogil:
    if i<j: return lia_idx_tri_col_l(j, i, n)
    else:   return lia_idx_tri_col_l(i, j, n)

#### Index helper for the inner loop [i]nitial and [f]inal for packed storage.
cdef size_t lia_idx_tri_u_i(size_t i, size_t n) noexcept nogil:
    return i
cdef size_t lia_idx_tri_u_f(size_t i, size_t n) noexcept nogil:
    return n
cdef size_t lia_idx_tri_l_i(size_t i, size_t n) noexcept nogil:
    return 0
cdef size_t lia_idx_tri_l_f(size_t i, size_t n) noexcept nogil:
    return i+1
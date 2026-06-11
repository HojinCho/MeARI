# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

from .types cimport func_t_double_par

cdef double integer_pow(double x0, size_t n) noexcept nogil

cdef void double_copy_arr_1d(long n, double [:] arr_in, double [:] arr_out)
cdef void double_copy_arr_2d(long n1, long n2, double [:,:] arr_in, double [:,:] arr_out)

cdef void kahan_sum_iterator(double *s, double to_add, double *c) noexcept nogil

cdef void find_min_max(double * arr, long n, double *m, double *M) noexcept nogil
cdef double max_ternary(double x, double y, double z) noexcept nogil

cdef void quicksort(double [:] arr, long [:] idx, long low, long high)
cdef void quicksort_ptr(double * arr, long * idx, long low, long high) noexcept nogil
cdef void mergesort(double [:] arr, long [:] idx, long low, long high)
cdef void mergesort_ptr(double * arr, long * idx, long low, long high) noexcept nogil
cdef void mergesort_ptr_buf(double * arr, long * idx, long low, long high, long * temp) noexcept nogil
# cdef void mergesort_ptr_buf_prealloc(double * arr, long * idx, long low, long high) noexcept nogil

cdef void countingsort(long [:] arr, long [:] sorted_idx, long n, long keymax)
cdef void countingsort_uchar(unsigned char [:] arr, long [:] sorted_idx, long n, long keymax)
cdef void countingsort_ptr(long * arr, long * sorted_idx, long n, long keymax) noexcept nogil
cdef void countingsort_uchar_ptr(unsigned char * arr, long * sorted_idx, long n, long keymax) noexcept nogil
cdef void countingsort_uchar_ptr_subdivide(unsigned char * arr, long * sorted_idx, long low, long high, long keymax) noexcept nogil
cdef void count_each_items(long [:] arr, long [:] count, long n, long keymax)
cdef void count_each_items_uchar(unsigned char [:] arr, long [:] count, long n, long keymax)
cdef void count_each_items_ptr(long * arr, long * count, long n, long keymax) noexcept nogil
cdef void count_each_items_uchar_ptr(unsigned char * arr, long * count, long n, long keymax) noexcept nogil

cdef void flip_index(long n, long * idx) noexcept nogil
cdef void flip_index_b(long n, long * idx, long * buffer) noexcept nogil
cdef void assign_sorted_vector_dbl(long nsize, long ndim, long [:] idx, double [:,:] vec_in, double [:,:] vec_out) noexcept nogil
cdef void assign_sorted_scalar_dbl(long nsize, long [:] idx, double [:] arr_in, double [:] arr_out) noexcept nogil
cdef void assign_sorted_vector_lng(long nsize, long ndim, long [:] idx, long [:,:] vec_in, long [:,:] vec_out) noexcept nogil
cdef void assign_sorted_scalar_lng(long nsize, long [:] idx, long [:] arr_in, long [:] arr_out) noexcept nogil
cdef void assign_sorted_vector_inplace_dbl(long nsize, long ndim, long [:] idx, double [:,:] vec) noexcept nogil
cdef void assign_sorted_scalar_inplace_dbl(long nsize, long [:] idx, double [:] arr) noexcept nogil
cdef void assign_sorted_vector_inplace_lng(long nsize, long ndim, long [:] idx, long [:,:] vec) noexcept nogil
cdef void assign_sorted_scalar_inplace_lng(long nsize, long [:] idx, long [:] arr) noexcept nogil
cdef void assign_sorted_scalar_inplace_uchar(long nsize, long [:] idx, unsigned char [:] arr) noexcept nogil

cdef void assign_sorted_vector_inplace_lng_ptr_bf(size_t nsize, size_t ndim, long * idx, long * vec, long * buffer) noexcept nogil
cdef void assign_sorted_scalar_inplace_lng_ptr_bf(size_t nsize, long * idx, long * arr, long * buffer) noexcept nogil
cdef void assign_sorted_vector_inplace_dbl_ptr_bf(size_t nsize, size_t ndim, long * idx, double * vec, double * buffer) noexcept nogil
cdef void assign_sorted_scalar_inplace_dbl_ptr_bf(size_t nsize, long * idx, double * arr, double * buffer) noexcept nogil
cdef void assign_sorted_scalar_inplace_uchar_ptr_bf(size_t nsize, long * idx, unsigned char * arr, unsigned char * buffer) noexcept nogil
cdef void assign_sorted_vector_inplace_lng_ptr(size_t nsize, size_t ndim, long * idx, long * vec) noexcept nogil
cdef void assign_sorted_scalar_inplace_lng_ptr(size_t nsize, long * idx, long * arr) noexcept nogil
cdef void assign_sorted_vector_inplace_dbl_ptr(size_t nsize, size_t ndim, long * idx, double * vec) noexcept nogil
cdef void assign_sorted_scalar_inplace_dbl_ptr(size_t nsize, long * idx, double * arr) noexcept nogil
cdef void assign_sorted_scalar_inplace_uchar_ptr(size_t nsize, long * idx, unsigned char * arr) noexcept nogil
cdef void assign_sorted_scalar_dbl_ptr(size_t nsize_out, long * idx, double * inarr, double * outarr) noexcept nogil
cdef void assign_sorted_vector_dbl_ptr(size_t nsize_out, size_t ndim, long * idx, double * inarr, double * outarr) noexcept nogil


cdef long interp_search_ptr_L_unsafe_unsafe(double x, double * arr, long n) noexcept nogil
cdef long interp_search_ptr_R_unsafe_unsafe(double x, double * arr, long n) noexcept nogil
cdef long interp_search_ptr_L_unsafe(double x, double * arr, long n) noexcept nogil
cdef long interp_search_ptr_R_unsafe(double x, double * arr, long n) noexcept nogil
cdef long interp_search_ptr_L(double x, double * arr, long n) noexcept nogil
cdef long interp_search_ptr_R(double x, double * arr, long n) noexcept nogil

cdef long binary_search_ptr_L(double x, double * arr, long n) noexcept nogil
cdef long binary_search_ptr_R(double x, double * arr, long n) noexcept nogil
cdef long binary_search_ptr_L_desc(double x, double * arr, long n) noexcept nogil
cdef long binary_search_ptr_R_desc(double x, double * arr, long n) noexcept nogil
cdef long binary_search(double x, double [:] arr, long n) noexcept nogil
cdef long binary_search_lng(long x, long [:] arr, long n) noexcept nogil

cdef double find_root(func_t_double_par f, void *params, double a, double b) noexcept nogil

cdef void compute_cumulsum_forward(long n, double *y, double * out) noexcept nogil
cdef void compute_cumulsum_backward(long n, double *y, double * out) noexcept nogil
cdef double integ_u_midpoint(size_t n, double * y, double h) noexcept nogil
cdef double integ_u_trapezoid(size_t n, double * y, double h) noexcept nogil
cdef double integ_u_simpson(size_t n, double * y, double h) noexcept nogil
cdef double riemann_sum_r(size_t n, double * y, double * dx) noexcept nogil
cdef void finite_difference_f(long n, double * y, double * dy) noexcept nogil
cdef void finite_difference_b(long n, double * y, double * dy) noexcept nogil
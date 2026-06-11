# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

import numpy as __numpy__
cimport numpy as __numpy__
cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdio  cimport printf

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, cos, sin, pow,
    fmin, fmax, fabs, floor, ceil, log2,
)


from .types cimport func_t_double_par, Real, Integer

### Faster integer exponentiation
# cdef double integer_pow(double x, size_t n) noexcept nogil:
#     # https://baptiste-wicht.com/posts/2017/09/cpp11-performance-tip-when-to-use-std-pow.html
#     # use when n<100, otherwise use pow.
#     cdef double y = 1.0
#     cdef size_t i
#     for i in range(n):
#         y *= x
#     return x

cdef double integer_pow(double x0, size_t n) noexcept nogil:
    # https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    cdef double result = 1.0
    cdef double x = x0
    cdef size_t m = n
    while m > 0:
        if m%2 == 1:
            result *= x
        x *= x
        m = m//2
    return result

# def int_pow(x, n): 
# 55.4 ns (mostly interface; e.g., pure python 2.**3 takes only 3.27ns. Similarly, loop inside cython would be much faster)
#     cdef double y = <double>(float(x))
#     cdef size_t m = <size_t>(abs(int(n)))
#     cdef double z
#     with nogil:
#         z = integer_pow(y, m)
#     return z

# def c_pow(x, n):
# 63.9 ns (mostly interface; e.g., pure python 2.**3 takes only 3.27ns. Similarly, loop inside cython would be much faster)
#     cdef double y = <double>(float(x))
#     cdef size_t m = <size_t>(abs(int(n)))
#     cdef double z
#     with nogil:
#         z = pow(y, m)
#     return z

# cdef double py_pow_backend(double x, double n) noexcept nogil:
#     return x**n

# def py_pow(x, n):
# 63.0 ns (mostly interface; pure python 2.**3 takes only 3.27ns: Even slower than pure python!)
#     cdef double y = <double>(float(x))
#     cdef size_t m = <size_t>(abs(int(n)))
#     cdef double z
#     with nogil:
#         z = py_pow_backend(y,m)
#     return z

### Copying array

# cdef void double_copy_ptr(long n, double * arr_in, double * arr_out):
#     memcpy(<void *>arr_out, <void *>arr_in, n * sizeof(double))

cdef void double_copy_arr_1d(long n, double [:] arr_in, double [:] arr_out):
    cdef long i
    for i in prange(n, nogil=True):
        arr_out[i] = arr_in[i]

cdef void double_copy_arr_2d(long n1, long n2, double [:,:] arr_in, double [:,:] arr_out):
    cdef long i, j
    for i in prange(n1, nogil=True):
        for j in range(n2):
            arr_out[i,j] = arr_in[i,j]

### Summation
cdef void kahan_sum_iterator(double *s, double to_add, double *c) noexcept nogil: 
    # Technically can be run without GIL, but never with parallel.
    # In the beginning of iteration, make sure to initialize c and s to 0.
    cdef volatile double y, t
    y = to_add - c[0]
    t = s[0] + y
    c[0] = (t - s[0]) - y
    s[0] = t

cdef double Kahan_Sum_Single(double * arr, long n, double * c) noexcept nogil:
    cdef double s = 0.0
    cdef long i
    c[0] = 0.0
    for i in range(n):
        kahan_sum_iterator(&s, arr[i], c)
    return s

cdef void __divide_load(long n, long m, long * res) noexcept nogil:
    cdef long i, r, q, q1
    r = n % m
    q = n // m
    q1 = q + 1
    for i in prange(r, nogil=True):
        res[i] = q1
    for i in prange(r, m, nogil=True):
        res[i] = q

# # Replace with Kahan_Sum function
# cdef double Kahan_Sum(double * arr, long n, long nthread) noexcept nogil:
#     cdef double * c = <double *> malloc(nthread * sizeof(double))


### MinMax
cdef void find_min_max(double * arr, long n, double *m, double *M) noexcept nogil:
    cdef long i
    cdef double localMin, localMax
    if n <= 0:
        return
    if n == 1:
        m[0] = M[0] = arr[0]
        return
    m[0] = M[0] = arr[0]
    for i in range(1, n-1, 2):
        if arr[i]<arr[i+1]:
            localMin = arr[i]
            localMax = arr[i+1]
        else:
            localMin = arr[i+1]
            localMax = arr[i]
        if (localMin < m[0]):
            m[0] = localMin
        if (localMax > M[0]):
            M[0] = localMax
    # Handle odd number of elements
    if n % 2 != 0:
        if (arr[n-1] < m[0]):
            m[0] = arr[n-1]
        if (arr[n-1] > M[0]):
            M[0] = arr[n-1]

cdef double max_ternary(double x, double y, double z) noexcept nogil:
    return fmax(fmax(x, y), z)

### Interpolation Search
# Never use these directly; they're just to provide initial guess.
cdef long interp_search_ptr_L_unsafe_unsafe(double x, double * arr, long n) noexcept nogil:
    return <long> (fmax(fmin(ceil((x-arr[0])/(arr[1]-arr[0])), n), 0.))
cdef long interp_search_ptr_R_unsafe_unsafe(double x, double * arr, long n) noexcept nogil:
    return <long> (fmax(fmin(floor((x-arr[0])/(arr[1]-arr[0])), n), 0.))
# Only use when it is absolutely sure that arr is equal-spaced within machine epsilon.
cdef long interp_search_ptr_L_unsafe(double x, double * arr, long n) noexcept nogil:
    cdef long idx = interp_search_ptr_L_unsafe_unsafe(x, arr, n)
    if idx==0 or idx==n:
        return idx
    if arr[idx]<x:
        return idx+1
    if x<=arr[idx-1]:
        return idx-1
    return idx
cdef long interp_search_ptr_R_unsafe(double x, double * arr, long n) noexcept nogil:
    cdef long idx = interp_search_ptr_R_unsafe_unsafe(x, arr, n)
    if idx==0 or idx==n:
        return idx
    if arr[idx]<=x:
        return idx+1
    if x<arr[idx-1]:
        return idx-1
    return idx
# General Purpose.
cdef long interp_search_ptr_L(double x, double * arr, long n) noexcept nogil:
    cdef long idx = interp_search_ptr_L_unsafe_unsafe(x, arr, n)
    if idx==0 or idx==n:
        return idx
    # arr[idx-1] < x <= arr[idx]
    while arr[idx]<x:
        idx +=1
        if idx==n:
            return idx
    while x<arr[idx-1]:
        idx -=1
        if idx==0:
            return idx
    return idx
cdef long interp_search_ptr_R(double x, double * arr, long n) noexcept nogil:
    cdef long idx = interp_search_ptr_R_unsafe_unsafe(x, arr, n)
    if idx==0 or idx==n:
        return idx
    # arr[idx-1] <= x < arr[idx]
    while arr[idx]<x:
        idx +=1
        if idx==n:
            return idx
    while x<arr[idx-1]:
        idx -=1
        if idx==0:
            return idx
    return idx

### Binary Search
cdef long binary_search_ptr_L(double x, double * arr, long n) noexcept nogil:
    cdef long low = 0
    cdef long high = n
    # According to ChatGPT, this is a deliberate choice to make open interval...Well, what do I know? I'm just a comment.
    cdef long mid
    # if x < arr[0]:
    #     return -1
    while low < high:
        mid = low + (high-low) // 2 # This is to prevent integer overflow.
        if arr[mid] < x:
            low = mid + 1
        else:
            high = mid
    return low # arr[low-1] < x <= arr[low]

cdef long binary_search_ptr_R(double x, double * arr, long n) noexcept nogil:
    cdef long low = 0
    cdef long high = n
    cdef long mid
    # if x > arr[n-1]:
    #     return n+1 # [:ret] should be the whole array, but :n+1 should raise error, similar to how x<arr[0] returns -1 in L.
    while low < high:
        mid = low + (high-low) // 2 # This is to prevent integer overflow.
        if arr[mid] <= x:
            low = mid + 1
        else:
            high = mid
    return low # arr[low-1] <= x < arr[low]

cdef long binary_search_ptr_L_desc(double x, double * arr, long n) noexcept nogil:
    cdef long low = 0
    cdef long high = n
    cdef long mid
    while low < high:
        mid = low + (high-low) // 2
        if arr[mid] >= x:
            low = mid + 1
        else:
            high = mid
    return low # arr[low-1] >= x > arr[low]

cdef long binary_search_ptr_R_desc(double x, double * arr, long n) noexcept nogil:
    cdef long low = 0
    cdef long high = n
    cdef long mid
    while low < high:
        mid = low + (high-low) // 2
        if arr[mid] > x:
            low = mid + 1
        else:
            high = mid
    return low # arr[low-1] > x >= arr[low]

cdef long binary_search(double x, double [:] arr, long n) noexcept nogil:
    cdef long low = 0
    cdef long high = n - 1
    cdef long mid

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] <= x < arr[mid + 1]:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1  # If no such index is found

cdef long binary_search_lng(long x, long [:] arr, long n) noexcept nogil:
    cdef long low = 0
    cdef long high = n - 1
    cdef long mid

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] <= x < arr[mid + 1]:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1  # If no such index is found

### Sorting
cdef void quicksort(double [:] arr, long [:] idx, long low, long high):
    cdef long i = low
    cdef long j = high
    cdef double pivot = arr[idx[(low + high) // 2]]
    cdef long temp
    while i <= j:
        while arr[idx[i]] < pivot:
            i += 1
        while arr[idx[j]] > pivot:
            j -= 1
        if i <= j:
            temp = idx[i]
            idx[i] = idx[j]
            idx[j] = temp
            i += 1
            j -= 1
    if low < j:
        quicksort(arr, idx, low, j)
    if i < high:
        quicksort(arr, idx, i, high)

cdef void quicksort_ptr(double * arr, long * idx, long low, long high) noexcept nogil:
    cdef long i = low
    cdef long j = high
    cdef double pivot = arr[idx[(low + high) // 2]]
    cdef long temp
    while i <= j:
        while arr[idx[i]] < pivot:
            i += 1
        while arr[idx[j]] > pivot:
            j -= 1
        if i <= j:
            temp = idx[i]
            idx[i] = idx[j]
            idx[j] = temp
            i += 1
            j -= 1
    if low < j:
        quicksort_ptr(arr, idx, low, j)
    if i < high:
        quicksort_ptr(arr, idx, i, high)

cdef void __merge(double [:] arr, long [:] idx, long low, long mid, long high):
    cdef long i = low
    cdef long j = mid + 1
    cdef long k = 0
    cdef long *temp = <long *> malloc((high - low + 1) * sizeof(long))
    while i <= mid and j <= high:
        if arr[idx[i]] < arr[idx[j]]:
            temp[k] = idx[i]
            i += 1
        else:
            temp[k] = idx[j]
            j += 1
        k += 1
    while i <= mid:
        temp[k] = idx[i]
        i += 1
        k += 1
    while j <= high:
        temp[k] = idx[j]
        j += 1
        k += 1
    for i in range(k):
        idx[low + i] = temp[i]
    free(temp)

cdef void mergesort(double [:] arr, long [:] idx, long low, long high):
    cdef long mid
    if low < high:
        mid = (low + high) // 2
        mergesort(arr, idx, low, mid)
        mergesort(arr, idx, mid + 1, high)
        __merge(arr, idx, low, mid, high)

# cdef void __merge_ptr(double * arr, long * idx, long low, long mid, long high) noexcept nogil:
#     cdef long i = low
#     cdef long j = mid + 1
#     cdef long k = 0
#     cdef long *temp = <long *> malloc((high - low + 1) * sizeof(long))
#     while i <= mid and j <= high:
#         if arr[idx[i]] < arr[idx[j]]:
#             temp[k] = idx[i]
#             i += 1
#         else:
#             temp[k] = idx[j]
#             j += 1
#         k += 1
#     while i <= mid:
#         temp[k] = idx[i]
#         i += 1
#         k += 1
#     while j <= high:
#         temp[k] = idx[j]
#         j += 1
#         k += 1
#     for i in range(k):
#         idx[low + i] = temp[i]
#     free(temp)

# cdef void mergesort_ptr(double * arr, long * idx, long low, long high) noexcept nogil:
#     cdef long mid
#     if low < high:
#         mid = (low + high) // 2
#         mergesort_ptr(arr, idx, low, mid)
#         mergesort_ptr(arr, idx, mid + 1, high)
#         __merge_ptr(arr, idx, low, mid, high)


cdef void __merge_ptr_buf(double * arr, long * idx, long low, long mid, long high, long * temp) noexcept nogil:
    cdef long i = low
    cdef long j = mid + 1
    cdef long k = low
    
    while i <= mid and j <= high:
        if arr[idx[i]] < arr[idx[j]]:
            temp[k] = idx[i]
            i += 1
        else:
            temp[k] = idx[j]
            j += 1
        k += 1
    while i <= mid:
        temp[k] = idx[i]
        i += 1
        k += 1
    while j <= high:
        temp[k] = idx[j]
        j += 1
        k += 1
    for k in range(low, high+1):
        idx[k] = temp[k]

cdef void mergesort_ptr_buf(double * arr, long * idx, long low, long high, long * temp) noexcept nogil:
    # buffer should be at least (high-low+1) long
    cdef long mid
    if low < high:
        mid = (low + high) // 2
        mergesort_ptr_buf(arr, idx, low, mid, temp)
        mergesort_ptr_buf(arr, idx, mid + 1, high, temp)
        __merge_ptr_buf(arr, idx, low, mid, high, temp)

cdef void mergesort_ptr(double * arr, long * idx, long low, long high) noexcept nogil:
    cdef long *temp = <long *> malloc((high - low + 1) * sizeof(long))
    cdef long mid
    if low < high:
        mid = (low + high) // 2
        mergesort_ptr_buf(arr, idx, low, mid, temp)
        mergesort_ptr_buf(arr, idx, mid + 1, high, temp)
        __merge_ptr_buf(arr, idx, low, mid, high, temp)
    free(temp)

# cdef void mergesort_bottomup_ptr_buf(double * arr, long * idx, long low, long high, long * temp) noexcept nogil:
# # This is unusably slow. Don't use it.
#     cdef long width, left, mid, right, N
#     N = high - low + 1
#     width = 1
#     while width < N:
#         left = low
#         # while left <= high:
#         for left in prange(low, high, 2*width, nogil=True):
#             mid = left + width - 1
#             if mid >= high:
#                 break
#             right = left + 2*width - 1
#             if right > high:
#                 right = high
#             __merge_ptr_buf(arr, idx, low, mid, high, temp)
#             # left += 2*width
#         width *= 2


####
# Slightly faster?
cdef long __count_one_key_uchar(unsigned char * arr, long n, unsigned char key) noexcept nogil:
    cdef long i, cnt = 0
    for i in range(n):
        if arr[i] == key:
            cnt += 1
    return cnt

cdef long __count_one_key_uchar_subdivide(unsigned char * arr, long low, long high, unsigned char key) noexcept nogil:
    cdef long i, cnt = 0
    for i in range(low, high):
        if arr[i] == key:
            cnt += 1
    return cnt



cdef void countingsort(long [:] arr, long [:] sorted_idx, long n, long keymax):
    cdef long i
    cdef long * count = <long *> calloc(keymax + 1, sizeof(long))
    for i in range(n):
        count[arr[i]] += 1
    for i in range(1, keymax + 1):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        count[arr[i]] -= 1
        sorted_idx[count[arr[i]]] = i
    free(count)

cdef void countingsort_uchar(unsigned char [:] arr, long [:] sorted_idx, long n, long keymax):
    cdef long i
    cdef long * count = <long *> calloc(keymax + 1, sizeof(long))
    for i in range(n):
        count[arr[i]] += 1
    for i in range(1, keymax + 1):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        count[arr[i]] -= 1
        sorted_idx[count[arr[i]]] = i
    free(count)

cdef void countingsort_ptr(long * arr, long * sorted_idx, long n, long keymax) noexcept nogil:
    cdef long i
    cdef long * count = <long *> calloc(keymax + 1, sizeof(long))
    for i in range(n):
        count[arr[i]] += 1
    for i in range(1, keymax + 1):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        count[arr[i]] -= 1
        sorted_idx[count[arr[i]]] = i
    free(count)

cdef void countingsort_uchar_ptr(unsigned char * arr, long * sorted_idx, long n, long keymax) noexcept nogil:
    cdef long i
    cdef long * count = <long *> calloc(keymax + 1, sizeof(long))
    for i in range(n):
        count[arr[i]] += 1
    for i in range(1, keymax + 1):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        count[arr[i]] -= 1
        sorted_idx[count[arr[i]]] = i
    free(count)

# cdef void countingsort_uchar_ptr_subdivide(unsigned char * arr, long * sorted_idx, long low, long high, long keymax) noexcept nogil:
#     cdef long i
#     cdef long * count = <long *> calloc(keymax + 1, sizeof(long))
#     for i in range(low, high):
#         count[arr[i]] += 1
#     for i in range(1, keymax + 1):
#         count[i] += count[i - 1]
#     for i in range(high-1, low-1, -1):
#         count[arr[i]] -= 1
#         sorted_idx[low+count[arr[i]]] = i
#         # if i>=high or i<low:
#         #     printf("outarr bound prob: %d -> %d out of [%d, %d)\n", low+count[arr[i]], i, low, high)
#         # if low+count[arr[i]]>=high or low+count[arr[i]]<low:
#         #     printf("inarr  bound prob: %d -> %d out of [%d, %d)\n", low+count[arr[i]], i, low, high)
#     free(count)
# Slightly Faster?
cdef void countingsort_uchar_ptr_subdivide(unsigned char * arr, long * sorted_idx, long low, long high, long keymax) noexcept nogil:
    cdef long i
    cdef long * count = <long *> malloc((keymax + 1)*sizeof(long))
    cdef long * ptr_c
    cdef long * sorted_ptr = sorted_idx + low
    for i in prange(keymax + 1, nogil=True):
        count[i] = __count_one_key_uchar_subdivide(arr, low, high, i)
    for i in range(1, keymax + 1):
        count[i] += count[i - 1]
    for i in range(high-1, low-1, -1):
        ptr_c = count+arr[i]
        ptr_c[0] -= 1
        sorted_ptr[ptr_c[0]] = i
    free(count)


# cdef void count_each_items_uchar_ptr(unsigned char * arr, long * count, long n, long keymax) noexcept nogil:
#     cdef long i
#     for i in range(keymax + 1):
#         count[i] = 0
#     for i in range(n):
#         count[arr[i]] += 1

cdef void count_each_items(long [:] arr, long [:] count, long n, long keymax):
    cdef long i
    for i in range(keymax + 1):
        count[i] = 0
    for i in range(n):
        count[arr[i]] += 1
cdef void count_each_items_uchar(unsigned char [:] arr, long [:] count, long n, long keymax):
    cdef long i
    for i in range(keymax + 1):
        count[i] = 0
    for i in range(n):
        count[arr[i]] += 1
cdef void count_each_items_ptr(long * arr, long * count, long n, long keymax) noexcept nogil:
    cdef long i
    for i in range(keymax + 1):
        count[i] = 0
    for i in range(n):
        count[arr[i]] += 1
# cdef void count_each_items_uchar_ptr(unsigned char * arr, long * count, long n, long keymax) noexcept nogil:
#     cdef long i
#     for i in range(keymax + 1):
#         count[i] = 0
#     for i in range(n):
#         count[arr[i]] += 1
# Slightly faster?
cdef void count_each_items_uchar_ptr(unsigned char * arr, long * count, long n, long keymax) noexcept nogil:
    cdef long i
    for i in prange(keymax + 1, nogil=True):
        count[i] = __count_one_key_uchar(arr, n, i)

# Sort helper functions
### Flip index
cdef void flip_index(long n, long * idx) noexcept nogil:
    cdef long i, nmax
    cdef long * buffer = <long *> malloc(n * sizeof(long))
    nmax = n - 1
    for i in prange(n, nogil=True):
        buffer[nmax - i] = idx[i]
    memcpy(<void *>idx, <void *>buffer, n * sizeof(long))
    free(buffer)

cdef void flip_index_b(long n, long * idx, long * buffer) noexcept nogil:
    cdef long i, nmax
    nmax = n - 1
    for i in prange(n, nogil=True):
        buffer[nmax - i] = idx[i]
    memcpy(<void *>idx, <void *>buffer, n * sizeof(long))

### Assign Sorted arrays
cdef void assign_sorted_vector_dbl(long nsize, long ndim, long [:] idx, double [:,:] vec_in, double [:,:] vec_out) noexcept nogil:
    cdef long i, j
    for i in prange(nsize, nogil=True):
        for j in range(ndim):
            vec_out[i,j] = vec_in[idx[i],j]

cdef void assign_sorted_scalar_dbl(long nsize, long [:] idx, double [:] arr_in, double [:] arr_out) noexcept nogil:
    cdef long i
    for i in prange(nsize, nogil=True):
        arr_out[i] = arr_in[idx[i]]

cdef void assign_sorted_vector_lng(long nsize, long ndim, long [:] idx, long [:,:] vec_in, long [:,:] vec_out) noexcept nogil:
    cdef long i, j
    for i in prange(nsize, nogil=True):
        for j in range(ndim):
            vec_out[i,j] = vec_in[idx[i],j]

cdef void assign_sorted_scalar_lng(long nsize, long [:] idx, long [:] arr_in, long [:] arr_out) noexcept nogil:
    cdef long i
    for i in prange(nsize, nogil=True):
        arr_out[i] = arr_in[idx[i]]

cdef void assign_sorted_vector_inplace_dbl(long nsize, long ndim, long [:] idx, double [:,:] vec) noexcept nogil:
    cdef long i, j
    cdef double * buffer = <double *> malloc(ndim * nsize * sizeof(double))
    for i in prange(nsize, nogil=True):
        for j in range(ndim):
            buffer[ndim*i+j] = vec[idx[i],j]
    for i in prange(nsize, nogil=True):
        for j in range(ndim):
            vec[i,j] = buffer[ndim*i+j]
    free(buffer)

cdef void assign_sorted_scalar_inplace_dbl(long nsize, long [:] idx, double [:] arr) noexcept nogil:
    cdef long i
    cdef double * buffer = <double *> malloc(nsize * sizeof(double))
    for i in prange(nsize, nogil=True):
        buffer[i] = arr[idx[i]]
    for i in prange(nsize, nogil=True):
        arr[i] = buffer[i]
    free(buffer)

cdef void assign_sorted_vector_inplace_lng(long nsize, long ndim, long [:] idx, long [:,:] vec) noexcept nogil:
    cdef long i, j
    cdef long * buffer = <long *> malloc(ndim * nsize * sizeof(long))
    for i in prange(nsize, nogil=True):
        for j in range(ndim):
            buffer[ndim*i+j] = vec[idx[i],j]
    for i in prange(nsize, nogil=True):
        for j in range(ndim):
            vec[i,j] = buffer[ndim*i+j]
    free(buffer)

cdef void assign_sorted_scalar_inplace_lng(long nsize, long [:] idx, long [:] arr) noexcept nogil:
    cdef long i
    cdef long * buffer = <long *> malloc(nsize * sizeof(long))
    for i in prange(nsize, nogil=True):
        buffer[i] = arr[idx[i]]
    for i in prange(nsize, nogil=True):
        arr[i] = buffer[i]
    free(buffer)

cdef void assign_sorted_scalar_inplace_uchar(long nsize, long [:] idx, unsigned char [:] arr) noexcept nogil:
    cdef long i
    cdef unsigned char * buffer = <unsigned char *> malloc(nsize * sizeof(unsigned char))
    for i in prange(nsize, nogil=True):
        buffer[i] = arr[idx[i]]
    for i in prange(nsize, nogil=True):
        arr[i] = buffer[i]
    free(buffer)


cdef void assign_sorted_vector_inplace_lng_ptr_bf(size_t nsize, size_t ndim, long * idx, long * vec, long * buffer) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
        memcpy(<void *>(buffer+ndim*i), <void *>(vec+ndim*idx[i]), ndim * sizeof(long))
    memcpy(<void *>vec, <void *>buffer, ndim * nsize * sizeof(long))
cdef void assign_sorted_scalar_inplace_lng_ptr_bf(size_t nsize, long * idx, long * arr, long * buffer) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
        buffer[i] = arr[idx[i]]
    memcpy(<void *>arr, <void *>buffer, nsize * sizeof(long))
cdef void assign_sorted_vector_inplace_dbl_ptr_bf(size_t nsize, size_t ndim, long * idx, double * vec, double * buffer) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
        memcpy(<void *>(buffer+ndim*i), <void *>(vec+ndim*idx[i]), ndim * sizeof(double))
    memcpy(<void *>vec, <void *>buffer, ndim * nsize * sizeof(double))
cdef void assign_sorted_scalar_inplace_dbl_ptr_bf(size_t nsize, long * idx, double * arr, double * buffer) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
        buffer[i] = arr[idx[i]]
    memcpy(<void *>arr, <void *>buffer, nsize * sizeof(double))
cdef void assign_sorted_scalar_inplace_uchar_ptr_bf(size_t nsize, long * idx, unsigned char * arr, unsigned char * buffer) noexcept nogil:
    cdef size_t i
    for i in prange(nsize, nogil=True):
        buffer[i] = arr[idx[i]]
    memcpy(<void *>arr, <void *>buffer, nsize * sizeof(unsigned char))

cdef void assign_sorted_vector_inplace_lng_ptr(size_t nsize, size_t ndim, long * idx, long * vec) noexcept nogil:
    cdef long * buffer = <long *> malloc(ndim * nsize * sizeof(long))
    assign_sorted_vector_inplace_lng_ptr_bf(nsize, ndim, idx, vec, buffer)
    free(buffer)
cdef void assign_sorted_scalar_inplace_lng_ptr(size_t nsize, long * idx, long * arr) noexcept nogil:
    cdef long * buffer = <long *> malloc(nsize * sizeof(long))
    assign_sorted_scalar_inplace_lng_ptr_bf(nsize, idx, arr, buffer) 
    free(buffer)
cdef void assign_sorted_vector_inplace_dbl_ptr(size_t nsize, size_t ndim, long * idx, double * vec) noexcept nogil:
    cdef double * buffer = <double *> malloc(ndim * nsize * sizeof(double))
    assign_sorted_vector_inplace_dbl_ptr_bf(nsize, ndim, idx, vec, buffer)
    free(buffer)
cdef void assign_sorted_scalar_inplace_dbl_ptr(size_t nsize, long * idx, double * arr) noexcept nogil:
    cdef double * buffer = <double *> malloc(nsize * sizeof(double))
    assign_sorted_scalar_inplace_dbl_ptr_bf(nsize, idx, arr, buffer)
    free(buffer)
cdef void assign_sorted_scalar_inplace_uchar_ptr(size_t nsize, long * idx, unsigned char * arr) noexcept nogil:
    cdef unsigned char * buffer = <unsigned char *> malloc(nsize * sizeof(unsigned char))
    assign_sorted_scalar_inplace_uchar_ptr_bf(nsize, idx, arr, buffer)
    free(buffer)


cdef void assign_sorted_scalar_dbl_ptr(size_t nsize_out, long * idx, double * inarr, double * outarr) noexcept nogil:
    cdef size_t i
    for i in prange(nsize_out, nogil=True):
        outarr[i] = inarr[idx[i]]
cdef void assign_sorted_vector_dbl_ptr(size_t nsize_out, size_t ndim, long * idx, double * inarr, double * outarr) noexcept nogil:
    cdef size_t i
    for i in prange(nsize_out, nogil=True):
        # outarr[i] = inarr[idx[i]]
        memcpy(<void *>(outarr+ndim*i), <void *>(inarr+ndim*idx[i]), ndim * sizeof(double))

def arg_quicksort(__numpy__.ndarray arr_in):
    cdef long n = arr_in.size
    idxnp = __numpy__.arange(n, dtype=__numpy__.long)
    cdef long [:] idx = idxnp
    arrnp = arr_in.astype(__numpy__.float64)
    cdef double [:] arr = arrnp.ravel(order='K')
    quicksort(arr, idx, 0, n - 1)
    return idxnp

def arg_mergesort(__numpy__.ndarray arr_in):
    cdef long n = arr_in.size
    idxnp = __numpy__.arange(n, dtype=__numpy__.long)
    cdef long [:] idx = idxnp
    arrnp = arr_in.astype(__numpy__.float64)
    cdef double [:] arr = arrnp.ravel(order='K')
    mergesort(arr, idx, 0, n - 1)
    return idxnp

def arg_countingsort(__numpy__.ndarray[cython.integral] arr_in, long max_key=0):
    cdef long keymax 
    if max_key==0:
        keymax = arr_in.max()
    else:
        keymax = max_key
    cdef long n = arr_in.size
    idxnp = __numpy__.empty(n, dtype=__numpy__.long)
    cdef long [:] sorted_idx = idxnp
    arrnp = arr_in.astype(__numpy__.long)
    counts = __numpy__.zeros(keymax + 1, dtype=__numpy__.long)
    cdef long [:] arr = arrnp.ravel(order='K')
    countingsort(arr, sorted_idx, n, keymax)
    count_each_items(arr, counts, n, keymax)
    return idxnp, counts

def arg_binary_search(double x, __numpy__.ndarray[cython.floating] arr_in):
    cdef long n = arr_in.size
    arrnp = arr_in.astype(__numpy__.float64)
    cdef double [:] arr = arrnp.ravel(order='K')
    return binary_search(x, arr, n)


### Root finding


cdef double find_root(func_t_double_par f, void *params, double a, double b) noexcept nogil:
    if f(a, params) * f(b, params) > 0:
        raise ValueError("Function values at the endpoints must have opposite signs.")
    cdef long niter = <long> ceil(log2((b-a)/2e-12)) # 2e-12: tolerance
    cdef long i
    cdef double c, u, v
    for i in range(niter):
        c = (a + b) / 2
        u = f(a, params)
        v = f(c, params)
        if v == 0:
            return c
        elif u * v < 0:
            b = c
        else:
            a = c
    # Regula falsi approximation.
    u = f(a, params)
    v = f(b, params)
    return (a*v - b*u) / (v - u)



### Numerical Integration
cdef void compute_cumulsum_forward(long n, double *y, double * out) noexcept nogil:
    cdef long i
    cdef double s, c
    s = 0
    c = 0
    for i in range(n):
        kahan_sum_iterator(&s, y[i], &c)
        out[i] = s

cdef void compute_cumulsum_backward(long n, double *y, double * out) noexcept nogil:
    cdef long i
    cdef double s, c
    s = 0
    c = 0
    for i in range(n):
        kahan_sum_iterator(&s, y[i], &c)
        out[n-i-1] = s

cdef double integ_u_midpoint(size_t n, double * y, double h) noexcept nogil:
    cdef size_t i
    cdef double integral=0
    cdef double c = 0
    for i in range(n):
        kahan_sum_iterator(&integral, y[i], &c)
    return integral * h

cdef double integ_u_trapezoid(size_t n, double * y, double h) noexcept nogil:
    cdef size_t i
    cdef double integral=0
    cdef double c = 0
    for i in range(1, n-1):
        kahan_sum_iterator(&integral, y[i], &c)
    kahan_sum_iterator(&integral, 0.5*(y[0]+y[n-1]), &c)
    return integral * h
    
cdef double integ_u_simpson(size_t n, double * y, double h) noexcept nogil:
    cdef size_t i
    cdef double integral=0
    cdef double c = 0
    for i in range(1, n - 1, 2):
        kahan_sum_iterator(&integral, 4*y[i], &c)
    for i in range(2, n - 2, 2):
        kahan_sum_iterator(&integral, 2*y[i], &c)
    kahan_sum_iterator(&integral, y[0], &c)
    kahan_sum_iterator(&integral, y[n-1], &c)
    return integral * h / 3

def simpson_integrate(double [:] y, double h) -> cython.floating:
    cdef size_t n = y.size
    cdef double * ynp = &y[0]
    return integ_u_simpson(n, ynp, h)

cdef double riemann_sum_r(size_t n, double * y, double * dx) noexcept nogil:
    # obtain dx before running this by finite_difference_b(n, x, dx)
    cdef size_t i
    cdef double integral=0
    cdef double c=0
    for i in range(1,n):
        kahan_sum_iterator(&integral, y[i]*dx[i], &c)
    return integral

### Differentiation
cdef void finite_difference_f(long n, double * y, double * dy) noexcept nogil:
    cdef long i
    for i in prange(n-1, nogil=True):
        dy[i] = (y[i+1] - y[i])

cdef void finite_difference_b(long n, double * y, double * dy) noexcept nogil:
    cdef long i
    for i in prange(1,n, nogil=True):
        dy[i] = (y[i] - y[i-1])

# # Testmath
# cdef double invsqcub(double x) nogil:
#     return 1/(sqrt(x*x*x))

from .lia cimport lia_0_dcopy, lia_sv_scalar_inplace

cdef void fft_complex_abs(double * y, fft_complex * x, size_t n) noexcept nogil:
    cdef size_t i
    for i in prange(n, nogil=True):
        y[i] = sqrt(x[i].real*x[i].real + x[i].imag*x[i].imag)

cdef void fft_complex_real(double * y, fft_complex * x, size_t n) noexcept nogil:
    lia_0_dcopy(n, (<double *>x), 2, y, 1)
    # cdef size_t i
    # for i in prange(n, nogil=True):
    #     y[i] = x[i].real

cdef void fft_complex_imag(double * y, fft_complex * x, size_t n) noexcept nogil:
    lia_0_dcopy(n, (<double *>x)+1, 2, y, 1)
    # cdef size_t i
    # for i in prange(n, nogil=True):
    #     y[i] = x[i].imag

cdef void fft_complex_conj_inplace(fft_complex * z, size_t n) noexcept nogil:
    lia_sv_scalar_inplace(-1, (<double *> z)+1, n, 2)
    # cdef size_t i
    # for i in prange(n, nogil=True):
    #     z[i].imag = -z[i].imag
cdef void fft_complex_conj(fft_complex * y, fft_complex * x, size_t n) noexcept nogil:
    memcpy(<void *>y, <void *>x, n * sizeof(fft_complex))
    fft_complex_conj_inplace(y, n)
    # cdef size_t i
    # for i in prange(n, nogil=True):
    #     y[i].real = x[i].real
    #     y[i].imag = -x[i].imag



cdef void fft_complex_scal_r_inplace(double      a, fft_complex * x, size_t n) noexcept nogil:
    lia_sv_scalar_inplace(a, (<double *> x), 2*n, 1)
    # cdef size_t i
    # for i in prange(n, nogil=True):
    #     x[i].real *= y
    #     x[i].imag *= y
cdef void fft_complex_scal_c_inplace(fft_complex a, fft_complex * x, size_t n) noexcept nogil:
    cdef size_t i
    cdef double real, imag
    for i in prange(n, nogil=True):
        real = x[i].real*a.real - x[i].imag*a.imag
        imag = x[i].real*a.imag + x[i].imag*a.real
        x[i].real = real
        x[i].imag = imag
cdef void fft_complex_scal_r(fft_complex * z, double      a, fft_complex * x, size_t n) noexcept nogil:
    memcpy(<void *>z, <void *>x, n * sizeof(fft_complex))
    fft_complex_scal_r_inplace(a, z, n)
    # cdef size_t i
    # for i in prange(n, nogil=True):
    #     z[i].real = x[i].real*y
    #     z[i].imag = x[i].imag*y
cdef void fft_complex_scal_c(fft_complex * z, fft_complex a, fft_complex * x, size_t n) noexcept nogil:
    cdef size_t i
    for i in prange(n, nogil=True):
        z[i].real = x[i].real*a.real - x[i].imag*a.imag
        z[i].imag = x[i].real*a.imag + x[i].imag*a.real


cdef void fft_complex_mul_inplace(fft_complex * x, fft_complex * y, size_t n) noexcept nogil:
    cdef size_t i
    cdef double real, imag
    for i in prange(n, nogil=True):
        real = x[i].real*y[i].real - x[i].imag*y[i].imag
        imag = x[i].real*y[i].imag + x[i].imag*y[i].real
        x[i].real = real
        x[i].imag = imag
cdef void fft_complex_mul(fft_complex * z, fft_complex * x, fft_complex * y, size_t n) noexcept nogil:
    cdef size_t i
    cdef double real, imag
    for i in prange(n, nogil=True):
        real = x[i].real*y[i].real - x[i].imag*y[i].imag
        imag = x[i].real*y[i].imag + x[i].imag*y[i].real
        z[i].real = real
        z[i].imag = imag
cdef void fft_complex_add_inplace(fft_complex * x, fft_complex * y, size_t n) noexcept nogil:
    cdef size_t i
    for i in prange(n, nogil=True):
        x[i].real += y[i].real
        x[i].imag += y[i].imag
cdef void fft_complex_add(fft_complex * z, fft_complex * x, fft_complex * y, size_t n) noexcept nogil:
    cdef size_t i
    for i in prange(n, nogil=True):
        z[i].real = x[i].real + y[i].real
        z[i].imag = x[i].imag + y[i].imag
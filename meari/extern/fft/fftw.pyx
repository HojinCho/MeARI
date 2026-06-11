# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True



# https://www.fftw.org/doc/Advanced-Complex-DFTs.html
# https://www.fftw.org/doc/Advanced-Real_002ddata-DFTs.html
# https://www.fftw.org/doc/Real_002ddata-DFT-Array-Format.html

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cython.parallel import prange
from libc.math cimport (
    sqrt,
)

cdef extern from "fftw3.h":
    ctypedef struct fftw_plan_s:
        pass
    ctypedef fftw_plan_s * fftw_plan
    ctypedef double fftw_complex[2]
    const int FFTW_FORWARD
    const int FFTW_BACKWARD
    const unsigned int FFTW_ESTIMATE
    const unsigned int FFTW_MEASURE
    const unsigned int FFTW_PATIENT
    const unsigned int FFTW_EXHAUSTIVE
    const unsigned int FFTW_WISDOM_ONLY

    # Memory allocation # https://www.fftw.org/doc/Memory-Allocation.html
    void *fftw_malloc(size_t n) noexcept nogil
    void fftw_free(void *p) noexcept nogil

    # Complex DFT Plans # https://www.fftw.org/doc/Complex-DFTs.html
    fftw_plan fftw_plan_dft_1d(int n0, fftw_complex *in_, fftw_complex *out, int sign, unsigned int flags) noexcept nogil
    fftw_plan fftw_plan_dft(int rank, int *n, fftw_complex *in_, fftw_complex *out, int sign, unsigned int flags) noexcept nogil
    # Real-to-Complex DFT Plans # https://www.fftw.org/doc/Real_002ddata-DFTs.html
    fftw_plan fftw_plan_dft_r2c_1d(int n0, double *in_, fftw_complex *out, unsigned int flags) noexcept nogil
    fftw_plan fftw_plan_dft_r2c(int rank, int *n, double *in_, fftw_complex *out, unsigned int flags) noexcept nogil
    fftw_plan fftw_plan_dft_c2r_1d(int n0, fftw_complex *in_, double *out, unsigned int flags) noexcept nogil
    fftw_plan fftw_plan_dft_c2r(int rank, int *n, fftw_complex *in_, double *out, unsigned int flags) noexcept nogil
    # Plan-related functions # https://www.fftw.org/doc/Using-Plans.html
    void fftw_execute(fftw_plan plan) noexcept nogil
    void fftw_destroy_plan(fftw_plan plan) noexcept nogil
    # void fftw_cleanup(void) noexcept nogil
    double fftw_cost(fftw_plan plan) noexcept nogil
    void fftw_flops(fftw_plan plan, double *add, double *mul, double *fma) noexcept nogil # Given plan, # of add, mul, fused-mu-ad

ctypedef enum FFTType:
    C2C
    C2R

from .fft cimport fft_complex, packet_fft

include "fft/fft_common.pyx"

# ctypedef struct packet_fftw:
#     FFTType type
#     size_t n
#     fftw_plan forw
#     fftw_plan back
#     # Buffers
#     fftw_complex * z
#     fftw_complex * w
#     double * r

ctypedef struct plan_fftw:
    fftw_plan forw
    fftw_plan back
    FFTType type

# https://www.fftw.org/doc/Planner-Flags.html
# Note on Algorithm: r2c algorigthm "destroys" input array.
# Thus, p.r should be only used as temporary.
# This is the exact reason why we have buffers inside plan struct.

# In practice, after init, one should use memcpy to transfer data from outside to inside.
cdef void fft_plan_init(packet_fft ** packet, size_t n) noexcept nogil:
    packet[0] = <packet_fft *>malloc(sizeof(packet_fft))
    packet[0].plan = <void *>malloc(sizeof(plan_fftw))
    cdef plan_fftw * p = <plan_fftw *>packet[0].plan
    p.type = FFTType.C2C
    packet[0].n = n
    packet[0].z = <fft_complex *>fftw_malloc(packet[0].n * sizeof(fft_complex))
    packet[0].w = <fft_complex *>fftw_malloc(packet[0].n * sizeof(fft_complex))
    p.forw = fftw_plan_dft_1d(packet[0].n, <fftw_complex *> packet[0].z, <fftw_complex *> packet[0].w, FFTW_FORWARD,  FFTW_MEASURE)
    p.back = fftw_plan_dft_1d(packet[0].n, <fftw_complex *> packet[0].w, <fftw_complex *> packet[0].z, FFTW_BACKWARD, FFTW_MEASURE)
cdef void fft_plan_init_real(packet_fft ** packet, size_t n) noexcept nogil:
    packet[0] = <packet_fft *>malloc(sizeof(packet_fft))
    packet[0].plan = <void *>malloc(sizeof(plan_fftw))
    cdef plan_fftw * p = <plan_fftw *>packet[0].plan
    p.type = FFTType.C2R
    packet[0].n = n
    # for z, only need n//2+1, but just to be safe to prevent segfault.
    packet[0].z = <fft_complex *>fftw_malloc((1+packet[0].n//2) * sizeof(fft_complex))
    # packet[0].z = <fft_complex *>fftw_malloc(packet[0].n * sizeof(fftw_complex))
    packet[0].r = <double *>fftw_malloc(packet[0].n * sizeof(double))
    p.forw = fftw_plan_dft_r2c_1d(packet[0].n, packet[0].r, <fftw_complex *> packet[0].z, FFTW_MEASURE)
    p.back = fftw_plan_dft_c2r_1d(packet[0].n, <fftw_complex *> packet[0].z, packet[0].r, FFTW_MEASURE)
cdef void fft_plan_fin(packet_fft * packet) noexcept nogil:
    cdef plan_fftw * p = <plan_fftw *>packet.plan
    fftw_destroy_plan(p.forw)
    fftw_destroy_plan(p.back)
    fftw_free(packet.z)
    if p.type == FFTType.C2C:
        fftw_free(packet.w)
    else: # p.type == FFTType.C2R
        fftw_free(packet.r)
    free(packet.plan)
    free(packet)

cdef void fft_plan_data_forw_in( packet_fft * packet, void * x, size_t n) noexcept nogil:
    cdef size_t nwr = n
    if packet.n < nwr: # safeguarding overflow
        nwr = packet.n 
    if (<plan_fftw *>packet.plan).type == FFTType.C2C:
        memcpy(<void *>(packet.z), x, nwr * sizeof(fftw_complex))
    else: # p.type == FFTType.C2R
        memcpy(<void *>(packet.r), x, nwr * sizeof(double))
    if nwr < packet.n: # zero-pad
        if (<plan_fftw *>packet.plan).type == FFTType.C2C:
            for i in range(nwr, packet.n):
                packet.z[i] = 0
        else: # p.type == FFTType.C2R
            for i in range(nwr, packet.n):
                packet.r[i] = 0
cdef void fft_plan_data_forw_out(packet_fft * packet, void * y, size_t n) noexcept nogil:
    cdef size_t nwr, ndata
    if (<plan_fftw *>packet.plan).type == FFTType.C2C:
        nsize = packet.n
        nwr = n
        if nsize < nwr: # safeguarding overflow
            nwr = nsize
        memcpy(y, <void *>(packet.w), nwr * sizeof(fft_complex))
    else: # p.type == FFTType.C2R
        nsize = 1 + packet.n//2
        nwr = 1 + n//2
        if nsize < nwr: # safeguarding overflow
            nwr = nsize
        memcpy(y, <void *>(packet.z), nwr * sizeof(fft_complex))
        # memcpy(y, <void *>(packet.z), n * sizeof(fft_complex))
    # no zero-pad on output; zero-padding should be handled by outside.
cdef void fft_plan_data_back_in( packet_fft * packet, void * x, size_t n) noexcept nogil:
    cdef size_t nwr, ndata
    if (<plan_fftw *>packet.plan).type == FFTType.C2C:
        nsize = packet.n
        nwr = n
        if nsize < nwr: # safeguarding overflow
            nwr = nsize
        memcpy(<void *>(packet.w), x, nwr * sizeof(fftw_complex))
    else: # p.type == FFTType.C2R
        nsize = 1 + packet.n//2
        nwr = 1 + n//2
        if nsize < nwr: # safeguarding overflow
            nwr = nsize
        memcpy(<void *>(packet.z), x, nwr * sizeof(fftw_complex))
        # memcpy(<void *>(packet.z), x, n * sizeof(fftw_complex))
    if nwr < nsize: # zero-pad
        if (<plan_fftw *>packet.plan).type == FFTType.C2C:
            for i in range(nwr, nsize):
                packet.w[i] = 0
        else: # p.type == FFTType.C2R
            for i in range(nwr, nsize):
                packet.z[i] = 0
cdef void fft_plan_data_back_out(packet_fft * packet, void * y, size_t n) noexcept nogil:
    cdef size_t nwr = n
    if packet.n < nwr: # safeguarding overflow
        nwr = packet.n 
    if (<plan_fftw *>packet.plan).type == FFTType.C2C:
        memcpy(y, <void *>(packet.z), nwr * sizeof(fft_complex))
    else: # p.type == FFTType.C2R
        memcpy(y, <void *>(packet.r), nwr * sizeof(double))
    # no zero-pad on output; zero-padding should be handled by outside.

cdef void fft_forw(packet_fft * packet) noexcept nogil:
    fftw_execute((<plan_fftw *>packet.plan).forw)
cdef void fft_back(packet_fft * packet) noexcept nogil:
    fftw_execute((<plan_fftw *>packet.plan).back)

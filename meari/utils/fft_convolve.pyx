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

from ..extern.fft cimport(
    fft_complex, fft_plan_init, fft_plan_init_real, fft_plan_fin, 
    fft_plan_data_forw_in, fft_plan_data_forw_out, fft_plan_data_back_in, fft_plan_data_back_out, 
    fft_forw, fft_back, 
    fft_complex_abs, fft_complex_real, fft_complex_imag, 
    fft_complex_conj, fft_complex_conj_inplace, 
    fft_complex_scal_r, fft_complex_scal_c, fft_complex_scal_r_inplace, fft_complex_scal_c_inplace, 
    fft_complex_mul, fft_complex_mul_inplace, fft_complex_add, fft_complex_add_inplace, 
)

from .fft_convolve cimport (
    plan_fftconv, packet_fftconv, 
    FFTConvMode, 
    fftconv_f, fftconv_0_f, 
)

ctypedef struct plan_fftconv:
    FFTConvMode mode
    fftconv_f   f_conv
    fftconv_0_f f_conv_0
    packet_fftconv * packet

ctypedef struct packet_fftconv:
    void * fftplan
    size_t n1
    double * signal_1
    size_t n2
    double * signal_2

cdef void init_fftconv(void ** plan, size_t n, FFTConvMode mode) noexcept nogil:
    plan[0] = malloc(sizeof(plan_fftconv))
    cdef plan_fftconv * pl = <plan_fftconv *>plan[0]

    pl.mode = mode
    if mode == CONVOLUTION or mode == CONVOLUTION_0:
        pl.f_conv = &fft_convolution
        pl.f_conv_0 = &fft_convolution_0
    else: #mode == CORRELATION or mode == CORRELATION_0:
        pl.f_conv = &fft_correlation
        pl.f_conv_0 = &fft_correlation_0
    
    pl.packet = <packet_fftconv *>malloc(sizeof(packet_fftconv))
    cdef size_t n = n1 + n2 - 1
    fft_plan_init_real(&(pl.packet.fftplan), n)
    pl.packet.n1 = n1
    pl.packet.n2 = n2
    pl.packet.signal_1 = <double *>malloc(n * sizeof(double))
    pl.packet.signal_2 = <double *>malloc(n * sizeof(double))

cdef void fin_fftconv(void * plan) noexcept nogil:
    cdef plan_fftconv * pl = <plan_fftconv *>plan
    cdef packet_fftconv * pk = pl.packet
    fft_plan_fin(pk.fftplan)
    free(pk.signal_1)
    free(pk.signal_2)
    free(pl.packet)
    free(pl)


cdef void fft_conv(void * plan) noexcept nogil:
    (<plan_fftconv *>plan).f_conv((<plan_fftconv *>plan).packet)
cdef double fft_conv0(void * plan) noexcept nogil:
    return (<plan_fftconv *>plan).f_conv_0((<plan_fftconv *>plan).packet)

cdef void fft_convolution(packet_fftconv * packet) noexcept nogil
cdef void fft_correlation(packet_fftconv * packet) noexcept nogil
cdef double fft_convolution_0(packet_fftconv * packet) noexcept nogil
cdef double fft_correlation_0(packet_fftconv * packet) noexcept nogil


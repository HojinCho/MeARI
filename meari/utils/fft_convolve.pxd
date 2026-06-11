ctypedef enum FFTConvMode:
    CONVOLUTION
    CORRELATION
    CONVOLUTION_0
    CORRELATION_0


ctypedef void (*fftconv_f)(packet_fftconv * packet) noexcept nogil
ctypedef double (*fftconv_0_f)(packet_fftconv * packet) noexcept nogil

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

cdef void init_fftconv(void ** plan, size_t n, FFTConvMode mode) noexcept nogil
cdef void fin_fftconv(void * plan) noexcept nogil    

cdef void fft_conv(void * plan) noexcept nogil
cdef double fft_conv0(void * plan) noexcept nogil

cdef void fft_convolution(packet_fftconv * packet) noexcept nogil
cdef void fft_correlation(packet_fftconv * packet) noexcept nogil
cdef double fft_convolution_0(packet_fftconv * packet) noexcept nogil
cdef double fft_correlation_0(packet_fftconv * packet) noexcept nogil


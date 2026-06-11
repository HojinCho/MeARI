ctypedef double complex fft_complex

ctypedef struct packet_fft:
    size_t n
    # Buffers
    fft_complex * z
    fft_complex * w
    double * r
    void * plan

cdef void fft_plan_init(packet_fft ** packet, size_t n) noexcept nogil
cdef void fft_plan_init_real(packet_fft ** packet, size_t n) noexcept nogil
cdef void fft_plan_fin(packet_fft * packet) noexcept nogil

cdef void fft_forw(packet_fft * packet) noexcept nogil
cdef void fft_back(packet_fft * packet) noexcept nogil

cdef void fft_plan_data_forw_in( packet_fft * packet, void * x, size_t n) noexcept nogil
cdef void fft_plan_data_forw_out(packet_fft * packet, void * y, size_t n) noexcept nogil
cdef void fft_plan_data_back_in( packet_fft * packet, void * x, size_t n) noexcept nogil
cdef void fft_plan_data_back_out(packet_fft * packet, void * y, size_t n) noexcept nogil


cdef void fft_complex_abs(double * y, fft_complex * x, size_t n) noexcept nogil

cdef void fft_complex_real(double * y, fft_complex * x, size_t n) noexcept nogil
cdef void fft_complex_imag(double * y, fft_complex * x, size_t n) noexcept nogil

cdef void fft_complex_conj(fft_complex * y, fft_complex * x, size_t n) noexcept nogil
cdef void fft_complex_conj_inplace(fft_complex * z, size_t n) noexcept nogil

cdef void fft_complex_scal_r(fft_complex * z, double      a, fft_complex * x, size_t n) noexcept nogil
cdef void fft_complex_scal_c(fft_complex * z, fft_complex a, fft_complex * x, size_t n) noexcept nogil
cdef void fft_complex_scal_r_inplace(double      a, fft_complex * x, size_t n) noexcept nogil
cdef void fft_complex_scal_c_inplace(fft_complex a, fft_complex * x, size_t n) noexcept nogil

cdef void fft_complex_mul(fft_complex * z, fft_complex * x, fft_complex * y, size_t n) noexcept nogil
cdef void fft_complex_mul_inplace(fft_complex * x, fft_complex * y, size_t n) noexcept nogil
cdef void fft_complex_add(fft_complex * z, fft_complex * x, fft_complex * y, size_t n) noexcept nogil
cdef void fft_complex_add_inplace(fft_complex * x, fft_complex * y, size_t n) noexcept nogil
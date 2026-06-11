# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True


ctypedef enum VarType:
    DRW
    DRW_FAST
    LOG_DRW
    LOG_DRW_FAST
    # CARMA # To be implemented?


########### Model Variability for Driving ###########

# DRW
cdef double var_DRW_oneside(double t, void *params) noexcept nogil
cdef double var_DRW(double t, void *params) noexcept nogil
cdef double var_dft_DRW(double f, double dt, void *params) noexcept nogil
cdef void set_par_var_DRW(void *params, double sigma_d, double tau_d) noexcept nogil
cdef void get_par_var_DRW(void *params, double * sigma_d, double * tau_d) noexcept nogil
cdef double get_invtau_var_DRW(void *params) noexcept nogil
cdef struct par_var_DRW:
    double sigma_d 
    double tau_d 

# DRW, but should be a bit faster, especially for GPUs / old CPUs that don't handle division well.
# However, this would require post-processing of the parameters.
# Use this if driving var. parameter is irrelevant, or if CPU time is extremely precious
cdef double var_DRW_fast_oneside(double t, void *params) noexcept nogil
cdef double var_DRW_fast(double t, void *params) noexcept nogil
cdef double var_dft_DRW_fast(double f, double dt, void *params) noexcept nogil
cdef void set_par_var_DRW_fast(void *params, double a_d, double itd) noexcept nogil
cdef void get_par_var_DRW_fast(void *params, double * sigma_d, double * tau_d) noexcept nogil
cdef double get_invtau_var_DRW_fast(void *params) noexcept nogil
cdef struct par_var_DRW_fast:
    # Would this matter?
    double a_d       # For performance; a_d == 2 log_e sigma_d
    double inv_tau_d # For performance; inv_tau_d == 1/tau_d


# Magspace to Fluxspace
cdef double magkernel_to_fluxkernel(double magker, double autovar) noexcept nogil
cdef struct par_var_LOG:
    double auto_var
    void * par_var
cdef double var_LOG_DRW_fast_oneside(double t, void *params) noexcept nogil
cdef double var_LOG_DRW_fast(double t, void *params) noexcept nogil
cdef void set_par_var_LOG_DRW_fast(void *params, double a_d, double itd) noexcept nogil
cdef void get_par_var_LOG_DRW_fast(void *params, double * sigma_d, double * tau_d) noexcept nogil
cdef double get_invtau_var_LOG_DRW_fast(void *params) noexcept nogil
cdef double var_LOG_DRW_oneside(double t, void *params) noexcept nogil
cdef double var_LOG_DRW(double t, void *params) noexcept nogil
cdef void set_par_var_LOG_DRW(void *params, double sigma_d, double tau_d) noexcept nogil
cdef void get_par_var_LOG_DRW(void *params, double * sigma_d, double * tau_d) noexcept nogil
cdef double get_invtau_var_LOG_DRW(void *params) noexcept nogil
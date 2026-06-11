# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

cimport cython
from cython.parallel import prange

from libc.stdlib cimport malloc, free
from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, cos, sin, pow, exp,
    fmin, fmax, fabs,
)

cdef double twoPI = 2*M_PI

from .variability cimport VarType, par_var_DRW, par_var_DRW_fast, par_var_LOG


cdef double var_DRW_fast_oneside(double t, void *params) noexcept nogil:
    return exp((<par_var_DRW_fast*>params).a_d - t*(<par_var_DRW_fast*>params).inv_tau_d)
cdef double var_DRW_fast(double t, void *params) noexcept nogil:
    return var_DRW_fast_oneside(fabs(t), params)
cdef double var_dft_DRW_fast(double f, double dt, void *params) noexcept nogil:
    cdef double e_term, c_term
    c_term = cos(twoPI*f*dt)
    e_term = exp(-dt*(<par_var_DRW_fast*>params).inv_tau_d)
    # return (dt*exp((<par_var_DRW*>params).a_d))*(1+2*e_term*(c_term-e_term)/(1-2*c_term*e_term+e_term*e_term))
    # Basis change from C, E to CE, E^2.
    c_term = c_term*e_term # Never mix this order.
    e_term = e_term*e_term # This comes last.
    return (exp((<par_var_DRW_fast*>params).a_d))*(1+2*(c_term-e_term)/(1-2*c_term+e_term)) # * dt is omitted since it is divided after ifft
# cdef void set_par_var_DRW_fast(void *params, double sigma_d, double tau_d) noexcept nogil:
#     (<par_var_DRW_fast*>params).a_d = 2*log(sigma_d)
#     (<par_var_DRW_fast*>params).inv_tau_d = 1/tau_d
cdef void set_par_var_DRW_fast(void *params, double a_d, double itd) noexcept nogil:
    (<par_var_DRW_fast*>params).a_d = a_d
    (<par_var_DRW_fast*>params).inv_tau_d = itd
# cdef void get_par_var_DRW_fast(void *params, double * a_d, double * itd) noexcept nogil:
#     a_d[0] = exp(0.5*(<par_var_DRW_fast*>params).a_d)
#     itd[0] = 1./(<par_var_DRW_fast*>params).inv_tau_d
cdef void get_par_var_DRW_fast(void *params, double * sigma_d, double * tau_d) noexcept nogil:
    sigma_d[0] = exp(0.5*(<par_var_DRW_fast*>params).a_d)
    tau_d[0]   = 1./(<par_var_DRW_fast*>params).inv_tau_d
cdef double get_invtau_var_DRW_fast(void *params) noexcept nogil:
    return (<par_var_DRW_fast*>params).inv_tau_d

cdef double var_DRW_oneside(double t, void *params) noexcept nogil:
    return ((<par_var_DRW*>params).sigma_d)*((<par_var_DRW*>params).sigma_d)*exp(
        - t/(<par_var_DRW*>params).tau_d
    )
cdef double var_DRW(double t, void *params) noexcept nogil:
    return var_DRW_oneside(fabs(t), params)
cdef double var_dft_DRW(double f, double dt, void *params) noexcept nogil:
    cdef double e_term, c_term, sigma_d
    sigma_d = (<par_var_DRW*>params).sigma_d
    c_term = cos(twoPI*f*dt)
    e_term = exp(-dt/(<par_var_DRW*>params).tau_d)
    # return (sigma_d*sigma_d*dt)*(1+2*e_term*(c_term-e_term)/(1-2*c_term*e_term+e_term*e_term))
    # Basis change from C, E to CE, E^2.
    c_term = c_term*e_term # Never mix this order.
    e_term = e_term*e_term # This comes last.
    return (sigma_d*sigma_d)*(1+2*(c_term-e_term)/(1-2*c_term+e_term)) # * dt is omitted since it is divided after ifft
cdef void set_par_var_DRW(void *params, double sigma_d, double tau_d) noexcept nogil:
    (<par_var_DRW*>params).sigma_d = sigma_d
    (<par_var_DRW*>params).tau_d = tau_d
cdef void get_par_var_DRW(void *params, double * sigma_d, double * tau_d) noexcept nogil:
    sigma_d[0] = (<par_var_DRW*>params).sigma_d
    tau_d[0]   = (<par_var_DRW*>params).tau_d
cdef double get_invtau_var_DRW(void *params) noexcept nogil:
    return 1./(<par_var_DRW*>params).tau_d

#######################################
# In flux space
#######################################
cdef double magkernel_to_fluxkernel(double magker, double autovar) noexcept nogil:
    # autovar = exp(0.16*M_LN10*M_LN10*magker0)
    cdef double A = 0.16*M_LN10*M_LN10
    return autovar*(exp(A*magker)-1.0)

cdef double var_LOG_DRW_fast_oneside(double t, void *params) noexcept nogil:
    return magkernel_to_fluxkernel(
        var_DRW_fast_oneside(t, (<par_var_LOG*> params).par_var),
        (<par_var_LOG*> params).auto_var,
    )
cdef double var_LOG_DRW_fast(double t, void *params) noexcept nogil:
    return var_LOG_DRW_fast_oneside(fabs(t), params)
cdef void set_par_var_LOG_DRW_fast(void *params, double a_d, double itd) noexcept nogil:
    cdef double A = 0.16*M_LN10*M_LN10
    (<par_var_DRW_fast*>((<par_var_LOG*>params).par_var)).a_d       = a_d
    (<par_var_DRW_fast*>((<par_var_LOG*>params).par_var)).inv_tau_d = itd
    ((<par_var_LOG*>params).auto_var) = exp(A*exp(a_d))
cdef void get_par_var_LOG_DRW_fast(void *params, double * sigma_d, double * tau_d) noexcept nogil:
    sigma_d[0] = exp(0.5*(<par_var_DRW_fast*>((<par_var_LOG*>params).par_var)).a_d)
    tau_d[0]   = 1./(<par_var_DRW_fast*>((<par_var_LOG*>params).par_var)).inv_tau_d
cdef double get_invtau_var_LOG_DRW_fast(void *params) noexcept nogil:
    return (<par_var_DRW_fast*>((<par_var_LOG*>params).par_var)).inv_tau_d

cdef double var_LOG_DRW_oneside(double t, void *params) noexcept nogil:
    return magkernel_to_fluxkernel(
        var_DRW_oneside(t, (<par_var_LOG*> params).par_var),
        (<par_var_LOG*> params).auto_var,
    )
cdef double var_LOG_DRW(double t, void *params) noexcept nogil:
    return var_LOG_DRW_oneside(fabs(t), params)
cdef void set_par_var_LOG_DRW(void *params, double sigma_d, double tau_d) noexcept nogil:
    cdef double A = 0.16*M_LN10*M_LN10
    (<par_var_DRW*>((<par_var_LOG*>params).par_var)).sigma_d = sigma_d
    (<par_var_DRW*>((<par_var_LOG*>params).par_var)).tau_d   = tau_d
    ((<par_var_LOG*>params).auto_var) = exp(A*sigma_d*sigma_d)
cdef void get_par_var_LOG_DRW(void *params, double * sigma_d, double * tau_d) noexcept nogil:
    sigma_d[0] = (<par_var_DRW*>((<par_var_LOG*>params).par_var)).sigma_d
    tau_d[0]   = (<par_var_DRW*>((<par_var_LOG*>params).par_var)).tau_d
cdef double get_invtau_var_LOG_DRW(void *params) noexcept nogil:
    return 1./(<par_var_DRW*>((<par_var_LOG*>params).par_var)).tau_d
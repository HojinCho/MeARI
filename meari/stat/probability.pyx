# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

import numpy as __numpy__
cimport numpy as __numpy__
cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, 
)

from ..utils.algorithms cimport (
    kahan_sum_iterator, 
    integ_u_simpson,
)


# # Defining constants
# cdef double twoPI = 2*M_PI
# cdef double neglogtwoPI = -0.5*log(twoPI)

# cdef double gaussian_loglike(long n, double [:] model, double [:] obs, double [:] err):
#     cdef long i
#     cdef double loglike, c
#     cdef double * buffer = <double *> malloc(n*sizeof(double))
#     for i in prange(n, nogil=True):
#         c = (obs[i] - model[i])/err[i]
#         buffer[i] = -0.5*buffer[i]*buffer[i] - log(err[i]) + neglogtwoPI

#     loglike = 0
#     c = 0
#     for i in range(n):
#         kahan_sum_iterator(&loglike, buffer[i], &c)
#     free(buffer)
#     return loglike

# cdef double logprior(double [:] theta, ):
#     if theta[0]>=1 and theta[1]>=0 and theta[2]>=0 and theta[2]<60:
#         return 0
#     return -__numpy__.inf


# def loglikelihood_gaussian(
#         lc_resp_obs_t, lc_resp_obs_flux, lc_resp_obs_err,
#         lc_resp_model_t, lc_resp_model_flux,
# ):
#     model_flux = np.interp(lc_resp_obs_t, lc_resp_model_t, lc_resp_model_flux)
#     nobs = lc_resp_obs_flux.size
#     diff = ((lc_resp_obs_flux - model_flux)/lc_resp_obs_err)
#     return -0.5*np.sum(diff*diff) - np.sum(np.log(lc_resp_obs_err)) - 0.5*nobs*np.log(2*np.pi)

# def loglike(alpha, beta, incl, inmodel, dt, tau_edges, tau_cents, obsdata):
#     model = model_lc(alpha, beta, incl, inmodel, dt=dt, tau_edges=tau_edges, tau_cents=tau_cents)
#     return loglikelihood_gaussian(*obsdata, *model)

# def logprior(alpha, beta, incl):
#     if alpha>=1 and beta>=0 and incl>=0 and incl<60:
#         return 0
#     return -np.inf

# def logposterior(theta, inmodel, dt, tau_edges, tau_cents, obsdata):
#     lp = logprior(*theta)
#     if np.isfinite(lp):
#         return lp + loglike(*theta, inmodel, dt, tau_edges, tau_cents, obsdata)
#     return lp

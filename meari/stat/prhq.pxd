
from ..utils.types cimport (
    func_t_double_par, func_t_param_2, func_t_param_dblptr2,
    func_t_void_void_ptr, func_t_double_void_ptr,
)
from .variability cimport VarType, par_var_DRW, par_var_DRW_fast #, par_var_CARMA
from .lightcurves cimport LightCurves, packet_LC

ctypedef enum CovMatMethod:
    PIECEWISE
    QUADRATURE

cdef struct packet_PRHQ:
    # LC information
    packet_LC * lc # light curves object containing everything needed for this.
    size_t nlc
    size_t ndata
    double * y     # pointer to the flux data (don't allocate)
    double * N     # 2d; size ndata*ndada (symmetric full), noise covariance
    double * dt_ij # 2d; size ndata*(ndata+1)//2 (symmetric packed), dt_ij = t_i - t_j

    # Driving Variability Model (DRW, DRW_fast, CARMA, ...)
    VarType var
    func_t_double_par acf
    void * par_acf
    func_t_param_2         set_var_par
    # func_t_param_dblptr2   get_var_par
    func_t_double_void_ptr get_invtau

    # Detrending Parameters
    long * n_det_order # 1d; size nlc, number of detrending parameters (norder) >=0. (0 means continuum).
    size_t n_det_pars  # total number of detrending parameters. \sum_{i=0}^{nlc} (n_det_order[i]+1).
    double * L         # 2d; size (ndata)*(n_det_pars), L[i*n_det_pars+j]=1 if i-th entry is from LC j else 0.
    double * q         # 1d; n_det_pars, q[j] = detrending parameter i.

    # Transfer Function
    size_t ntau
    size_t nresp  
    double * tau  # size ntau
    double * tf   # size nresp * ntau. tf[i*ntau + k] = tf_i(tau_k)
    double dtau

    # PRHQ Model
    double * buffer_mat1 # 2d; size ndata*ndata, buffer for 2d full matrix. Store Cinv.L, or Cq.L^T.
    double * buffer_mat2 # 2d; size ndata*ndata, because dgemm does not allow A and C sharing the same memory...
    double * Cinv        # 2d; size ndata*ndata (symmetric full), inverse of C., use S instead
    double * Cq          # 2d; size n_det_pars*n_det_pars, stores Cq = (L^T.Cinv.L)^-1 use buffer_mat1 instead.

    # Method-specific buffers
    void * bundle
    
    # LAPACK buffer
    int * ipiv
    double * work
    size_t lwork

    # ACF matrix buffer
    double * acf_dtau1   # 1d; size ntau
    double * acf_dtau2   # 2d; size ntau*(ntau+1)//2 (symmetric packed)
    double * acf_buffer2 # 2d; size ntau*(ntau+1)//2 (symmetric packed)
    double * buffer      # 1d; size ntau
    
    func_t_void_void_ptr covmat
    # TODO: add GP matrices (Zu+2011 Eq. 21)

cdef int initialize_PRHQ(packet_PRHQ * p, packet_LC * lc, VarType vartype, long * ndet_orders, size_t ntau, double * tau, double * tf, ) noexcept nogil
cdef void finalize_PRHQ(packet_PRHQ * packet) noexcept nogil
cdef double loglike_PRHQ_detr_marginalized(packet_PRHQ * packet) noexcept nogil
cdef double loglike_PRHQ_detr_full(packet_PRHQ * p) noexcept nogil
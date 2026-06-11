cimport cython
from .lightcurves cimport LightCurves, packet_LC
from .variability cimport VarType
from .prhq cimport packet_PRHQ
from ..mesh.diskmesh cimport packet_Mesh1D, DiskMesh

ctypedef enum LikeType:
    PRHQ      # Q-likelihood by Press, Rybicki & Hewitt (1992)
    SEQ_BAYES # Sequential Bayesian Inference (e.g., Pancoast et al. 2014a)


ctypedef double (*func_t_prhq)(packet_PRHQ * p) noexcept nogil

@cython.final
cdef class Stats:
    cdef:
        packet_LC * lc
        packet_Mesh1D * mesh
        LikeType like_type
        void * packet
        # Transfer Function
        size_t ntau
        size_t nresp
        double * _tau # size ntau
        double * _dtau
        double * _tau0 # size ntau
        double * _dtau0
        double * _tf_pdf  # size nresp * ntau. tf[i*ntau + j] = tf_i(tau_j)
        double * _tf_cdf  # size nresp * ntau. tf[i*ntau + j] = tf_i(tau_j)
        
        # Numpy Interface
        public object tau
        public object dtau
        public object tf_pdf
        public object tf_cdf

    cdef void assign_data(self, LikeType like_type,
        packet_LC * lc, packet_Mesh1D * mesh,
        long * ndet_orders, 
        # Modeling parameters
        size_t nresp, size_t ntau, double tau_0, double tau_1,
        VarType vartype, 
        # DiskMesh parameters?
    ) nogil
    cdef void _c_compute_tf_pdf(self, long nv_good, long nf_good) noexcept nogil
    cdef void _c_compute_tf_cdf(self, long nv_good, long nf_good) noexcept nogil
    cdef void _c_assign_tf_one(self, double * tf, size_t iresp) noexcept nogil
    cdef void _c_assign_tf_all(self, double * tfs) noexcept nogil
    cpdef int assign_tf(self, object tf, size_t iresp=*)
    # cpdef int update_tf(self, size_t iresp, size_t nv_good, size_t nf_good, bint normalized=*, double fvar=*)
    cdef void _set_detrending(self, double * q) noexcept nogil
    cpdef int set_detrending(self, double [:] q)
    cpdef int set_var_par(self, double sigma, double tau)
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

cimport cython
from cython.parallel import prange

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, exp, cos, sin, pow,
    fmin, fmax, fabs,
)

from ..utils.numpy_interface cimport (
    numpy_dbl_1d, numpy_dbl_2d,
    numpy_lng_1d, numpy_lng_2d,
    numpy_ascontiguousarray, numpy_ravel_C,
)


from ..utils.algorithms cimport finite_difference_b, riemann_sum_r

from .prhq cimport (
    packet_PRHQ, initialize_PRHQ, finalize_PRHQ, 
    loglike_PRHQ_detr_marginalized, loglike_PRHQ_detr_full,
)
from .variability cimport VarType
from .lightcurves cimport LightCurves, packet_LC, free_packet_LC
from ..mesh.diskmesh cimport (
    DiskMesh, packet_Mesh1D, free_packet_Mesh1D, # shallow_copy_packet_Mesh1D,
)
from ..mesh.compute_tf cimport compute_tf_pdf, compute_tf_cdf
from ..extern.lia cimport lia_sv_scalar_inplace
from ..physics.thermal cimport Rgrav

class UnderdeterminedError(ValueError):
    pass

# cdef struct packet_PRHQ
# cdef int initialize_PRHQ(packet_PRHQ * packet_out, packet_LC * lc, VarType vartype, long * ndet_orders, size_t ntau, double * tau, double * tf) noexcept nogil
# cdef void finalize_PRHQ(packet_PRHQ * packet) noexcept nogil
# cdef double loglike_PRHQ_detr_marginalized(packet_PRHQ * packet) noexcept nogil

from libc.stdio cimport printf

from .stats cimport LikeType

cdef void compute_taus(
    double tau_0, double tau_1, size_t ntau, double * tau
) noexcept nogil:
    cdef size_t i
    cdef double dtau = (tau_1 - tau_0) / (ntau-1)
    for i in range(ntau):
        tau[i] = tau_0 + dtau * i

# finite_difference_b(n_resp_tau, tau_resps, d_tau_bins) # Precomp when fitting

@cython.final
cdef class Stats:
    def __cinit__(self, 
        # LikeType like_type,
        # packet_LC * lc, long * ndet_orders, 
        # # Modeling parameters
        # size_t nresp, size_t ntau, double tau_0, double tau_1,
        # VarType vartype, 
    ):
        self.lc        = NULL
        self.mesh      = NULL
        self.packet    = NULL
        self._tau      = NULL
        self._dtau     = NULL
        self._tau0     = NULL
        self._dtau0    = NULL
        self._tf_pdf   = NULL
        self._tf_cdf   = NULL
        self.like_type = LikeType.PRHQ
        self.ntau      = 0
        self.nresp     = 0
        
    
    cdef void assign_data(self, LikeType like_type,
        packet_LC * lc, packet_Mesh1D * mesh,
        long * ndet_orders, 
        # Modeling parameters
        size_t nresp, size_t ntau, double tau_0, double tau_1,
        VarType vartype, 
        # DiskMesh parameters?
    ) nogil:
        # Sanity check
        cdef size_t i
        for i in range(lc.NLCs):
            if ndet_orders[i] < 0:
                raise ValueError("Detrending order must be non-negative")
            elif ndet_orders[i]+1 >= lc.npts[i]:
                raise UnderdeterminedError(
                    "Too many detrending parameters, "
                    + "must be less than (number of points - 1) of each light curves"
                )
            elif ndet_orders[i] != 0:
                raise NotImplemented("Not implemented for detrending order > 0")
        self.ntau    = ntau
        self.nresp   = nresp
        self._tau    = <double *>malloc(ntau*sizeof(double))
        self._dtau   = <double *>malloc(ntau*sizeof(double))
        self._tau0   = <double *>malloc(ntau*sizeof(double))
        self._dtau0  = <double *>malloc(ntau*sizeof(double))
        self._tf_pdf = <double *>malloc(nresp*ntau*sizeof(double))
        self._tf_cdf = <double *>malloc(nresp*ntau*sizeof(double))
        if self._tau == NULL or self._dtau == NULL or self._tf_pdf == NULL or self._tf_cdf == NULL:
            raise MemoryError("Could not allocate pointers for the Likelihood Struct")
        compute_taus(tau_0, tau_1, ntau, self._tau)
        finite_difference_b(self.ntau, self._tau, self._dtau) # Precomp when fitting
        # memorize original tau and dtau
        memcpy(<void*>self._tau0,  <void*>self._tau,  ntau*sizeof(double))
        memcpy(<void*>self._dtau0, <void*>self._dtau, ntau*sizeof(double))

        # Is copy ok? Or should I just pass the pointer?
        # Perhaps it is ok since packet mostly holds pointer to the LC object; only pointer is copied.
        self.lc = <packet_LC*> malloc(sizeof(packet_LC))
        self.mesh = <packet_Mesh1D*> malloc(sizeof(packet_Mesh1D))
        memcpy(<void*>self.lc, <void*>lc, sizeof(packet_LC))
        # shallow_copy_packet_Mesh1D(self.mesh, mesh)
        memcpy(<void*>self.mesh, <void*>mesh, sizeof(packet_Mesh1D))
        # printf("%p\n", self._tf_pdf)
        self.like_type = like_type
        if self.like_type == LikeType.PRHQ:
            self.packet = <void *> malloc(sizeof(packet_PRHQ))
            if self.packet == NULL:
                raise MemoryError("Could not allocate pointers for the Likelihood Datapacket Struct")
            if initialize_PRHQ(
                <packet_PRHQ *>self.packet, self.lc, 
                vartype, ndet_orders, ntau, self._tau, self._tf_pdf,
            )!=0:
                raise MemoryError("Could not allocate enough memory for computation")
        else:
            raise NotImplemented("Only PRHQ likelihood is implemented for now")
        
        # printf("%p %p\n", self._tf_pdf, (<packet_PRHQ *>self.packet).tf)

        # Numpy Interface
        with gil:    
            self.tau = numpy_dbl_1d(self._tau, self.ntau, False)
            self.dtau = numpy_dbl_1d(self._dtau, self.ntau, False)
            self.tf_cdf = numpy_dbl_2d(self._tf_cdf, self.nresp, self.ntau, False)
            self.tf_pdf = numpy_dbl_2d(self._tf_pdf, self.nresp, self.ntau, False)
        

    def __dealloc__(self):
        if self.like_type == LikeType.PRHQ:
            if self.packet != NULL:
                finalize_PRHQ(<packet_PRHQ *> self.packet)
        if self.lc != NULL:
            free_packet_LC(self.lc)
        if self.mesh != NULL:
            free_packet_Mesh1D(self.mesh)
        if self._tau != NULL:
            free(self._tau)
        if self._dtau != NULL:
            free(self._dtau)
        if self._tau0 != NULL:
            free(self._tau0)
        if self._dtau0 != NULL:
            free(self._dtau0)
        if self._tf_pdf != NULL:
            free(self._tf_pdf)
        if self._tf_cdf != NULL:
            free(self._tf_cdf)
    
    # Run the DiskMesh First, then run this. No communication needed.
    cdef void _c_compute_tf_pdf(self, long nv_good, long nf_good) noexcept nogil:
        compute_tf_pdf(
            # Response
            self.ntau, self.mesh.ntfs,
            self._tf_pdf,
            self._tf_cdf, # Obtained for free.
            # persistent
            self._tau,
            self._dtau,
            # Mesh Properties
            nv_good, nf_good, 
            self.mesh.nvpf,
            self.mesh.__b_xyz,
            self.mesh.__b_f_v_i,
            self.mesh.__b_v_f_i,
            self.mesh.__b_v_f_n,
            self.mesh.__b_area,
            self.mesh.__b_resp,
            self.mesh.__b_emis,
            self.mesh.__b_tau,
            self.mesh.__b_precomp,
            self.mesh.__b_average,
            self.mesh.__b_cumulsum,
            self.mesh.__b_mult,
            self.mesh.__b_tau_f,
            self.mesh.__b_idx_f,
            self.mesh.lb_nf,
            self.mesh.db_nf,
            self.mesh.db_nfntfs,
            self.mesh.db_ntfs,
            self.mesh.mask_v,
            self.mesh.mask_f,
            self.mesh.mask_potential,
        )
    cdef void _c_compute_tf_cdf(self, long nv_good, long nf_good) noexcept nogil:
        compute_tf_cdf(
            # Response
            self.ntau, self.mesh.ntfs,
            self._tf_cdf,
            # persistent
            self._tau,
            # Mesh Properties
            nv_good, nf_good, 
            self.mesh.nvpf,
            self.mesh.__b_xyz,
            self.mesh.__b_f_v_i,
            self.mesh.__b_v_f_i,
            self.mesh.__b_v_f_n,
            self.mesh.__b_area,
            self.mesh.__b_resp,
            self.mesh.__b_emis,
            self.mesh.__b_tau,
            self.mesh.__b_precomp,
            self.mesh.__b_average,
            self.mesh.__b_cumulsum,
            self.mesh.__b_mult,
            self.mesh.__b_tau_f,
            self.mesh.__b_idx_f,
            self.mesh.lb_nf,
            self.mesh.db_nf,
            self.mesh.db_nfntfs,
            self.mesh.db_ntfs,
            self.mesh.mask_v,
            self.mesh.mask_f,
            self.mesh.mask_potential,
        )
    cdef void _c_assign_tf_one(self, double * tf, size_t iresp) noexcept nogil:
        memcpy(<void *>(self._tf_pdf + iresp*self.ntau), <void *>tf, self.ntau*sizeof(double))
    cdef void _c_assign_tf_all(self, double * tfs) noexcept nogil:
        memcpy(<void *>(self._tf_pdf), <void *>tfs, self.ntau*self.nresp*sizeof(double))
    cpdef int assign_tf(self, object tf, size_t iresp=0):
        cdef double [:] tf_mv
        tf_np = numpy_ascontiguousarray(tf)
        cdef long ndim = len(tf_np.shape)
        if ndim == 2:
            if tf_np.shape[0]==self.nresp and tf_np.shape[1]==self.ntau:
                tf_mv = numpy_ravel_C(tf_np)
            elif tf_np.shape[0]==self.ntau and tf_np.shape[1]==self.nresp:
                tf_mv = numpy_ravel_C(tf_np.T)
            else:
                raise ValueError("tf must be of shape (nresp, ntau)")
            self._c_assign_tf_all(&tf_mv[0])
        elif ndim == 1:
            tf_mv = numpy_ravel_C(tf_np)
            self._c_assign_tf_one(&tf_mv[0], iresp)
        return 0

    cdef void _set_detrending(self, double * q) noexcept nogil:
        memcpy(<void*>((<packet_PRHQ *> self.packet).q), <void*>q, ((<packet_PRHQ *> self.packet).n_det_pars)*sizeof(double))
        return
    cpdef int set_detrending(self, double [:] q):
        with nogil:
            self._set_detrending(&q[0])
        return 0
    cpdef int set_var_par(self, double sigma, double tau):
        with nogil:
            (<packet_PRHQ *> self.packet).set_var_par((<packet_PRHQ *> self.packet).par_acf, sigma, tau)
        return 0

def constructStats(
    LightCurves lc, long [:] ndet_orders,
    # Modeling parameters
    long nresp, long ntau, 
    # DiskMesh parameters?
    DiskMesh mesh,
    var='DRW', taulim=[0, 200], like='PRHQ',
):
    cdef LikeType like_type
    cdef VarType vartype
    cdef double tau_0, tau_1
    cdef packet_LC * lcp
    cdef packet_Mesh1D * meshpacket
    cdef size_t nr, nt
    if like == 'PRHQ':
        like_type = LikeType.PRHQ
    else:
        raise NotImplemented("Only PRHQ likelihood is implemented for now")
    if var == 'DRW':
        vartype = VarType.DRW
    elif var == 'DRW_fast':
        vartype = VarType.DRW_FAST
    # elif var == 'LOG_DRW':
    #     vartype = VarType.LOG_DRW
    # elif var == 'LOG_DRW_fast':
    #     vartype = VarType.LOG_DRW_FAST
    else:
        raise NotImplemented("Only DRW and DRW_fast variability models are implemented for now")
    tau_0 = float(taulim[0])
    tau_1 = float(taulim[1])
    nr = nresp
    nt = ntau
    cdef Stats stat = Stats.__new__(Stats)
    lcp = <packet_LC *>malloc(sizeof(packet_LC))
    meshpacket = <packet_Mesh1D *>malloc(sizeof(packet_Mesh1D))
    with nogil:
        lc._c_expose_packet(lcp)
        mesh.expose_packet(meshpacket)
        stat.assign_data(
            like_type, lcp, meshpacket, &ndet_orders[0], 
            nr, nt, tau_0, tau_1, vartype,
        )
    # since stat is copying the data instead of taking the pointer.
    free(lcp) 
    free(meshpacket)
    return stat

# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

cimport cython
from cython.parallel import prange

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy, memset
from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN, 
    M_PI, M_E, M_LN10, 
    sqrt, log, cos, sin, pow, exp,
    fmin, fmax, fabs,
)
from libc.float cimport DBL_MAX


from ..utils.algorithms cimport (
    kahan_sum_iterator, 
    binary_search_ptr_L, binary_search_ptr_R,
    binary_search_ptr_L_desc, # arr[low-1] >= x > arr[low]
    binary_search_ptr_R_desc, # arr[low-1] > x >= arr[low]
)

from ..utils.types cimport func_t_double_par, func_t_double2_par, func_t_param_dblptr2, func_t_void_void_ptr

from ..extern.lia cimport (
    lia_idx_tri_row_u, lia_mv_mul, 
    lia_m_full_to_packed_sym, lia_m_packed_to_full_sym,
    lia_m_BK_sym_factorize_inplace, lia_m_BK_sym_factorize, 
    lia_m_logdet_sym_BK, lia_m_det_sym_BK, lia_m_BK_sym_query_lwork,
    lia_m_inv_sym_BK_inplace, lia_m_inv_sym_BK, lia_m_inv_sym_BK_bf_inplace, lia_m_inv_sym_BK_bf, 
    lia_m_cholesky_sym_inplace, lia_m_cholesky_sym_norm_inplace, lia_m_cholesky_sym, lia_m_cholesky_sym_norm,
    lia_m_logsqrtdet_sym_cholesky, lia_m_sqrtdet_sym_cholesky, lia_m_det_sym_cholesky,
    lia_m_inv_sym_cholesky_inplace, lia_m_inv_sym_cholesky,
    lia_sv_scalar,
    lia_vv_add_inplace, lia_vv_sub_inplace,
    lia_vv_cpy,
    lia_vv_mul,
    lia_vv_mul_A_i0_inplace, lia_vv_div_A_i0_inplace,
    lia_vmv_quadform_sym_bf, lia_vmv_quadform_sym, 
    lia_vmv_quadform_sym_A_i0i_bf, lia_vmv_quadform_sym_A_i0i, 
    lia_vmv_quadform_sym_A_iii_bf, lia_vmv_quadform_sym_A_iii, 
    lia_vv_dot,
    # Low Level interface
    LIA_ORDER, LIA_TRANS, LIA_UPLO, LIA_DIAG, LIA_SIDE, 
    lia_0_dsymm, lia_0_dgemm, 
)

# cdef int lia_m_BK_sym_factorize_inplace(double * full, int * ipiv, long ndim, long lda=*, bint upper=*) noexcept nogil
# cdef int lia_m_BK_sym_factorize(double * out, int * ipiv, double * full, long ndim, long lda=*, bint upper=*) noexcept nogil
# cdef double lia_m_logdet_sym_BK(double * a, int * ipiv, long ndim, long lda=*) noexcept nogil
# cdef double lia_m_det_sym_BK(double * a, int * ipiv, long ndim, long lda=*) noexcept nogil
# cdef int lia_m_inv_sym_BK_inplace(double * full, int * ipiv, long ndim, long lda=*, bint upper=*) noexcept nogil
# cdef int lia_m_inv_sym_BK(double * out, int * ipiv, double * full, long ndim, long lda=*, bint upper=*) noexcept nogil
# cdef long lia_m_BK_sym_query_lwork(double * full, int * ipiv, long ndim, long lda=*, bint upper=*) noexcept nogil
# cdef int lia_m_inv_sym_BK_bf_inplace(double * full, int * ipiv, double * work, int lwork, long ndim, long lda=*, bint upper=*) noexcept nogil
# cdef int lia_m_inv_sym_BK_bf(double * out, int * ipiv, double * full, double * work, int lwork, long ndim, long lda=*, bint upper=*) noexcept nogil


from .variability cimport (
    VarType, par_var_LOG,
    par_var_DRW,      var_DRW,      var_DRW_oneside,      set_par_var_DRW,      get_invtau_var_DRW,
          var_LOG_DRW,      var_LOG_DRW_oneside,      set_par_var_LOG_DRW,      get_invtau_var_LOG_DRW,
    par_var_DRW_fast, var_DRW_fast, var_DRW_fast_oneside, set_par_var_DRW_fast, get_invtau_var_DRW_fast,
          var_LOG_DRW_fast, var_LOG_DRW_fast_oneside, set_par_var_LOG_DRW_fast, get_invtau_var_LOG_DRW_fast,
    # par_var_CARMA,
)

cdef double twoPI = 2*M_PI

from .lightcurves cimport LightCurves, packet_LC

from libc.stdio cimport (
    printf, 
    FILE, fopen, fwrite, fclose, fflush,
    fprintf,
)

from .prhq cimport packet_PRHQ

ctypedef void (*func_prhq_piecewise)(packet_PRHQ *) noexcept nogil
ctypedef struct bundle_piecewise:
    # Global buffer
    long nijsym   # = ndata*(ndata+1)//2
    long npqsym   # = nresps*(nresps+1)//2
    double * t_cf
    # F and U are only for DR; if not has_driving, don't bother allocating mem.
    double * F_p_L
    double * F_p_R
    double * U_p_L
    double * U_p_R
    # G and V are for RR; should be always allocated.
    double * G_pq_L
    double * G_pq_R
    double * V_pq_L
    double * V_pq_R
    func_t_double_par acf1
    # Temporary number (should be removed?)
    # DR
    long k_l
    long k_r
    long n_v
    double tau_l
    double tau_r
    # RR
    long n_cf
    long n_ccf
    double t_cf_max
    # Computing Functions
    func_prhq_piecewise prep
    func_prhq_piecewise compute_covmat

cdef double compute_ccf_tf_l(double * f, double * g, long k, long ntotal, long nvalid) noexcept nogil:
    # use when k<nvalid-1
    # Replace with Kahan_Sum function
    cdef long i
    cdef double s, c
    s = 0
    c = 0
    for i in range(ntotal-nvalid+1+k):
        kahan_sum_iterator(&s, f[i - (k - nvalid + 1)]*g[i], &c)
    return s

cdef double compute_ccf_tf_r(double * f, double * g, long k, long ntotal, long nvalid) noexcept nogil:
    # use when k>=nvalid-1
    # Replace with Kahan_Sum function
    cdef long i
    cdef double s, c
    s = 0
    c = 0
    for i in range(ntotal+nvalid-1+k):
        kahan_sum_iterator(&s, f[i]*g[i + (k - nvalid + 1)], &c)
    return s

cdef void covmat_ij_piecewise_prep_DRW_has_driv(packet_PRHQ * packet) noexcept nogil:
    cdef bundle_piecewise * bundle = (<bundle_piecewise *>packet.bundle)
    # called once every loglike call
    cdef long p, q, k, pq, PQ, P, Q, i
    cdef long k_l, k_r, n_v, n_cf
    cdef double tau_l, tau_r
    cdef double * tf_p
    cdef double * tf_q
    cdef double s, c
    
    k_l = packet.ntau-1
    for k in range(packet.ntau):
        for p in range(packet.nresp):
            if packet.tf[p*packet.ntau + k]>0:
                k_l = k
                break
        if k_l == k:
            if k_l>0:
                k_l = k_l - 1 # Move one back because of backward finite difference; 1st is always 0.
            break
    k_r = k_l
    for k in range(packet.ntau-1, k_l, -1):
        for p in range(packet.nresp):
            if packet.tf[p*packet.ntau + k]>0:
                k_r = k
                break
        if k_r == k: # no need for correction.
            break
    tau_l = packet.tau[k_l]
    tau_r = packet.tau[k_r]
    n_v = k_r - k_l + 1
    n_cf = 2*n_v - 1

    bundle.k_l = k_l
    bundle.k_r = k_r
    bundle.tau_l = tau_l
    bundle.tau_r = tau_r
    bundle.n_v = n_v
    bundle.n_cf = n_cf
    bundle.t_cf_max = tau_r - tau_l
    
    for k in range(n_cf):
        bundle.t_cf[k] = (k - n_v + 1)*packet.dtau
    for p in range(packet.nresp):
        P = p*packet.ntau 
        tf_p = packet.tf + P
        for k in range(k_l, k_r+1):
            bundle.F_p_L[P+k] = packet.dtau * tf_p[k] * bundle.acf1(-(packet.tau[k]+tau_l), packet.par_acf)
            bundle.F_p_R[P+k] = packet.dtau * tf_p[k] * bundle.acf1(+(packet.tau[k]+tau_r), packet.par_acf)
        for q in range(p, packet.nresp):
            Q = q*packet.ntau
            tf_q = packet.tf + Q
            pq = lia_idx_tri_row_u(p, q, packet.nresp)
            PQ = pq*bundle.n_ccf
            for k in range(n_v-1):
                s = 0
                c = 0
                for i in range(packet.ntau + k + 1 - n_v):
                    kahan_sum_iterator(&s, tf_p[i-(k - n_v + 1)]*tf_q[i], &c)
                bundle.G_pq_L[PQ+k] = packet.dtau * packet.dtau * s * bundle.acf1(-(bundle.t_cf[k]-bundle.t_cf_max), packet.par_acf)
                bundle.G_pq_R[PQ+k] = packet.dtau * packet.dtau * s * bundle.acf1(+(bundle.t_cf[k]+bundle.t_cf_max), packet.par_acf)
            for k in range(n_v-1, n_cf):
                s = 0
                c = 0
                for i in range(packet.ntau + n_v - 1 - k):
                    kahan_sum_iterator(&s, tf_p[i]*tf_q[i+(k - n_v + 1)], &c)
                bundle.G_pq_L[PQ+k] = packet.dtau * packet.dtau * s * bundle.acf1(-(bundle.t_cf[k]-bundle.t_cf_max), packet.par_acf)
                bundle.G_pq_R[PQ+k] = packet.dtau * packet.dtau * s * bundle.acf1(+(bundle.t_cf[k]+bundle.t_cf_max), packet.par_acf)
    for p in range(packet.nresp):
        P = p*packet.ntau 
        s = 0
        c = 0
        bundle.U_p_L[p] = 0
        bundle.U_p_R[p] = 0
        for k in range(k_l, k_r+1):
            # k - k_l = k_r - i # k = k_l, i = k_r, k = k_r, i = k_l
            i = k_r + k_l - k
            # Technically it is possible to use the next F_p for the destination and remove U_p,
            #   but it is dangerous and cannot guarantee if Kahan summation would work in that case.
            #   also, it should not take too much space.
            # It is also possible to rewrite the Kahan summation in cumulative sum form,
            #   but it is a bit of too wasteful.
            kahan_sum_iterator(bundle.U_p_L + p, bundle.F_p_L[P+k], &s)
            kahan_sum_iterator(bundle.U_p_R + p, bundle.F_p_R[P+i], &c)
            bundle.F_p_L[P+k] = bundle.U_p_L[p]
            bundle.F_p_R[P+i] = bundle.U_p_R[p]
        for q in range(p, packet.nresp):
            Q = q*packet.ntau
            pq = lia_idx_tri_row_u(p, q, packet.nresp)
            PQ = pq*bundle.n_ccf
            s = 0
            c = 0
            bundle.V_pq_L[pq] = 0
            bundle.V_pq_R[pq] = 0
            for k in range(n_cf):
                # k - 0 = n_cf-1 - i # k = 0, i = n_cf-1, k = n_cf-1, i = 0
                i = n_cf - 1 - k
                kahan_sum_iterator(bundle.V_pq_L + pq, bundle.G_pq_L[PQ+k], &s)
                kahan_sum_iterator(bundle.V_pq_R + pq, bundle.G_pq_R[PQ+i], &c)
                bundle.G_pq_L[PQ+k] = bundle.V_pq_L[pq]
                bundle.G_pq_R[PQ+i] = bundle.V_pq_R[pq]

cdef void covmat_ij_piecewise_prep_DRW_no_driv(packet_PRHQ * packet) noexcept nogil:
    cdef bundle_piecewise * bundle = (<bundle_piecewise *>packet.bundle)
    # called once every loglike call
    cdef long p, q, k, pq, PQ, i
    cdef long k_l, k_r, n_v, n_cf
    cdef double tau_l, tau_r
    cdef double * tf_p
    cdef double * tf_q
    cdef double s, c
    
    k_l = packet.ntau-1
    for k in range(packet.ntau):
        for p in range(packet.nresp):
            if packet.tf[p*packet.ntau + k]>0:
                k_l = k
                break
        if k_l == k:
            if k_l>0:
                k_l = k_l - 1 # Move one back because of backward finite difference; 1st is always 0.
            break
    k_r = k_l
    for k in range(packet.ntau-1, k_l, -1):
        for p in range(packet.nresp):
            if packet.tf[p*packet.ntau + k]>0:
                k_r = k
                break
        if k_r == k: # no need for correction.
            break
    
    tau_l = packet.tau[k_l]
    tau_r = packet.tau[k_r]
    n_v = k_r - k_l + 1
    n_cf = 2*n_v - 1

    bundle.k_l = k_l
    bundle.k_r = k_r
    bundle.tau_l = tau_l
    bundle.tau_r = tau_r
    bundle.n_v = n_v
    bundle.n_cf = n_cf
    bundle.t_cf_max = tau_r - tau_l
    
    for k in range(n_cf):
        bundle.t_cf[k] = (k - n_v + 1)*packet.dtau
    for p in range(packet.nresp):
        tf_p = packet.tf + p*packet.ntau 
        for q in range(p, packet.nresp):
            tf_q = packet.tf + q*packet.ntau
            pq = lia_idx_tri_row_u(p, q, packet.nresp)
            PQ = pq*bundle.n_ccf
            for k in range(n_v-1):
                s = 0
                c = 0
                for i in range(packet.ntau + k + 1 - n_v):
                    kahan_sum_iterator(&s, tf_p[i-(k - n_v + 1)]*tf_q[i], &c)
                bundle.G_pq_L[PQ+k] = packet.dtau * packet.dtau * s * bundle.acf1(-(bundle.t_cf[k]-bundle.t_cf_max), packet.par_acf)
                bundle.G_pq_R[PQ+k] = packet.dtau * packet.dtau * s * bundle.acf1(+(bundle.t_cf[k]+bundle.t_cf_max), packet.par_acf)
            for k in range(n_v-1, n_cf):
                s = 0
                c = 0
                for i in range(packet.ntau + n_v - 1 - k):
                    kahan_sum_iterator(&s, tf_p[i]*tf_q[i+(k - n_v + 1)], &c)
                bundle.G_pq_L[PQ+k] = packet.dtau * packet.dtau * s * bundle.acf1(-(bundle.t_cf[k]-bundle.t_cf_max), packet.par_acf)
                bundle.G_pq_R[PQ+k] = packet.dtau * packet.dtau * s * bundle.acf1(+(bundle.t_cf[k]+bundle.t_cf_max), packet.par_acf)
    for p in range(packet.nresp):
        for q in range(p, packet.nresp):
            pq = lia_idx_tri_row_u(p, q, packet.nresp)
            PQ = pq*bundle.n_ccf
            s = 0
            c = 0
            bundle.V_pq_L[pq] = 0
            bundle.V_pq_R[pq] = 0
            for k in range(n_cf):
                # k - 0 = n_cf-1 - i # k = 0, i = n_cf-1, k = n_cf-1, i = 0
                i = n_cf - 1 - k
                kahan_sum_iterator(bundle.V_pq_L + pq, bundle.G_pq_L[PQ+k], &s)
                kahan_sum_iterator(bundle.V_pq_R + pq, bundle.G_pq_R[PQ+i], &c)
                bundle.G_pq_L[PQ+k] = bundle.V_pq_L[pq]
                bundle.G_pq_R[PQ+i] = bundle.V_pq_R[pq]


cdef double covmat_ij_dd(double dt_ij, func_t_double_par acf, void * acf_params) noexcept nogil:
    # Covmat_ij between two epochs of a driving light curve.
    # e.g., Eq. 2 of Rybicki & Kleyna 1994 ASPConf., 69, 85
    # e.g., Eq. 4 of Zu et al. 2011 ApJ, 735, 80
    return acf(dt_ij, acf_params)

# Computing Covmat_ij via Piecewise Integrations
cdef double piecewise_l( # use when -dt < bound_l
    double dt_ij, double bound_l, double bound_r, double invtau_d,
    double sum_L, double sum_R,
) noexcept nogil: # invtau_d = 1/tau_d
    return sum_R*exp((bound_r - dt_ij)*invtau_d)

cdef double piecewise_r( # use when -dt > bound_r
    double dt_ij, double bound_l, double bound_r, double invtau_d,
    double sum_L, double sum_R,
) noexcept nogil: # invtau_d = 1/tau_d
    return sum_L*exp((dt_ij - bound_l)*invtau_d)

cdef double piecewise_c(
    double dt_ij, double bound_l, double bound_r, double invtau_d,
    double * arr_L, double * arr_R, # arr_L = bundle.F_p_L + k_l, arr_R = bundle.F_p_R + k_l
    double sum_L, double sum_R,
    double * tau,                   # tau = packet.tau + k_l
    long n,                         # n = k_r - k_l + 1 = bundle.n_v
) noexcept nogil: # invtau_d = 1/tau_d
    cdef long i, m
    cdef double c, s_l, s_r
    m = binary_search_ptr_R(-dt_ij, tau, n)
    return arr_L[m-1]*exp((dt_ij - bound_l)*invtau_d) + arr_R[m]*exp((bound_r - dt_ij)*invtau_d)

cdef void compute_covmat_has_driv(packet_PRHQ * p) noexcept nogil:
    # printf("YES DRIV!\n")
    # C is not packed.
    cdef bundle_piecewise * b = (<bundle_piecewise *>p.bundle)
    cdef long i, j, ij, j0, j1, pq, PQ, plc, qlc, I
    cdef long j_l, j_r
    cdef double * dtij
    cdef double * tau = p.tau + b.k_l
    cdef double invtau = p.get_invtau(p.par_acf)
    for i in range(p.lc.offset[0], p.lc.offset[0]+p.lc.npts[0]): # For all i in Driving LCs
        I = i*p.ndata
        for j in range(i, p.lc.offset[0]+p.lc.npts[0]): # 
            p.Cinv[I + j] = covmat_ij_dd(p.dt_ij[lia_idx_tri_row_u(i, j, p.ndata)], p.acf, p.par_acf)
            # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
        for plc in range(1, p.nlc): # R
            j0   = p.lc.offset[plc]
            j1   = p.lc.offset[plc] + p.lc.npts[plc]
            dtij = p.dt_ij + lia_idx_tri_row_u(i, j0, p.ndata)
            
            # Note that dt_ij is decreasing for j, and should be compared with NEGATIVE tau.
            #   thus, following relation would satisfy.
            #   -dt_ij[j0] <?= tau_l < -dt_ij[j1] <?= tau_r
            # use binary search twice between (i, nlc[plc]) to find
            #     j_l <= j_r
            #   corresponding to 
            #     -dt_ij[i, j_l-1] <= tau_l < -dt_ij[i, j_l]
            #   and
            #     -dt_ij[i, j_r-1] < tau_r <= -dt_ij[i, j_r]
            # OR
            #     dt_ij[i, j_l-1] >= -tau_l > dt_ij[i, j_l] # L_desc
            #   and
            #     dt_ij[i, j_r-1] > -tau_r >= dt_ij[i, j_r] # R_desc
            j_l = binary_search_ptr_L_desc(-b.tau_l, dtij, j1 - j0) + j0
            j_r = binary_search_ptr_R_desc(-b.tau_r, dtij, j1 - j0) + j0
            #   then, run (j0, j_l) for U_L, (j_l, j_r) for F_L+F_R, and (j_r, nj) for U_R.
            for j in range(j0, j_l):
                p.Cinv[I + j] = piecewise_l(
                    dtij[j-j0], b.tau_l, b.tau_r, invtau, b.U_p_L[plc-1], b.U_p_R[plc-1], # plc-1 because driving
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
            for j in range(j_l, j_r):
                # arr_L = bundle.F_p_L + k_l, arr_R = bundle.F_p_R + k_l
                p.Cinv[I + j] = piecewise_c(
                    dtij[j-j0], b.tau_l, b.tau_r, invtau, 
                    b.F_p_L + (plc-1)*p.ntau + b.k_l, b.F_p_R + (plc-1)*p.ntau + b.k_l, 
                    b.U_p_L[plc-1], b.U_p_R[plc-1], tau, b.n_v,
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
            for j in range(j_r, j1):
                p.Cinv[I + j] = piecewise_r(
                    dtij[j-j0], b.tau_l, b.tau_r, invtau, b.U_p_L[plc-1], b.U_p_R[plc-1], # plc-1 because driving
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
    for plc in range(1, p.nlc): # RR
        for i in range(p.lc.offset[plc], p.lc.offset[plc]+p.lc.npts[plc]): # Auto
            I = i*p.ndata
            pq = lia_idx_tri_row_u(plc-1, plc-1, p.nresp)
            PQ = pq*b.n_ccf
            j0   = i
            j1   = p.lc.offset[plc] + p.lc.npts[plc]
            dtij = p.dt_ij + lia_idx_tri_row_u(i, j0, p.ndata)
            j_l = binary_search_ptr_L_desc(+b.t_cf_max, dtij, j1 - j0) + j0
            j_r = binary_search_ptr_R_desc(-b.t_cf_max, dtij, j1 - j0) + j0
            for j in range(j0, j_l):
                p.Cinv[I + j] = piecewise_l(
                    dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, b.V_pq_L[pq], b.V_pq_R[pq], # plc-1 because driving
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
            for j in range(j_l, j_r):
                # arr_L = bundle.F_p_L + k_l, arr_R = bundle.F_p_R + k_l
                p.Cinv[I + j] = piecewise_c(
                    dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, 
                    b.G_pq_L + PQ, b.G_pq_R + PQ, 
                    b.V_pq_L[pq], b.V_pq_R[pq], b.t_cf, b.n_cf,
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
            for j in range(j_r, j1):
                p.Cinv[I + j] = piecewise_r(
                    dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, b.V_pq_L[pq], b.V_pq_R[pq], # plc-1 because driving
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
            for qlc in range(plc+1, p.nlc): # Cross
                pq = lia_idx_tri_row_u(plc-1, qlc-1, p.nresp)
                PQ = pq*b.n_ccf
                j0   = p.lc.offset[qlc]
                j1   = p.lc.offset[qlc] + p.lc.npts[qlc]
                dtij = p.dt_ij + lia_idx_tri_row_u(i, j0, p.ndata)
                j_l = binary_search_ptr_L_desc(+b.t_cf_max, dtij, j1 - j0) + j0
                j_r = binary_search_ptr_R_desc(-b.t_cf_max, dtij, j1 - j0) + j0
                for j in range(j0, j_l):
                    p.Cinv[I + j] = piecewise_l(
                        dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, b.V_pq_L[pq], b.V_pq_R[pq], # plc-1 because driving
                    )
                    # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
                for j in range(j_l, j_r):
                    # arr_L = bundle.F_p_L + k_l, arr_R = bundle.F_p_R + k_l
                    p.Cinv[I + j] = piecewise_c(
                        dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, 
                        b.G_pq_L + PQ, b.G_pq_R + PQ, 
                        b.V_pq_L[pq], b.V_pq_R[pq], b.t_cf, b.n_cf,
                    )
                    # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
                for j in range(j_r, j1):
                    p.Cinv[I + j] = piecewise_r(
                        dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, b.V_pq_L[pq], b.V_pq_R[pq], # plc-1 because driving
                    )
                    # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]

cdef void compute_covmat_no_driv(packet_PRHQ * p) noexcept nogil:
    # printf("NO DRIV!\n")
    # C is not packed.
    cdef bundle_piecewise * b = (<bundle_piecewise *>p.bundle)
    cdef long i, j, ij, j0, j1, pq, PQ, plc, qlc, I, j_l, j_r
    cdef double * dtij
    cdef double * tau = p.tau + b.k_l
    cdef double invtau = p.get_invtau(p.par_acf)
    for plc in range(p.nlc): # RR
        for i in range(p.lc.offset[plc], p.lc.offset[plc]+p.lc.npts[plc]):
            I = i*p.ndata
            pq = lia_idx_tri_row_u(plc, plc, p.nresp)
            PQ = pq*b.n_ccf
            j0   = i
            j1   = p.lc.offset[plc] + p.lc.npts[plc]
            dtij = p.dt_ij + lia_idx_tri_row_u(i, j0, p.ndata)
            j_l = binary_search_ptr_L_desc(+b.t_cf_max, dtij, j1 - j0) + j0
            j_r = binary_search_ptr_R_desc(-b.t_cf_max, dtij, j1 - j0) + j0
            for j in range(j0, j_l):
                p.Cinv[I + j] = piecewise_l(
                    dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, b.V_pq_L[pq], b.V_pq_R[pq],
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
            for j in range(j_l, j_r):
                p.Cinv[I + j] = piecewise_c(
                    dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, 
                    b.G_pq_L + PQ, b.G_pq_R + PQ, 
                    b.V_pq_L[pq], b.V_pq_R[pq], b.t_cf, b.n_cf,
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
            for j in range(j_r, j1):
                p.Cinv[I + j] = piecewise_r(
                    dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, b.V_pq_L[pq], b.V_pq_R[pq],
                )
                # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
            for qlc in range(plc+1, p.nlc): # Cross
                pq = lia_idx_tri_row_u(plc, qlc, p.nresp)
                PQ = pq*b.n_ccf
                j0   = p.lc.offset[qlc]
                j1   = p.lc.offset[qlc] + p.lc.npts[qlc]
                dtij = p.dt_ij + lia_idx_tri_row_u(i, j0, p.ndata)
                j_l = binary_search_ptr_L_desc(+b.t_cf_max, dtij, j1 - j0) + j0
                j_r = binary_search_ptr_R_desc(-b.t_cf_max, dtij, j1 - j0) + j0
                for j in range(j0, j_l):
                    p.Cinv[I + j] = piecewise_l(
                        dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, b.V_pq_L[pq], b.V_pq_R[pq],
                    )
                    # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
                for j in range(j_l, j_r):
                    p.Cinv[I + j] = piecewise_c(
                        dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, 
                        b.G_pq_L + PQ, b.G_pq_R + PQ, 
                        b.V_pq_L[pq], b.V_pq_R[pq], b.t_cf, b.n_cf,
                    )
                    # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
                for j in range(j_r, j1):
                    p.Cinv[I + j] = piecewise_r(
                        dtij[j-j0], -b.t_cf_max, +b.t_cf_max, invtau, b.V_pq_L[pq], b.V_pq_R[pq],
                    )
                    # p.Cinv[j*p.ndata + i] = p.Cinv[I + j]
        
cdef int initialize_PRHQ_piecewise(packet_PRHQ * p) noexcept nogil:
    cdef:
        long n_ccf = 2*p.ntau - 1
        long nijsym = p.ndata*(p.ndata+1)//2
        long npqsym = p.nresp*(p.nresp+1)//2
    # Method-specific buffers
    p.bundle = <void *> malloc(sizeof(bundle_piecewise))
    (<bundle_piecewise *>p.bundle).n_ccf = n_ccf
    (<bundle_piecewise *>p.bundle).nijsym = nijsym
    (<bundle_piecewise *>p.bundle).npqsym = npqsym
    (<bundle_piecewise *>p.bundle).t_cf   = <double *> malloc(n_ccf * sizeof(double))
    if p.lc.has_driving:
        (<bundle_piecewise *>p.bundle).F_p_L = <double *> malloc((p.nresp * p.ntau) * sizeof(double))
        (<bundle_piecewise *>p.bundle).F_p_R = <double *> malloc((p.nresp * p.ntau) * sizeof(double))
        (<bundle_piecewise *>p.bundle).U_p_L = <double *> malloc((p.nresp         ) * sizeof(double))
        (<bundle_piecewise *>p.bundle).U_p_R = <double *> malloc((p.nresp         ) * sizeof(double))
        (<bundle_piecewise *>p.bundle).prep           = covmat_ij_piecewise_prep_DRW_has_driv
        (<bundle_piecewise *>p.bundle).compute_covmat = compute_covmat_has_driv
    else:
        (<bundle_piecewise *>p.bundle).F_p_L = NULL
        (<bundle_piecewise *>p.bundle).F_p_R = NULL
        (<bundle_piecewise *>p.bundle).U_p_L = NULL
        (<bundle_piecewise *>p.bundle).U_p_R = NULL
        (<bundle_piecewise *>p.bundle).prep           = covmat_ij_piecewise_prep_DRW_no_driv
        (<bundle_piecewise *>p.bundle).compute_covmat = compute_covmat_no_driv
    (<bundle_piecewise *>p.bundle).G_pq_L = <double *> malloc((npqsym * n_ccf) * sizeof(double))
    (<bundle_piecewise *>p.bundle).G_pq_R = <double *> malloc((npqsym * n_ccf) * sizeof(double))
    (<bundle_piecewise *>p.bundle).V_pq_L = <double *> malloc((npqsym        ) * sizeof(double))
    (<bundle_piecewise *>p.bundle).V_pq_R = <double *> malloc((npqsym        ) * sizeof(double))
    if p.var == VarType.DRW:
        (<bundle_piecewise *>p.bundle).acf1 = var_DRW_oneside
    elif p.var == VarType.DRW_FAST:
        (<bundle_piecewise *>p.bundle).acf1 = var_DRW_fast_oneside
    elif p.var == VarType.LOG_DRW:
        (<bundle_piecewise *>p.bundle).acf1 = var_LOG_DRW_oneside
    elif p.var == VarType.LOG_DRW_FAST:
        (<bundle_piecewise *>p.bundle).acf1 = var_LOG_DRW_fast_oneside

    if (
        (<bundle_piecewise *>p.bundle).t_cf == NULL 
        or (<bundle_piecewise *>p.bundle).G_pq_L == NULL or (<bundle_piecewise *>p.bundle).G_pq_R == NULL
        or (<bundle_piecewise *>p.bundle).V_pq_L == NULL or (<bundle_piecewise *>p.bundle).V_pq_R == NULL
    ):
        return -1
    if p.lc.has_driving and (
           (<bundle_piecewise *>p.bundle).F_p_L == NULL or (<bundle_piecewise *>p.bundle).F_p_R == NULL
        or (<bundle_piecewise *>p.bundle).U_p_L == NULL or (<bundle_piecewise *>p.bundle).U_p_R == NULL
    ):
        return -1
    return 0

cdef void finalize_PRHQ_piecewise(packet_PRHQ * p) noexcept nogil:
    cdef bundle_piecewise * b = <bundle_piecewise *> p.bundle
    if b != NULL:
        if b.t_cf != NULL:
            free(b.t_cf)
        if p.lc.has_driving:
            if b.F_p_L != NULL:
                free(b.F_p_L)
            if b.F_p_R != NULL:
                free(b.F_p_R)
            if b.U_p_L != NULL:
                free(b.U_p_L)
            if b.U_p_R != NULL:
                free(b.U_p_R)
        if b.G_pq_L != NULL:
            free(b.G_pq_L)
        if b.G_pq_R != NULL:
            free(b.G_pq_R)
        if b.V_pq_L != NULL:
            free(b.V_pq_L)
        if b.V_pq_R != NULL:
            free(b.V_pq_R)
        b.acf1 = NULL
        free(p.bundle)

cdef void covmat_piecewise(void * packet) noexcept nogil:
    cdef packet_PRHQ * p = <packet_PRHQ *> packet
    cdef bundle_piecewise * b = <bundle_piecewise *> p.bundle
    # # dt_ij should be precomputed.
    # Prepare the Fourier domain variables
    b.prep(p)
    # Assign the covariance matrices. 
    b.compute_covmat(p)

cdef int initialize_PRHQ(
    packet_PRHQ * p, packet_LC * lc, VarType vartype, long * ndet_orders,
    size_t ntau, double * tau, double * tf, 
) noexcept nogil:
    cdef long i, j, k, nijsym, nfreq, npqsym, nfft, I, n_ccf
    p.lc    = lc
    p.nlc   = p.lc.NLCs
    p.ndata = p.lc.n
    p.y     = p.lc.y
    nijsym  = p.ndata*(p.ndata+1)//2
    p.N     = <double *> calloc(p.ndata*p.ndata, sizeof(double))
    if p.lc.has_cov:
        lia_vv_cpy(p.N, p.lc.cov, p.ndata*p.ndata)
    elif p.lc.has_corr:
        for i in range(p.ndata):
            I = i*p.ndata
            for j in range(p.ndata):
                p.N[I+j] = p.lc.e[i]*p.lc.e[j]*p.lc.corr[I+j]
    else:
        for i in range(p.ndata):
            p.N[i*(1+p.ndata)] = p.lc.e[i]*p.lc.e[i]
    p.dt_ij = <double *> malloc(nijsym*sizeof(double))
    for i in range(p.ndata):
        for j in range(i, p.ndata):
            # for j, within same i, dt_ij is decreasing.
            p.dt_ij[lia_idx_tri_row_u(i, j, p.ndata)] = p.lc.t[i] - p.lc.t[j]
    p.var   = vartype
    # Detrending Parameters
    # Maybe redesign memory management here?
    p.n_det_order = <long *> malloc(p.nlc*sizeof(long))
    memcpy(<void *>p.n_det_order, <void *>ndet_orders, p.nlc*sizeof(long))
    p.n_det_pars = 0
    for i in range(p.nlc):
        p.n_det_pars += p.n_det_order[i] + 1
    p.L     = <double *> calloc(p.ndata*p.n_det_pars, sizeof(double))
    cdef long * offset = <long *> malloc(p.nlc*sizeof(long))
    offset[0] = 0
    for i in range(1, p.nlc):
        # offset[i] = offset[i-1] + p.lc.npts[i-1]
        offset[i] = offset[i-1] + (p.n_det_order[i-1] + 1)
    for i in range(p.nlc):
        for j in range(p.lc.npts[i]):
            for k in range(p.n_det_order[i]+1):
                p.L[p.n_det_pars*(p.lc.offset[i]+j)+offset[i]+k] = 1 # need to fix.
    free(offset)
    p.q     = <double *> calloc(p.n_det_pars, sizeof(double))
    # Transfer Functions
    p.ntau  = ntau
    p.nresp = p.lc.NResps
    p.tau   = tau # This should be managed by the parent class.
    p.tf    = tf  # This should be managed by the parent class.
    # printf("%p %p\n", tf, p.tf)
    p.dtau  = p.tau[1] - p.tau[0]
    # Buffers for PRH-Q
    p.buffer_mat1 = <double *> calloc(p.ndata*p.ndata, sizeof(double))
    p.buffer_mat2 = <double *> calloc(p.ndata*p.ndata, sizeof(double))
    p.buffer      = <double *> calloc(p.ndata, sizeof(double))
    p.Cinv        = <double *> calloc(p.ndata*p.ndata, sizeof(double))
    p.Cq          = <double *> calloc(p.n_det_pars*p.n_det_pars, sizeof(double))
    p.ipiv = <int *> malloc(p.ndata*sizeof(int))
    p.lwork = lia_m_BK_sym_query_lwork(p.Cinv, p.ipiv, p.ndata)
    p.work = <double *> malloc(p.lwork*sizeof(double))

    # cdef func_t_double_par oneside_func
    if p.var == VarType.DRW:
        p.acf = var_DRW
        p.par_acf = <void *>malloc(sizeof(par_var_DRW))
        p.set_var_par = set_par_var_DRW
        p.get_invtau  = get_invtau_var_DRW
    elif p.var == VarType.DRW_FAST:
        p.acf = var_DRW_fast
        p.par_acf = <void *>malloc(sizeof(par_var_DRW_fast))
        p.set_var_par = set_par_var_DRW_fast
        p.get_invtau  = get_invtau_var_DRW_fast
    elif p.var == VarType.LOG_DRW:
        p.acf = var_LOG_DRW
        p.par_acf = <void *>malloc(sizeof(par_var_LOG))
        (<par_var_LOG *>(p.par_acf)).par_var = <void *>malloc(sizeof(par_var_DRW))
        p.set_var_par = set_par_var_LOG_DRW
        p.get_invtau  = get_invtau_var_LOG_DRW
    elif p.var == VarType.LOG_DRW_FAST:
        p.acf = var_LOG_DRW_fast
        p.par_acf = <void *>malloc(sizeof(par_var_LOG))
        (<par_var_LOG *>(p.par_acf)).par_var = <void *>malloc(sizeof(par_var_DRW_fast))
        p.set_var_par = set_par_var_LOG_DRW_fast
        p.get_invtau  = get_invtau_var_LOG_DRW_fast
    
    cdef int ret_method=0
    p.covmat = covmat_piecewise
    ret_method = initialize_PRHQ_piecewise(p)
    if ret_method !=0:
        return ret_method
    if (
        p.N == NULL or p.dt_ij == NULL or p.par_acf == NULL or p.n_det_order == NULL 
        or p.L == NULL or p.q == NULL or p.buffer_mat1 == NULL or p.buffer_mat2 == NULL or p.Cinv == NULL
        or p.Cq == NULL or p.bundle == NULL or p.ipiv == NULL or p.work == NULL
        or p.buffer == NULL
    ):
        return -1
    return 0

cdef void finalize_PRHQ(packet_PRHQ * p) noexcept nogil:
    finalize_PRHQ_piecewise(p)
    free(p.ipiv)
    free(p.work)
    free(p.Cq)
    free(p.Cinv)
    free(p.buffer_mat1)
    free(p.buffer_mat2)
    free(p.buffer)
    p.tf = NULL
    p.tau = NULL
    free(p.q)
    free(p.L)
    free(p.n_det_order)
    if p.var==VarType.LOG_DRW or p.var==VarType.LOG_DRW_FAST:
        free((<par_var_LOG *>(p.par_acf)).par_var)
    free(p.par_acf)
    p.acf = NULL
    p.set_var_par = NULL
    p.get_invtau = NULL
    free(p.dt_ij)
    free(p.N)
    p.y = NULL
    p.lc = NULL
    p.covmat = NULL
    free(p)

cdef double loglike_PRHQ_detr_marginalized(packet_PRHQ * p) noexcept nogil:
    cdef double loglike
    cdef long i, j
    cdef int sign
    p.covmat(<void *>p)
    # --- p.Cinv is constructed. ---
    # p.N should be constant, thus, is prepared in the initialization.
    # 1.  Cinv <- lia_vv_sum(S,N) 
    lia_vv_add_inplace(p.Cinv, p.N, p.ndata*p.ndata)
    # 2.1. Cinv <- Cholesky_U(Cinv) # contains S+N, outputs cholesky decomp. of S+N
    # 3.  cdef double loglike = log(sqrtdet(C-1)) = -log(sqrtdet(Cinv)) = -log(sqrtdet(S+N))
    # 2.2.  Cinv <- inv(Cinv)

    ### Cholesky Implementation
    lia_m_cholesky_sym_inplace(p.Cinv, p.ndata)
    loglike = -lia_m_logsqrtdet_sym_cholesky(p.Cinv, p.ndata) # 0.5 is multiplied already.
    if not (loglike > -DBL_MAX):
        return -INFINITY
    lia_m_inv_sym_cholesky_inplace(p.Cinv, p.ndata)
    
    ### BK Implementation
    # lia_m_BK_sym_factorize_inplace(p.Cinv, p.ipiv, p.ndata)
    # loglike = -0.5*lia_m_logdet_sym_BK(p.Cinv, p.ipiv, p.ndata, &sign)
    # # loglike = lia_m_logdet_sym_BK(p.Cinv, p.ipiv, p.ndata, &sign)
    # if sign<0:
    #     return -INFINITY    
    # lia_m_inv_sym_BK_bf_inplace(p.Cinv, p.ipiv, p.work, p.lwork, p.ndata)
    
    ### C-1 computed.
    # 4.  buffer_mat1 <- symm(Cinv, L)  # be careful of lda and make sure it is contiguous.
    #                                   # Output should be ndata*n_det_pars matrix.
    lia_0_dsymm(LIA_ORDER.RowMajor, LIA_SIDE.Left, LIA_UPLO.Up, p.ndata, p.n_det_pars, 
                # 1.0, p.Cinv, p.ndata, p.L, p.ndata, 0.0, p.buffer_mat1, p.ndata)
                1.0, p.Cinv, p.ndata, p.L, p.n_det_pars, 0.0, p.buffer_mat1, p.n_det_pars)
    # 5.  Cq <- gemm(L^T, buffer_mat1) # Cq-1, Output should be n_det_pars*n_det_pars matrix.
    lia_0_dgemm(LIA_ORDER.RowMajor, LIA_TRANS.Trans, LIA_TRANS.NoTrans, p.n_det_pars, p.n_det_pars, p.ndata, 
                1.0, p.L, p.n_det_pars, p.buffer_mat1, p.n_det_pars, 0.0, p.Cq, p.n_det_pars)
    
    # 7.1. Cq <- BKfactorize(Cq)
    # 8.  loglike += log(sqrtdet(Cq))
    # 7.2.  Cq <- inv(Cq) # Cq, size n_det_pars*n_det_pars.

    ### Cholesky Implementation
    lia_m_cholesky_sym_inplace(p.Cq, p.n_det_pars)
    loglike += -lia_m_logsqrtdet_sym_cholesky(p.Cq, p.n_det_pars) # Since p.Cq = Cq^{-1} currently, this means +log(sqrtdet(Cq)), which can be interpreted as Jeffreys prior on q.
    if not (loglike > -DBL_MAX):
        return -INFINITY
    lia_m_inv_sym_cholesky_inplace(p.Cq, p.n_det_pars)

    ### BK Implementation
    # lia_m_BK_sym_factorize_inplace(p.Cq, p.ipiv, p.n_det_pars)
    # loglike += -0.5*lia_m_logdet_sym_BK(p.Cq, p.ipiv, p.n_det_pars, &sign)
    # # loglike += lia_m_logdet_sym_BK(p.Cq, p.ipiv, p.n_det_pars, &sign)
    # if sign <0:
    #     return -INFINITY
    # lia_m_inv_sym_BK_bf_inplace(p.Cq, p.ipiv, p.work, p.lwork, p.n_det_pars)
    
    # 9.  buffer_mat1 <- symm(L, Cq)          # n_det_pars*ndata
    lia_0_dsymm(LIA_ORDER.RowMajor, LIA_SIDE.Right, LIA_UPLO.Up, p.ndata, p.n_det_pars,  
                1.0, p.Cq, p.n_det_pars, p.L, p.n_det_pars, 0.0, p.buffer_mat1, p.n_det_pars)
    # 10. buffer_mat2 <- gemm(buffer_mat1, L^T) # dimensionality recovered to ndata*ndata
    # It is so sad to assign buffer_mat2 just for this one line, but this is the only way to make dgemm predictable.
    lia_0_dgemm(LIA_ORDER.RowMajor, LIA_TRANS.NoTrans, LIA_TRANS.Trans, p.ndata, p.ndata, p.n_det_pars, 
                1.0, p.buffer_mat1, p.n_det_pars, p.L, p.n_det_pars, 0.0, p.buffer_mat2, p.ndata)
    # 11. buffer_mat1 <- symm(Cinv, buffer_mat2, alpha=-1) # negate sign of buffer_mat1.
    # Actually, buffer_mat2 is utilized here as well!
    lia_0_dsymm(LIA_ORDER.RowMajor, LIA_SIDE.Left, LIA_UPLO.Up, p.ndata, p.ndata, 
                -1.0, p.Cinv, p.ndata, p.buffer_mat2, p.ndata, 0.0, p.buffer_mat1, p.ndata)
    # 12. +1 to the diagonal of buffer_mat1 to get (I-C-1.L.Cq.L^T)
    for i in range(p.ndata): # cblas_daxpy can do this as well, but this may be faster.
        p.buffer_mat1[(1+p.ndata)*i] += 1
    # 13. buffer_mat1 <- symm(buffer_mat1, Cinv, alpha=-0.5) # -0.5 to make Gaussian likelihood.
    # And here!
    lia_0_dsymm(LIA_ORDER.RowMajor, LIA_SIDE.Right, LIA_UPLO.Up, p.ndata, p.ndata, 
                1.0, p.Cinv, p.ndata, p.buffer_mat1, p.ndata, 0.0, p.buffer_mat2, p.ndata)
    ### Numerical accuracy of resulting matrix: 1.21e-10 abs. relative error (99.7%-ile) compared to numpy result.
    ### Note that the intermediate accuracy is not as good as the final result (e.g., I-C-1.L.Cq.L^T has ~1e-5).
    # 14. loglike += -0.5*lia_vmv_quadform_sym_bf(y, Cinv, y)
    loglike += -0.5*lia_vmv_quadform_sym_bf(p.y, p.buffer_mat2, p.y, p.ndata, p.buffer_mat1)
    if not (loglike > -DBL_MAX):
        return -INFINITY
    # loglike +=      lia_vmv_quadform_sym_bf(p.y, p.buffer_mat2, p.y, p.ndata, p.buffer_mat1)
    # 15. return loglike
    return loglike
    # return -0.5*loglike



cdef double loglike_PRHQ_detr_full(packet_PRHQ * p) noexcept nogil:
    cdef double loglike
    cdef long i, j
    cdef int sign
    p.covmat(<void *>p)
    # --- p.Cinv is constructed. ---
    # p.N should be constant, thus, is prepared in the initialization.
    # 1.  Cinv <- lia_vv_sum(S,N) 
    lia_vv_add_inplace(p.Cinv, p.N, p.ndata*p.ndata)
    # 2.1. Cinv <- Cholesky_U(Cinv) # contains S+N, outputs cholesky decomp. of S+N
    # 3.  cdef double loglike = log(sqrtdet(C-1)) = -log(sqrtdet(Cinv)) = -log(sqrtdet(S+N))
    # 2.2.  Cinv <- inv(Cinv)

    ### Cholesky Implementation
    lia_m_cholesky_sym_inplace(p.Cinv, p.ndata)
    loglike = -lia_m_logsqrtdet_sym_cholesky(p.Cinv, p.ndata) # 0.5 is multiplied already.
    if not (loglike > -DBL_MAX):
        return -INFINITY
    lia_m_inv_sym_cholesky_inplace(p.Cinv, p.ndata)
    
    ### BK Implementation
    # lia_m_BK_sym_factorize_inplace(p.Cinv, p.ipiv, p.ndata)
    # loglike = -0.5*lia_m_logdet_sym_BK(p.Cinv, p.ipiv, p.ndata, &sign)
    # # loglike = lia_m_logdet_sym_BK(p.Cinv, p.ipiv, p.ndata, &sign)
    # if sign<0:
    #     return -INFINITY    
    # lia_m_inv_sym_BK_bf_inplace(p.Cinv, p.ipiv, p.work, p.lwork, p.ndata)
    
    # Make L*q - y at p.buffer_mat1
    lia_mv_mul(p.buffer_mat1, p.L, p.q, p.ndata, p.n_det_pars) # L*q
    lia_vv_sub_inplace(p.buffer_mat1, p.y, p.ndata) # L*q - y

    # Final Likelihood
    loglike += -0.5*lia_vmv_quadform_sym_bf(p.buffer_mat1, p.Cinv, p.buffer_mat1, p.ndata, p.buffer_mat2)
    if not (loglike > -DBL_MAX):
        return -INFINITY
    # loglike +=      lia_vmv_quadform_sym_bf(p.buffer_mat1, p.Cinv, p.buffer_mat1, p.ndata, p.buffer_mat2)
    # 15. return loglike
    return loglike
    # return -0.5*loglike
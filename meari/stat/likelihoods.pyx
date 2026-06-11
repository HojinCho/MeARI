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


from libc.stdio cimport printf

from .stats cimport LikeType, Stats

### Marginalized Detrending

cdef double _loglike_chromatic(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp, 
    # Geometry
    double R_out, double R_in, double cos_i, double c_f, double curv,
    # Astrophysics 
    double pindex, 
    # DRW Characteristic Timescale
    double tau_d, 
    # Chromatic Variability
    double * fvars, 
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False
) noexcept nogil:
    cdef long iresp, nv_good, nf_good
    mesh.update_mesh(
        R_in, R_out, c_f, curv, 
        log1pz=log1pz, logLbol=logLbol, logMass=logMass, H_lamp=H_lamp, pindex=pindex, 
        LampEff=mesh.LampEff,
    )
    mesh.rotate_y_cosine(cos_i)
    mesh.assign_time_lag()
    mesh.assign_response_obs()
    mesh.reorder_by_tau()
    mesh.obscuration_potential()
    mesh.get_obscuration_mask()
    mesh.__trim_mesh(&nv_good, &nf_good)
    # stat._c_compute_tf_cdf(nv_good, nf_good)
    stat._c_compute_tf_pdf(nv_good, nf_good)
    for iresp in range(mesh.ntfs):
        lia_sv_scalar_inplace(fvars[iresp]/stat._tf_cdf[(iresp+1)*stat.ntau-1], stat._tf_pdf + iresp*stat.ntau, stat.ntau) # can be removed when working with cdfs.
        lia_sv_scalar_inplace(fvars[iresp]/stat._tf_cdf[(iresp+1)*stat.ntau-1], stat._tf_cdf + iresp*stat.ntau, stat.ntau)
    (<packet_PRHQ *> stat.packet).set_var_par((<packet_PRHQ *> stat.packet).par_acf, 1., tau_d)
    return loglike_PRHQ_detr_marginalized(<packet_PRHQ *> stat.packet)

cpdef double loglike_chromatic(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp,
    # Geometry
    double R_out, double R_in, double cos_i, double c_f, double curv, 
    # Astrophysics 
    double pindex, 
    # DRW Characteristic Timescale
    double tau_d, 
    # Chromatic Variability
    double [:] fvars, 
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False,
    double underflow=-1e100,
) noexcept nogil:
    return fmax(_loglike_chromatic(
        log1pz, logLbol, logMass, H_lamp, R_out, R_in, cos_i, c_f, curv, 
        pindex, tau_d, &fvars[0],
        stat, mesh, has_driving=has_driving
    ), underflow)

cdef double _loglike_achromatic_norm(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp, 
    # Geometry
    double R_out, double R_in, double cos_i, double c_f, double curv, 
    # Astrophysics 
    double pindex, 
    # DRW Parameters
    double tau_d, 
    double sigma_d,
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False
) noexcept nogil:
    cdef long iresp, nv_good, nf_good
    cdef double denom
    mesh.update_mesh(
        R_in, R_out, c_f, curv, 
        log1pz=log1pz, logLbol=logLbol, logMass=logMass, H_lamp=H_lamp, pindex=pindex, 
        LampEff=mesh.LampEff,
    )
    mesh.rotate_y_cosine(cos_i)
    mesh.assign_time_lag()
    mesh.assign_response_obs()
    mesh.reorder_by_tau()
    mesh.obscuration_potential()
    mesh.get_obscuration_mask()
    mesh.__trim_mesh(&nv_good, &nf_good)
    # stat._c_compute_tf_cdf(nv_good, nf_good)
    stat._c_compute_tf_pdf(nv_good, nf_good)
    for iresp in range(mesh.ntfs):
        denom = 1./stat._tf_cdf[(iresp+1)*stat.ntau-1]
        lia_sv_scalar_inplace(denom, stat._tf_pdf + iresp*stat.ntau, stat.ntau) # can be removed when working with cdfs.
        lia_sv_scalar_inplace(denom, stat._tf_cdf + iresp*stat.ntau, stat.ntau)
    (<packet_PRHQ *> stat.packet).set_var_par((<packet_PRHQ *> stat.packet).par_acf, sigma_d, tau_d)
    return loglike_PRHQ_detr_marginalized(<packet_PRHQ *> stat.packet)

cpdef double loglike_achromatic_norm(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp,
    # Geometry
    double R_out, double R_in, double cos_i, double c_f, double curv, 
    # Astrophysics 
    double pindex, 
    # DRW Parameters
    double tau_d,
    double sigma_d,
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False,
    double underflow=-1e100,
) noexcept nogil:
    return fmax(_loglike_achromatic_norm(
        log1pz, logLbol, logMass, H_lamp, R_out, R_in, cos_i, c_f, curv, 
        pindex, tau_d, sigma_d,
        stat, mesh, has_driving=has_driving
    ), underflow)

cdef double _loglike_achromatic(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp, 
    # Geometry
    double R_out, double R_in, double cos_i, double c_f, double curv, 
    # Astrophysics 
    double pindex, 
    # DRW Parameters
    double tau_d, 
    double sigma_d,
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False
) noexcept nogil:
    cdef long iresp, nv_good, nf_good
    cdef double denom
    mesh.update_mesh(
        R_in, R_out, c_f, curv, 
        log1pz=log1pz, logLbol=logLbol, logMass=logMass, H_lamp=H_lamp, pindex=pindex, 
        LampEff=mesh.LampEff,
    )
    mesh.rotate_y_cosine(cos_i)
    mesh.assign_time_lag()
    mesh.assign_response_obs()
    mesh.reorder_by_tau()
    mesh.obscuration_potential()
    mesh.get_obscuration_mask()
    mesh.__trim_mesh(&nv_good, &nf_good)
    # stat._c_compute_tf_cdf(nv_good, nf_good)
    stat._c_compute_tf_pdf(nv_good, nf_good)
    (<packet_PRHQ *> stat.packet).set_var_par((<packet_PRHQ *> stat.packet).par_acf, sigma_d, tau_d)
    return loglike_PRHQ_detr_marginalized(<packet_PRHQ *> stat.packet)

cpdef double loglike_achromatic(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp,
    # Geometry
    double R_out, double R_in, double cos_i, double c_f, double curv, 
    # Astrophysics 
    double pindex, 
    # DRW Parameters
    double tau_d,
    double sigma_d,
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False,
    double underflow=-1e100,
) noexcept nogil:
    return fmax(_loglike_achromatic(
        log1pz, logLbol, logMass, H_lamp, R_out, R_in, cos_i, c_f, curv, 
        pindex, tau_d, sigma_d,
        stat, mesh, has_driving=has_driving
    ), underflow)

cdef double _loglike_achromatic_Rsub(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp, 
    # Geometry
    double R_in, double R_out,
    double cos_i, double c_f, double curv, 
    # DRW Parameters
    double tau_d, 
    double sigma_d,
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False
) noexcept nogil:
    cdef long iresp, nv_good, nf_good
    cdef double denom
    mesh.update_mesh(
        R_in*Rgrav(logMass), R_out*mesh.fRsub(logLbol, 0.05, 1500., 10.), c_f, curv, 
        log1pz=log1pz, logLbol=logLbol, logMass=logMass, H_lamp=H_lamp, 
        pindex=mesh.pindex, LampEff=mesh.LampEff,
    )
    mesh.rotate_y_cosine(cos_i)
    mesh.assign_time_lag()
    mesh.assign_response_obs()
    mesh.reorder_by_tau()
    mesh.obscuration_potential()
    mesh.get_obscuration_mask()
    mesh.__trim_mesh(&nv_good, &nf_good)
    # stat._c_compute_tf_cdf(nv_good, nf_good)
    stat._c_compute_tf_pdf(nv_good, nf_good)
    (<packet_PRHQ *> stat.packet).set_var_par((<packet_PRHQ *> stat.packet).par_acf, sigma_d, tau_d)
    return loglike_PRHQ_detr_marginalized(<packet_PRHQ *> stat.packet)

cpdef double loglike_achromatic_Rsub(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp,
    # Geometry
    double R_in, double R_out,
    double cos_i, double c_f, double curv, 
    # DRW Parameters
    double tau_d,
    double sigma_d,
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False,
    double underflow=-1e100,
) noexcept nogil:
    return fmax(_loglike_achromatic_Rsub(
        log1pz, logLbol, logMass, H_lamp, R_in, R_out, cos_i, c_f, curv, 
        tau_d, sigma_d,
        stat, mesh, has_driving=has_driving
    ), underflow)

cdef double _loglike_achromatic_Rsub_fast(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp, 
    # Geometry
    double R_in, double R_out,
    double cos_i, double c_f, double curv, 
    # DRW Parameters
    double logtau_d, 
    double logsigma_d,
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False
) noexcept nogil:
    cdef long iresp, nv_good, nf_good
    cdef double denom
    mesh.update_mesh(
        R_in*Rgrav(logMass), R_out*mesh.fRsub(logLbol, 0.05, 1500., 10.), c_f, curv, 
        log1pz=log1pz, logLbol=logLbol, logMass=logMass, H_lamp=H_lamp, 
        pindex=mesh.pindex, LampEff=mesh.LampEff,
    )
    mesh.rotate_y_cosine(cos_i)
    mesh.assign_time_lag()
    mesh.assign_response_obs()
    mesh.reorder_by_tau()
    mesh.obscuration_potential()
    mesh.get_obscuration_mask()
    mesh.__trim_mesh(&nv_good, &nf_good)
    # stat._c_compute_tf_cdf(nv_good, nf_good)
    stat._c_compute_tf_pdf(nv_good, nf_good)
    (<packet_PRHQ *> stat.packet).set_var_par((<packet_PRHQ *> stat.packet).par_acf, 2.*logsigma_d, exp(-logtau_d))
    return loglike_PRHQ_detr_marginalized(<packet_PRHQ *> stat.packet)

cpdef double loglike_achromatic_Rsub_fast(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp,
    # Geometry
    double R_in, double R_out,
    double cos_i, double c_f, double curv, 
    # DRW Parameters
    double logtau_d,
    double logsigma_d,
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False,
    double underflow=-1e100,
) noexcept nogil:
    return fmax(_loglike_achromatic_Rsub_fast(
        log1pz, logLbol, logMass, H_lamp, R_in, R_out, cos_i, c_f, curv, 
        logtau_d, logsigma_d,
        stat, mesh, has_driving=has_driving
    ), underflow)


### Full detrending

cdef double _loglike_full_achromatic_Rsub_fast(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp, 
    # Geometry
    double R_in, double R_out,
    double cos_i, double c_f, double curv, 
    # DRW Parameters
    double logtau_d, 
    double logsigma_d,
    double * q, 
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False
) noexcept nogil:
    cdef long iresp, nv_good, nf_good
    cdef double denom
    mesh.update_mesh(
        R_in*Rgrav(logMass), R_out*mesh.fRsub(logLbol, 0.05, 1500., 10.), c_f, curv, 
        log1pz=log1pz, logLbol=logLbol, logMass=logMass, H_lamp=H_lamp, 
        pindex=mesh.pindex, LampEff=mesh.LampEff,
    )
    mesh.rotate_y_cosine(cos_i)
    mesh.assign_time_lag()
    mesh.assign_response_obs()
    mesh.reorder_by_tau()
    mesh.obscuration_potential()
    mesh.get_obscuration_mask()
    mesh.__trim_mesh(&nv_good, &nf_good)
    # stat._c_compute_tf_cdf(nv_good, nf_good)
    stat._set_detrending(q)
    stat._c_compute_tf_pdf(nv_good, nf_good)
    (<packet_PRHQ *> stat.packet).set_var_par((<packet_PRHQ *> stat.packet).par_acf, 2.*logsigma_d, exp(-logtau_d))
    return loglike_PRHQ_detr_full(<packet_PRHQ *> stat.packet)

cpdef double loglike_full_achromatic_Rsub_fast(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp,
    # Geometry
    double R_in, double R_out,
    double cos_i, double c_f, double curv, 
    # DRW Parameters
    double logtau_d,
    double logsigma_d,
    double [:] q, 
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False,
    double underflow=-1e100,
) noexcept nogil:
    return fmax(_loglike_full_achromatic_Rsub_fast(
        log1pz, logLbol, logMass, H_lamp, R_in, R_out, cos_i, c_f, curv, 
        logtau_d, logsigma_d, &q[0],
        stat, mesh, has_driving=has_driving
    ), underflow)

### Full scaling and detrending (Technically chromatic)

cdef double _loglike_all_Rsub_fast(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp, 
    # Geometry
    double R_in, double R_out,
    double cos_i, double c_f, double curv, 
    # DRW Parameters
    double logtau_d, 
    double * log_w,
    double * q, 
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False
) noexcept nogil:
    cdef long nv_good, nf_good
    cdef long iresp
    cdef double * log_t = log_w + <int>has_driving
    cdef double scale
    mesh.update_mesh(
        R_in*Rgrav(logMass), R_out*mesh.fRsub(logLbol, 0.05, 1500., 10.), c_f, curv, 
        log1pz=log1pz, logLbol=logLbol, logMass=logMass, H_lamp=H_lamp, 
        pindex=mesh.pindex, LampEff=mesh.LampEff,
    )
    mesh.rotate_y_cosine(cos_i)
    mesh.assign_time_lag()
    mesh.assign_response_obs()
    mesh.reorder_by_tau()
    mesh.obscuration_potential()
    mesh.get_obscuration_mask()
    mesh.__trim_mesh(&nv_good, &nf_good)
    # stat._c_compute_tf_cdf(nv_good, nf_good)
    stat._set_detrending(q)
    stat._c_compute_tf_pdf(nv_good, nf_good)
    # log_w[0] = logsigma_d
    # log_w[1:nlc]: difference in scaling in log-space
    # IF     has_driving: scale tfs[0:ntfs]
    # IF NOT has_driving: scale tfs[1:ntfs] (assume tfs[0] is scaled to 1.)
    # Therefore:    scale tfs[1-has_driving:ntfs] with log_w[1:nlc]
    # for i in range(1-has_driving, ntfs):
    #   ilc   = i+has_driving
    #   iresp = i
    for iresp in range(1 - has_driving, mesh.ntfs):
        scale = exp(log_t[iresp])
        lia_sv_scalar_inplace(scale, stat._tf_pdf + iresp*stat.ntau, stat.ntau) # can be removed when working with cdfs.
        lia_sv_scalar_inplace(scale, stat._tf_cdf + iresp*stat.ntau, stat.ntau)
    (<packet_PRHQ *> stat.packet).set_var_par((<packet_PRHQ *> stat.packet).par_acf, 2.*log_w[0], exp(-logtau_d))
    return loglike_PRHQ_detr_full(<packet_PRHQ *> stat.packet)

cpdef double loglike_all_Rsub_fast(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp,
    # Geometry
    double R_in, double R_out,
    double cos_i, double c_f, double curv, 
    # DRW Parameters
    double logtau_d,
    double [:] log_w,
    double [:] q, 
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=False,
    double underflow=-1e100,
) noexcept nogil:
    return fmax(_loglike_all_Rsub_fast(
        log1pz, logLbol, logMass, H_lamp, R_in, R_out, cos_i, c_f, curv, 
        logtau_d, &log_w[0], &q[0],
        stat, mesh, has_driving=has_driving
    ), underflow)
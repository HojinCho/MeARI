# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strlen, strcmp, strcpy, memcpy, memset

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, M_LN2,
    sqrt, log, cos, sin, pow, exp,
    fmin, fmax, fabs,
)

from .thermal cimport (
    ThermalPacket,
    Ephoton_wave, Ephoton_freq, Rgrav,
    Planck_Nphoton_um, Planck_Intensity_um, Planck_Responsivity_um, 
    Rsub_GRAVITY, Rsub_Nenkova, 
    TProfile_Nenkova,
)

from .response cimport par_resp, ReflectionModel, ResponsePacket
from ..utils.interpolation cimport (# Interpolate
    ppi3_eval, 
    ppi3_eval_bulk, 
    ppi3_eval_bulk_and_idx, ppi3_eval_bulk_recycle_idx,
)
# cdef double ppi3_eval(double x, double * x_rb, double * coef, long n_rb) noexcept nogil
# cdef void ppi3_eval_bulk(double * rsp, long ndim, double x, double * x_rb, double * coef, long n_rb) noexcept nogil

cdef void freeResponsePacket(ResponsePacket * Rpacket) noexcept nogil:
    if Rpacket.pars != NULL:
        free(Rpacket.pars)
    if Rpacket.Tpacket != NULL:
        if Rpacket.Tpacket.par_thermal != NULL:
            free(Rpacket.Tpacket.par_thermal)
        free(Rpacket.Tpacket)
    if Rpacket.band != NULL:
        free(Rpacket.band)
    if Rpacket.x_rb != NULL:
        free(Rpacket.x_rb)
    if Rpacket.coef_r != NULL:
        free(Rpacket.coef_r)
    if Rpacket.coef_e != NULL:
        free(Rpacket.coef_e)
    free(Rpacket)

cdef double redshifted_logT(
    func_TProfile T, double R, double * xyz, double * nor, double fgeom, void * par, double log1pz,
) noexcept nogil:
    # T * lambda_e = (T / (1+z)) * lambda_o = T_r * lambda_o
    return T(R, xyz, nor, fgeom, par) - log1pz 

cdef double redshifted_T(
    func_TProfile T, double R, double * xyz, double * nor, double fgeom, void * par, double log1pz,
) noexcept nogil:
    return exp(redshifted_logT(T, R, xyz, nor, fgeom, par, log1pz))

cdef double incident_projection(double * xyz, double * nor, double Hlamp) noexcept nogil:
    cdef double zdelt = xyz[2] - Hlamp
    cdef double R     = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + zdelt*zdelt)
    return -(nor[0]*xyz[0] + nor[1]*xyz[1] + nor[2]*zdelt)/R

####
# Illuminations
####

cdef void respmodel_goad_pow_flat(
    long nv, 
    void * resp, 
    void * emis, 
    void ** data_arr, 
    par_resp * par
) noexcept nogil:
    cdef long i
    cdef double p     = -par.pindex[0]
    cdef double * R   = <double*>data_arr[0]
    cdef double * rsp = <double*>resp
    cdef double * ems = <double*>emis
    for i in prange(nv, nogil=True):
        rsp[i] = pow(R[i], p)
        ems[i] = rsp[i] # Normalizing to the integration.

cdef void respmodel_goad_pow_proj(
    long nv, 
    void * resp, 
    void * emis, 
    void ** data_arr, 
    par_resp * par
) noexcept nogil:
    cdef long i, I
    cdef double p        = -par.pindex[0]
    cdef double * R      = <double*>data_arr[0]
    cdef double * R3d    = <double*>data_arr[3]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef double * rsp = <double*>resp
    cdef double * ems = <double*>emis
    for i in prange(nv, nogil=True): # Use BLAS?
        I = 3*i
        # rsp[i] = -pow(R[i], p)*(
        #     normal[I  ]*coords[I  ]
        #   + normal[I+1]*coords[I+1]
        #   + normal[I+2]*coords[I+2]
        # )/R3d[i]
        rsp[i] = pow(R[i], p)*incident_projection(coords+I, normal+I, par.H_lamp[0])
        ems[i] = rsp[i] # Normalizing to the integration.

cdef void respmodel_goad_pow(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    if par.c_f[0]==0 or par.curv[0]==1 or par.mod_inci==ReflectionModel.REF_ISOTROPIC:
        # REF_ISOTROPIC means incoming light is received fully without inclination effect;
        # This effectively treats the responding surface as a cloud of optically thin particles.
        # This is opposed to REFMODEL_LAMBERTIAN, which should be used to model optically thick surface.
        respmodel_goad_pow_flat(nv, resp, emis, data_arr, par)
    else:
        respmodel_goad_pow_proj(nv, resp, emis, data_arr, par)

# pow(R, -1) is slower than using 1/R (extra ~10ms in 256x360), 
# but still implemented in this way to make the power index fittable.

cdef void respmodel_goad_inv_flat(
    long nv, 
    void * resp, 
    void * emis, 
    void ** data_arr, 
    par_resp * par
) noexcept nogil:
    cdef long i
    cdef double * R   = <double*>data_arr[0]
    cdef double * rsp = <double*>resp
    cdef double * ems = <double*>emis
    for i in prange(nv, nogil=True): # Use BLAS?
        rsp[i] = 1/R[i]
        ems[i] = rsp[i] # Normalizing to the integration.

cdef void respmodel_goad_inv_proj(
    long nv, 
    void * resp, 
    void * emis, 
    void ** data_arr, 
    par_resp * par
) noexcept nogil:
    cdef long i, I
    cdef double * R      = <double*>data_arr[0]
    cdef double * R3d    = <double*>data_arr[3]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef double * rsp = <double*>resp
    cdef double * ems = <double*>emis
    for i in prange(nv, nogil=True): # Use BLAS?
        I = 3*i
        # rsp[i] = -(
        #     normal[I  ]*coords[I  ]
        #   + normal[I+1]*coords[I+1]
        #   + normal[I+2]*coords[I+2]
        # )/(R3d[i]*R[i]) # xyz is normalized by R3d because p is -pindex "-1".
        rsp[i] = incident_projection(coords+I, normal+I, par.H_lamp[0])/R[i]
        ems[i] = rsp[i] # Normalizing to the integration.

cdef void respmodel_goad_inv(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    if par.c_f[0]==0 or par.curv[0]==1 or par.mod_inci==ReflectionModel.REF_ISOTROPIC:
        # REF_ISOTROPIC means incoming light is received fully without inclination effect;
        # This effectively treats the responding surface as a cloud of optically thin particles.
        # This is opposed to REFMODEL_LAMBERTIAN, which should be used to model optically thick surface.
        respmodel_goad_inv_flat(nv, resp, emis, data_arr, par)
    else:
        respmodel_goad_inv_proj(nv, resp, emis, data_arr, par)

cdef void respmodel_goad_invsq_flat(
    long nv, 
    void * resp, 
    void * emis, 
    void ** data_arr, 
    par_resp * par
) noexcept nogil:
    cdef long i
    cdef double * R   = <double*>data_arr[0]
    cdef double * rsp = <double*>resp
    cdef double * ems = <double*>emis
    for i in prange(nv, nogil=True): # Use BLAS?
        rsp[i] = 1/(R[i]*R[i])
        ems[i] = rsp[i] # Normalizing to the integration.

cdef void respmodel_goad_invsq_proj(
    long nv, 
    void * resp, 
    void * emis, 
    void ** data_arr, 
    par_resp * par
) noexcept nogil:
    cdef long i, I
    cdef double * R      = <double*>data_arr[0]
    cdef double * R3d    = <double*>data_arr[3]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef double * rsp = <double*>resp
    cdef double * ems = <double*>emis
    for i in prange(nv, nogil=True): # Use BLAS?
        I = 3*i
        # rsp[i] = -(
        #     normal[I  ]*coords[I  ]
        #   + normal[I+1]*coords[I+1]
        #   + normal[I+2]*coords[I+2]
        # )/(R3d[i]*R[i]*R[i]) # xyz is normalized by R because p is -pindex "-1".
        rsp[i] = incident_projection(coords+I, normal+I, par.H_lamp[0])/(R[i]*R[i])
        ems[i] = rsp[i] # Normalizing to the integration.

cdef void respmodel_goad_invsq(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    if par.c_f[0]==0 or par.curv[0]==1 or par.mod_inci==ReflectionModel.REF_ISOTROPIC:
        # REF_ISOTROPIC means incoming light is received fully without inclination effect;
        # This effectively treats the responding surface as a cloud of optically thin particles.
        # This is opposed to REFMODEL_LAMBERTIAN, which should be used to model optically thick surface.
        respmodel_goad_invsq_flat(nv, resp, emis, data_arr, par)
    else:
        respmodel_goad_invsq_proj(nv, resp, emis, data_arr, par)

#########################################
# Black Body Reprocessing Response
## Monochromatic
cdef void _respmodel_T_M_flat(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef long i, I, j, Iv
    cdef double fgeom
    cdef double * R      = <double*>data_arr[0]
    cdef double * RLamp  = <double*>data_arr[4]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef double * rsp    = <double*>resp
    cdef double * ems    = <double*>emis
    cdef double log1pz = (Rpacket.pars).log1pz[0]
    cdef long offset = Rpacket.offset
    cdef long nband  = Rpacket.nband
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0]))
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2) # Added ln(2) to halve the luminosity (one-side of disk), modify this to use Lx instead of Lbol
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2)/((Rpacket.pars).LampEff[0])
    cdef double emis_factor = exp(4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2)/((Rpacket.pars).LampEff[0])
    cdef double T
    for i in prange(nv, nogil=True): # Use BLAS?
        I = ntfs*i + offset
        Iv = 3*i
        T = redshifted_T(
            Rpacket.Tpacket.T, 
            R[i], coords+Iv, normal+Iv, 1., Rpacket.Tpacket.par_thermal,
            log1pz
        )
        fgeom = 1./(RLamp[i]*RLamp[i])
        for j in range(nband):
            rsp[I+j] = Planck_Responsivity_um(Rpacket.band[j], T)*fgeom
            ems[I+j] = Planck_Nphoton_um(Rpacket.band[j], T)*emis_factor
            
cdef void _respmodel_T_M_proj(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef long i, I, j, Iv
    cdef double fgeom
    cdef double * R      = <double*>data_arr[0]
    cdef double * R3d    = <double*>data_arr[3]
    cdef double * RLamp  = <double*>data_arr[4]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef double * rsp    = <double*>resp
    cdef double * ems    = <double*>emis
    cdef double log1pz = (Rpacket.pars).log1pz[0]
    cdef long offset = Rpacket.offset
    cdef long nband  = Rpacket.nband
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0]))
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2) # Added ln(2) to halve the luminosity (one-side of disk), modify this to use Lx instead of Lbol
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2)/((Rpacket.pars).LampEff[0])
    cdef double emis_factor = exp(4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2)/((Rpacket.pars).LampEff[0])
    cdef double T
    for i in prange(nv, nogil=True): # Use BLAS?
        I = ntfs*i + offset
        Iv = 3*i
        # fgeom = -(normal[Iv  ]*coords[Iv  ] + normal[Iv+1]*coords[Iv+1] + normal[Iv+2]*coords[Iv+2])/R3d[i]
        fgeom = incident_projection(coords+Iv, normal+Iv, (Rpacket.pars).H_lamp[0])
        T = redshifted_T(
            Rpacket.Tpacket.T, 
            R[i], coords+Iv, normal+Iv, fgeom, Rpacket.Tpacket.par_thermal,
            log1pz
        )
        fgeom = fgeom/(RLamp[i]*RLamp[i])
        for j in range(nband):
            rsp[I+j] = Planck_Responsivity_um(Rpacket.band[j], T)*fgeom
            ems[I+j] = Planck_Nphoton_um(Rpacket.band[j], T)*emis_factor

#### No symmetry
cdef void respmodel_T_M(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    if par.c_f[0]==0 or par.curv[0]==1 or par.mod_inci==ReflectionModel.REF_ISOTROPIC:
        # REF_ISOTROPIC means incoming light is received fully without (source) inclination effect;
        # This effectively treats the responding surface as a cloud of optically thin particles.
        # This is opposed to REFMODEL_LAMBERTIAN, which should be used to model optically thick surface.
        _respmodel_T_M_flat(nv, ntfs, resp, emis, data_arr, Rpacket)
    else:
        _respmodel_T_M_proj(nv, ntfs, resp, emis, data_arr, Rpacket)

cdef void respmodel_T_M_proj(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    _respmodel_T_M_proj(nv, ntfs, resp, emis, data_arr, Rpacket)

#### Azimuthal symmetry
cdef void respmodel_T_M_sym(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    cdef long i, j, I, K
    cdef long offset = Rpacket.offset
    cdef long nrad1  = Rpacket.nvr
    cdef size_t blocksize = Rpacket.nband*sizeof(double)
    cdef double * resps_now
    cdef double * emiss_now
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    if par.c_f[0]==0 or par.curv[0]==1 or par.mod_inci==ReflectionModel.REF_ISOTROPIC:
        # REF_ISOTROPIC means incoming light is received fully without (source) inclination effect;
        # This effectively treats the responding surface as a cloud of optically thin particles.
        # This is opposed to REFMODEL_LAMBERTIAN, which should be used to model optically thick surface.
        _respmodel_T_M_flat(Rpacket.nvr, ntfs, resp, emis, data_arr, Rpacket)
    else:
        _respmodel_T_M_proj(Rpacket.nvr, ntfs, resp, emis, data_arr, Rpacket)
    # this seems more logical than for i for j.
    for j in prange(nrad1, nogil=True):
        K = ntfs*j + offset
        resps_now = (<double*>resp) + K
        emiss_now = (<double*>emis) + K
        for i in range(1, Rpacket.nva):
            I = i*nrad1*ntfs
            memcpy(<void*>(resps_now + I), <void*>(resps_now), blocksize)
            memcpy(<void*>(emiss_now + I), <void*>(emiss_now), blocksize)

#### Azimuthal symmetry
cdef void respmodel_T_M_symproj(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    cdef long i, j, I, K
    cdef long offset = Rpacket.offset
    cdef long nrad1  = Rpacket.nvr
    cdef size_t blocksize = Rpacket.nband*sizeof(double)
    cdef double * resps_now
    cdef double * emiss_now
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    _respmodel_T_M_proj(Rpacket.nvr, ntfs, resp, emis, data_arr, Rpacket)
    # this seems more logical than for i for j.
    for j in prange(nrad1, nogil=True):
        K = ntfs*j + offset
        resps_now = (<double*>resp) + K
        emiss_now = (<double*>emis) + K
        for i in range(1, Rpacket.nva):
            I = i*nrad1*ntfs
            memcpy(<void*>(resps_now + I), <void*>(resps_now), blocksize)
            memcpy(<void*>(emiss_now + I), <void*>(emiss_now), blocksize)


## Bandpass
cdef void _respmodel_T_I_flat(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef long i, I, j, Iv
    cdef double fgeom
    cdef double * R      = <double*>data_arr[0]
    cdef double * RLamp  = <double*>data_arr[4]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef double * rsp    = <double*>resp
    cdef double * ems    = <double*>emis
    cdef double log1pz = (Rpacket.pars).log1pz[0]
    cdef long offset = Rpacket.offset
    cdef long nband  = Rpacket.nband
    cdef double * logT_r = Rpacket.x_rb
    cdef double * coef_r = Rpacket.coef_r
    cdef double * coef_e = Rpacket.coef_e
    cdef long n_rb = Rpacket.n_rb
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0]))
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2) # Added ln(2) to halve the luminosity (one-side of disk), modify this to use Lx instead of Lbol
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2)/((Rpacket.pars).LampEff[0])
    cdef double emis_factor = exp(4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2)/((Rpacket.pars).LampEff[0])
    cdef long idx_ppi3
    cdef double logT
    for i in prange(nv, nogil=True): # Use BLAS?
        I = ntfs*i + offset
        Iv = 3*i
        logT = redshifted_logT(
            Rpacket.Tpacket.T, 
            R[i], coords+Iv, normal+Iv, 1., Rpacket.Tpacket.par_thermal,
            log1pz
        )
        # ppi3_eval_bulk(rsp+I, nband, logT, logT_r, coef_r, n_rb) 
        idx_ppi3 = ppi3_eval_bulk_and_idx(rsp+I, nband, logT, logT_r, coef_r, n_rb) 
        ppi3_eval_bulk_recycle_idx(ems+I, nband, logT, logT_r, coef_e, n_rb, idx_ppi3) 
        fgeom = 1./(RLamp[i]*RLamp[i])
        for j in range(nband):
            rsp[I+j] = exp(rsp[I+j])*fgeom
            ems[I+j] = exp(ems[I+j])*emis_factor

cdef void _respmodel_T_I_proj(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef long i, I, j, Iv
    cdef double fgeom
    cdef double * R      = <double*>data_arr[0]
    cdef double * R3d    = <double*>data_arr[3]
    cdef double * RLamp  = <double*>data_arr[4]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef double * rsp    = <double*>resp
    cdef double * ems    = <double*>emis
    cdef double log1pz = (Rpacket.pars).log1pz[0]
    cdef long offset = Rpacket.offset
    cdef long nband  = Rpacket.nband
    cdef double * logT_r = Rpacket.x_rb
    cdef double * coef_r = Rpacket.coef_r
    cdef double * coef_e = Rpacket.coef_e
    cdef long n_rb = Rpacket.n_rb
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0]))
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2) # Added ln(2) to halve the luminosity (one-side of disk), modify this to use Lx instead of Lbol
    # cdef double emis_factor = exp(-4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2)/((Rpacket.pars).LampEff[0])
    cdef double emis_factor = exp(4*log1pz-M_LN10*((Rpacket.pars).logLbol[0])+M_LN2)/((Rpacket.pars).LampEff[0])
    cdef long idx_ppi3
    cdef double logT
    for i in prange(nv, nogil=True): # Use BLAS?
        I = ntfs*i + offset
        Iv = 3*i
        # fgeom = -(normal[Iv  ]*coords[Iv  ] + normal[Iv+1]*coords[Iv+1] + normal[Iv+2]*coords[Iv+2])/R3d[i]
        fgeom = incident_projection(coords+Iv, normal+Iv, (Rpacket.pars).H_lamp[0])
        logT = redshifted_logT(
            Rpacket.Tpacket.T, 
            R[i], coords+Iv, normal+Iv, fgeom, Rpacket.Tpacket.par_thermal,
            log1pz
        )
        # ppi3_eval_bulk(rsp+I, nband, logT, logT_r, coef_r, n_rb)
        idx_ppi3 = ppi3_eval_bulk_and_idx(rsp+I, nband, logT, logT_r, coef_r, n_rb) 
        ppi3_eval_bulk_recycle_idx(ems+I, nband, logT, logT_r, coef_e, n_rb, idx_ppi3) 
        # fgeom = fgeom/(R[i]*R[i])
        fgeom = fgeom/(RLamp[i]*RLamp[i])
        for j in range(nband):
            rsp[I+j] = exp(rsp[I+j])*fgeom
            ems[I+j] = exp(ems[I+j])*emis_factor

#### No symmetry
cdef void respmodel_T_I(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    if par.c_f[0]==0 or par.curv[0]==1 or par.mod_inci==ReflectionModel.REF_ISOTROPIC:
        # REF_ISOTROPIC means incoming light is received fully without (source) inclination effect;
        # This effectively treats the responding surface as a cloud of optically thin particles.
        # This is opposed to REFMODEL_LAMBERTIAN, which should be used to model optically thick surface.
        _respmodel_T_I_flat(nv, ntfs, resp, emis, data_arr, Rpacket)
    else:
        _respmodel_T_I_proj(nv, ntfs, resp, emis, data_arr, Rpacket)

cdef void respmodel_T_I_proj(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    _respmodel_T_I_proj(nv, ntfs, resp, emis, data_arr, Rpacket)

#### Azimuthal symmetry
cdef void respmodel_T_I_sym(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    cdef long i, j, I, K
    cdef long offset = Rpacket.offset
    cdef long nrad1  = Rpacket.nvr
    cdef size_t blocksize = Rpacket.nband*sizeof(double)
    cdef double * resps_now
    cdef double * emiss_now
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    if par.c_f[0]==0 or par.curv[0]==1 or par.mod_inci==ReflectionModel.REF_ISOTROPIC:
        # REF_ISOTROPIC means incoming light is received fully without (source) inclination effect;
        # This effectively treats the responding surface as a cloud of optically thin particles.
        # This is opposed to REFMODEL_LAMBERTIAN, which should be used to model optically thick surface.
        _respmodel_T_I_flat(Rpacket.nvr, ntfs, resp, emis, data_arr, Rpacket)
    else:
        _respmodel_T_I_proj(Rpacket.nvr, ntfs, resp, emis, data_arr, Rpacket)
    # this seems more logical than for i for j.
    for j in prange(nrad1, nogil=True):
        K = ntfs*j + offset
        resps_now = (<double*>resp) + K
        emiss_now = (<double*>emis) + K
        for i in range(1, Rpacket.nva):
            I = i*nrad1*ntfs
            memcpy(<void*>(resps_now + I), <void*>(resps_now), blocksize)
            memcpy(<void*>(emiss_now + I), <void*>(emiss_now), blocksize)

#### Azimuthal symmetry
cdef void respmodel_T_I_symproj(
    long nv, long ntfs, void * resp, void * emis, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    cdef long i, j, I, K
    cdef long offset = Rpacket.offset
    cdef long nrad1  = Rpacket.nvr
    cdef size_t blocksize = Rpacket.nband*sizeof(double)
    cdef double * resps_now
    cdef double * emiss_now
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    _respmodel_T_I_proj(Rpacket.nvr, ntfs, resp, emis, data_arr, Rpacket)
    # this seems more logical than for i for j.
    for j in prange(nrad1, nogil=True):
        K = ntfs*j + offset
        resps_now = (<double*>resp) + K
        emiss_now = (<double*>emis) + K
        for i in range(1, Rpacket.nva):
            I = i*nrad1*ntfs
            memcpy(<void*>(resps_now + I), <void*>(resps_now), blocksize)
            memcpy(<void*>(emiss_now + I), <void*>(emiss_now), blocksize)






######################
# Helper

cdef void _T_flat(
    long nv, double * outarr, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef long i, Iv
    cdef double * R      = <double*>data_arr[0]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef long offset = Rpacket.offset
    for i in prange(nv, nogil=True): # Use BLAS?
        Iv = 3*i
        outarr[i] = redshifted_logT(
            Rpacket.Tpacket.T, 
            R[i], coords+Iv, normal+Iv, 1., Rpacket.Tpacket.par_thermal,
            0.,
        )

cdef void _T_proj(
    long nv, double * outarr, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef long i, Iv
    cdef double fgeom
    cdef double * R      = <double*>data_arr[0]
    # cdef double * R3d    = <double*>data_arr[3]
    cdef double * normal = <double*>data_arr[1]
    cdef double * coords = <double*>data_arr[2]
    cdef long offset = Rpacket.offset
    for i in prange(nv, nogil=True): # Use BLAS?
        Iv = 3*i
        fgeom = incident_projection(coords+Iv, normal+Iv, (Rpacket.pars).H_lamp[0])
        outarr[i] = redshifted_logT(
            Rpacket.Tpacket.T, 
            R[i], coords+Iv, normal+Iv, fgeom, Rpacket.Tpacket.par_thermal,
            0.,
        )


cdef void T_all(
    long nv, double * outarr, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    if par.c_f[0]==0 or par.curv[0]==1 or par.mod_inci==ReflectionModel.REF_ISOTROPIC:
        # REF_ISOTROPIC means incoming light is received fully without (source) inclination effect;
        # This effectively treats the responding surface as a cloud of optically thin particles.
        # This is opposed to REFMODEL_LAMBERTIAN, which should be used to model optically thick surface.
        _T_flat(nv, outarr, data_arr, Rpacket)
    else:
        _T_proj(nv, outarr, data_arr, Rpacket)

cdef void T_proj(
    long nv, double * outarr, void ** data_arr, ResponsePacket * Rpacket,
) noexcept nogil:
    cdef par_resp * par = Rpacket.pars
    Rpacket.Tpacket.Tprep(Rpacket.Tpacket.par_thermal)
    _T_proj(nv, outarr, data_arr, Rpacket)
    


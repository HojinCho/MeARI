from .stats cimport Stats
from ..mesh.diskmesh cimport DiskMesh

### Marginalized Detrending
cpdef double loglike_chromatic(
    # AGN Parameter
    double log1pz, double logLbol, double logMass, double H_lamp, 
    # Geometry
    double R_out, double R_in, double incl, double f_c, double curv, 
    # Astrophysics 
    double pindex, 
    # DRW Characteristic Timescale
    double tau_d, 
    # Chromatic Variability
    double [:] fvars, 
    # Backends
    Stats stat, DiskMesh mesh,
    bint has_driving=*,
    double underflow=*,
) noexcept nogil
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
    bint has_driving=*,
    double underflow=*,
) noexcept nogil
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
    bint has_driving=*,
    double underflow=*,
) noexcept nogil
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
    bint has_driving=*,
    double underflow=*,
) noexcept nogil
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
    bint has_driving=*,
    double underflow=*,
) noexcept nogil


### Full Detrending
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
    bint has_driving=*,
    double underflow=*,
) noexcept nogil


### Full scaling and detrending
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
    bint has_driving=*,
    double underflow=*,
) noexcept nogil
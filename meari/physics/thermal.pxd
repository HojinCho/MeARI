# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

from ..utils.types cimport func_t_double_par


# func_TProfile: returns ln(T).
ctypedef double (*func_TProfile)(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil
ctypedef double (*func_Rsub)(double logL, double adust, double Tsub, double BCvLv) noexcept nogil
ctypedef void (*func_prep_thermal)(void * par) noexcept nogil # Prepare some normalizing values. For some reasons, setting (par_resp * par) instead of (void * par) makes compilation error.

# Common Packet
cdef struct ThermalPacket:
    func_prep_thermal Tprep
    func_TProfile     T
    void *            par_thermal

# Model-specific parameter carriers
### Starkey+2023
cdef struct par_starkey: # This is superset of par_nenkova
    double * logLbol
    double * logMass
    double * rotmat # input z is rotated!
    double * Rg
    double * Rsub
    double eps_disk # L_bol_disk/(Mdot c^2)
    double eps_lamp # L_bol_lamp(1-A)/(Mdot c^2)
    double A        # Albedo
    double * H_lamp   # in R_g, typcically <20
    double Rin_visc # in R_g, 6 for Schwarzschild, 3 for maximally rotating prograde Kerr
    bint is_radial_2d # whether radial is being computed with 2d or 3d distance.
    # func_Rsub fRsub
    # double adust
    # double Tsub
    # double BCvLv # =LBol/vLv, typically ~10. e.g., BC5100=LBol/L5100~10.

### Nenkova+2008
cdef struct par_nenkova:
    double * logLbol
    double Rsub
    func_Rsub fRsub
    double adust
    double Tsub
    double BCvLv # =LBol/vLv, typically ~10. e.g., BC5100=LBol/L5100~10.

### Nenkova+2008 with R_in cutoff
cdef struct par_nenkovac:
    double * logLbol
    double * Rg
    double r_ISCO   # 6 for nonrotating black hole, 1 for maximally prograde, 9 for maximally retrograde.
    double f_cutoff # 1 if fully viscous, 0 if fully radiative.
    double factor   # buffer for storing result.
    double r_in     # buffer for storing result.
    double Rsub
    func_Rsub fRsub
    double adust
    double Tsub
    double BCvLv # =LBol/vLv, typically ~10. e.g., BC5100=LBol/L5100~10.

### Cackett+2007
cdef struct par_disk:
    double T_bright
    double T_faint
    # Viscous heating parameter
    double R_0
    # Lamppost parameter
    double * H_lamp
    double * logLbol
    double A

# Fundamental Physics Functions
cdef double Rgrav(double logM) noexcept nogil
cdef double Ephoton_wave(double wave) noexcept nogil
cdef double Ephoton_freq(double freq) noexcept nogil
cdef double Planck_Nphoton_um(double wave, double T) noexcept nogil
cdef double Planck_Intensity_um(double wave, double T) noexcept nogil
cdef double Planck_Responsivity_um(double wave, double T) noexcept nogil
# Auxiliary Functions
### Dust Sublimination Radii: See Barvainis 1987, and other papers
cdef double Rsub_Kishimoto2007(double logL, double adust, double Tsub, double BCvLv) noexcept nogil
cdef double Rsub_GRAVITY(double logL, double adust, double Tsub, double BCvLv) noexcept nogil
cdef double Rsub_Nenkova(double logL, double adust, double Tsub, double BCvLv) noexcept nogil
# Thermal Profiles
## Disk
### Cackett+2007
cdef double TProfile_Cackett(double R, double R0, double T0, double alpha) noexcept nogil
### Starkey+2023
cdef double TProfile_Starkey2023_ThickDisk_r3d(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil
cdef double TProfile_Starkey2023_ThickDisk_r2d(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil
## Dusts
### Nenkova
cdef void TPrep_Nenkova(void * par) noexcept nogil
cdef double TProfile_Nenkova(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil
cdef double TProfile_Nenkova_max(double rho) noexcept nogil
cdef double TProfile_Nenkova_min(double rho) noexcept nogil
## Hybrid
### Nenkova with Cutoff
cdef void TPrep_NenkovaC(void * par) noexcept nogil
cdef double TProfile_NenkovaC(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil
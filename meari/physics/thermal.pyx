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
    M_PI, M_E, M_LN10, 
    sqrt, log, cos, sin, pow, exp, log10,
    fmin, fmax, fabs,
)

from ..utils.algorithms cimport integer_pow

# from .response cimport par_resp
from .thermal cimport par_nenkova, par_nenkovac, par_starkey

################################################################################
# Physical Constants
################################################################################

## Exact-value Fundamental Constants
################################################################################
cdef double hPlanck = 6.62607015e-27 # erg s # Planck Constant
cdef double c0 = 29979245800.0       # cm/s  # Vacuum Speed of Light
cdef double kB = 1.380649e-16        # erg/K # Boltzmann Constant

## 2nd-order Exact-value Constants (double-precision approximats)
################################################################################
cdef double hc = 1.986445857148928675008777382e-16      # erg cm            # h*c
cdef double hckB = 1.4387768775039337901676013279982    # cm K # hc/kB
cdef double hckB_um = 14387.768775039337242300174056379 # cm K # hc/kB
cdef double invsigma_SB = 17635.51973952066253185169490 # erg^-1 s cm^2 K^4 # inverse of Stefan-Boltzmann Constant in cgs
### For the Black-body response in terms of wavelength microns
# cdef double C1_BBemission = 4.0227180057120193701709325734611E+49
# cdef double C1_BBresponse = 3.0266524669317244421289896765449e+29
cdef double C1_BBemission = 4.0227180057120193701709325734611E+53
cdef double C1_BBresponse = 3.0266524669317244421289896765449e+29
cdef double C2_BBresponse = 14387.76877503933790167601328


## Measured Constants
################################################################################
cdef double GMSun = 1.32712440041279419e+26 # cm^3/s^2 double precision # https://ssd.jpl.nasa.gov/astro_par.html

################################################################################
# Unit Conversion Constants
################################################################################
cdef double um_to_cm = 1e-4
cdef double pc_to_ltday = 1191.2862616913267
cdef double cm_to_ltday = 3.86069554627490798E-16 # lt-day/cm

################################################################################
# Derived Constants
################################################################################
# cdef double T4Starkey = 2.3849280806110996950418878871737E-38 # (0.5*GMSun*invsigma_SB/(M_PI*c0*c0))*integer_pow(cm_to_ltday,3) # in cgs, need to convert to lt-day related constants.
cdef double T4Starkey = -86.629064564768141045283182865560 # ln((0.5*GMSun*invsigma_SB/(M_PI*c0*c0))*integer_pow(cm_to_ltday,3)) # in cgs, need to convert to lt-day related constants.
cdef double Gc2 = 5.7007997096913211035632997933311E-11 # cm_to_ltday*GMSun/(c0*c0) # in lt-days
cdef double StarkeyViscTerm = 0.75
cdef double StarkeyLampTerm = 1.0

################################################################################
# Fundamental Physics Functions
################################################################################
cdef double Rgrav(double logM) noexcept nogil:
    # R_g = GM/c^2
    # returns in lt-days
    return Gc2*exp(M_LN10*logM)

cdef double Ephoton_wave(double wave) noexcept nogil:
    return hc/wave
cdef double Ephoton_freq(double freq) noexcept nogil:
    return hPlanck*freq

cdef double Planck_Nphoton_um(double wave, double T) noexcept nogil:
    return ( # C1_BBemission is in um scale, so no need for um_to_cm conversion.
        (C1_BBemission/integer_pow(wave,4)) # numerator, divided by Ephoton
        /(exp(hckB_um/(wave*T)) - 1)        # denominator
    )
cdef double Planck_Intensity_um(double wave, double T) noexcept nogil:
    # cm = um * (d cm/d um)
    cdef double wv = wave*um_to_cm       # cm
    cdef double Ephot = Ephoton_wave(wv)
    # F_cm d cm = F_um d um -> F_um = F_cm * (d cm/d um)
    return um_to_cm*( # to convert back to /um scale
        (2*Ephot*c0/integer_pow(wv,4)) # numerator
        /(exp(Ephot/(kB*T)) - 1) # denominator
    )
cdef double Planck_Responsivity_um(double wave, double T) noexcept nogil:
    # Use to obtain Nenkova or Disk(CREAM) Responsivity.
    # multiply this by 
    #     cos theta_i / r^2
    # to obtain the response_raw
    # and further multiply by 
    #     cos theta_r
    # to obtain responsess
    cdef double invBoltzmann = exp(C2_BBresponse/(wave*T))
    return (
        C1_BBresponse
        *(1./integer_pow(wave*T, 5))
        *((invBoltzmann)/integer_pow((invBoltzmann-1.), 2))
    ) 
    # Note on perfomance: This function is computed (1000 * 360 * 257) times within 1.43s. 
    # Makes it run 0.015 us per call (probably automatic vecotriaztion).
    # Even assuming this is vectorized (across 64 threads), it took only ~1 us per call.
    # So, no need for further optimization.

################################################################################
# Sublimation Radius
################################################################################
### See Barvainis 1987, 1992, Kishimoto+2007, and other papers
cdef double Rsub_Kishimoto2007(double logL, double adust, double Tsub, double BCvLv) noexcept nogil:
    # Kishimoto+2007, Lbol in erg/s, adust in um, Tsub in K.
    # 1.3pc * sqrt(1/10) * sqrt(LUV/1e+45) * (Tsub/1500)**-2.8 * (adust/0.05)**-0.5
    # 489.7330932556 lt-day * sqrt(LUV/1e+45) * (Tsub/1500)**-2.8 * (adust/0.05)**-0.5
    # using LUV ~ 6vLv for Vband
    # 1199.59618863 lt-day * sqrt(vLv/1e+45) * (Tsub/1500)**-2.8 * (adust/0.05)**-0.5
    # 1199.59618863 lt-day * sqrt(1/10) * sqrt(10/BCvLv) * sqrt(Lbol/1e+45) * (Tsub/1500)**-2.8 * (adust/0.05)**-0.5
    # return 379.34565223508480*sqrt((10./BCvLv)*(Lbol/1e+45)/(adust/0.05))*pow(Tsub/1500., -2.8)
    # return 379.34565223508480*sqrt(Lbol/(BCvLv*adust*2e+45))*pow(Tsub/1500., -2.8)
    return 379.34565223508480*exp(0.5*(M_LN10*logL-log(BCvLv*adust*2e+45)))*pow(Tsub/1500., -2.8)
    
cdef double Rsub_GRAVITY(double logL, double adust, double Tsub, double BCvLv) noexcept nogil:
    # Gravity+2024, Tsub=1900K, using exponent -2.8....
    # return 245.8224639397038*sqrt(Lbol/1e+45)
    return 245.8224639397038*exp(0.5*M_LN10*(logL-45.))

cdef double Rsub_Nenkova(double logL, double adust, double Tsub, double BCvLv) noexcept nogil:
    # Nenkova+2008, Tsub=1500K
    # Used L as Lbol, but equivalently BCvLv ~ 40 according to Barvainis1987 and Kishimoto+2007
    # Lbol/LUV ~ 6.34 ~ BCvLv vLv / LUV ~ BCvLv /6
    # BCvLv ~ 40
    # return 476.5145046765307*sqrt(Lbol/1e+45)
    return 476.5145046765307*exp(0.5*M_LN10*(logL-45.))


################################################################################
# Thermal Profiles
################################################################################

## Disk
################################################################################
### Cackett+2007
cdef double TProfile_Cackett(double R, double R0, double T0, double alpha) noexcept nogil:
    return log(T0) - alpha*log(R/R0)

### Starkey+2023
# Actually this comes from old papers, e.g., Cackett+2007, but the convention is from Starkey+2023.
# fgeom is cos i or something appropriate.
# Albedo=0.2 can be used adopted from Netzer2022
cdef double TProfile_Starkey2023_ThickDisk_r3d(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil:
    # Don't use this, since viscous term should be from R2d, not R3d,
    #     as it is direct result of angular momentum loss.
    cdef par_starkey *p = <par_starkey*>par
    cdef double rg = Rgrav(p.logMass[0])
    cdef double rgr = rg/R
    cdef double z = p.rotmat[0]*xyz[2] + p.rotmat[2]*xyz[0]
    return 0.25*(
        T4Starkey
      + p.logMass[0] + p.logLbol[0] - log(p.eps_disk+p.eps_lamp/(1.-p.A))
      + log(
            StarkeyViscTerm*(1.-sqrt(p.Rin_visc*rgr))/integer_pow(R,3) # Viscous heating term
          + StarkeyLampTerm*p.eps_lamp*fgeom/(rg*(R*R - z*z + integer_pow(z-p.H_lamp[0],2))) # Lamppost radiation term
        )
    )

cdef double TProfile_Starkey2023_ThickDisk_r2d(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil:
    cdef par_starkey *p = <par_starkey*>par
    cdef double rg = Rgrav(p.logMass[0])
    cdef double rgr = rg/R
    cdef double z = p.rotmat[0]*xyz[2] + p.rotmat[2]*xyz[0]
    cdef double nz = p.rotmat[0]*normal[2] + p.rotmat[2]*normal[0] # Normalize to the 2d-planar area element.
    return 0.25*(
        T4Starkey
      + p.logMass[0] + p.logLbol[0] - log(p.eps_disk+p.eps_lamp/(1.-p.A))
      + log(        
            StarkeyViscTerm*(1.-sqrt(p.Rin_visc*rgr))/integer_pow(R,3) # Viscous heating term
          + StarkeyLampTerm*p.eps_lamp*fgeom/(rg*(R*R + integer_pow(z-p.H_lamp[0],2))/nz) # Lamppost radiation term
        )
    )

## Dusts
################################################################################
### Nenkova

cdef void TPrep_Nenkova(void * par) noexcept nogil:
    cdef par_nenkova * p = <par_nenkova*>(par)
    p.Rsub = p.fRsub(
        p.logLbol[0], p.adust, p.Tsub, p.BCvLv
    )

cdef double TProfile_Nenkova(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil:
    # return TProfile_Nenkova_max(R, Rsub_Nenkova((<par_nenkova *>par).L_bol[0], 0.05, 1500., 10.,))
    return TProfile_Nenkova_max(R/((<par_nenkova *>par).Rsub))
    

cdef double TProfile_Nenkova_max(double rho) noexcept nogil:
    # Branchless version
    # rho = R/Rsub, not R_xy.
    cdef double logrho = log(rho)
    cdef double inner = 7.3132203870903014340319874795174 - 0.39*logrho    # 1500.*pow(rho,-0.39)
    cdef double outer = 7.4502192361550161659074612779620 - 0.45*logrho    # 1720.2402428942632*pow(rho,-0.45)
    return inner if (logrho<=2.1972245773362193827904904738451) else outer # return inner if (rho<=9.) else outer
    
cdef double TProfile_Nenkova_min(double rho) noexcept nogil:
    return 5.9914645471079819868704471522851 - 0.42*log(rho) # return 400*pow(rho, -0.42)

## Hybrid
################################################################################
### Nenkova with Cutoff

cdef void TPrep_NenkovaC(void * par) noexcept nogil:
    cdef par_nenkovac * p = <par_nenkovac*>(par)
    p.Rsub = p.fRsub(
        p.logLbol[0], p.adust, p.Tsub, p.BCvLv
    )
    p.r_in = ((<par_nenkovac*>par).r_ISCO)*((<par_nenkovac*>par).Rg[0])
    p.factor = ((<par_nenkovac*>par).f_cutoff)*sqrt(p.r_in)

cdef double TProfile_NenkovaC(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil:
    # return TProfile_Nenkova_max(R, Rsub_Nenkova((<par_nenkova *>par).L_bol[0], 0.05, 1500., 10.,))
    cdef double TNen = TProfile_Nenkova_max(R/((<par_nenkovac *>par).Rsub))
    return TNen + log(1.-((<par_nenkovac *>par).factor)/sqrt(R)) if R > (<par_nenkovac *>par).r_in else -INFINITY
    
# cdef double TProfile_Hyb_Starkey_Nenkova_r3d(double R, double * xyz, double * normal, double fgeom, void * par) noexcept nogil:

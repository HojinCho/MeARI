cdef enum DiskGeometryModel:
    # Thin Disk Models using Goad et al. Parameterizations
    THIN_GOAD_LOG_HEXA
    THIN_GOAD_LOG_RECT
    THIN_GOAD_LIN_HEXA
    THIN_GOAD_LIN_RECT
    # Slim Disk Models (without any tied parameter to the radiation)
    SLIM_FREE_LOG_HEXA
    SLIM_FREE_LOG_RECT
    SLIM_FREE_LIN_HEXA
    SLIM_FREE_LIN_RECT
    # Slim Disk Models (with tied parameters to radiation)
    SLIM_RADI_LOG_HEXA
    SLIM_RADI_LOG_RECT
    SLIM_RADI_LIN_HEXA
    SLIM_RADI_LIN_RECT


ctypedef void (*func_t_diskgeom)(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double curv,
    double * azm1d, double * rad1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil

cdef void assign_azimuthal(long nazm, double *azm) noexcept nogil
cdef void assign_radial_lin(long nrad, double rad_in, double rad_out, double *rad) noexcept nogil
cdef void assign_radial_log(long nrad, double rad_in, double rad_out, double *rad) noexcept nogil
cdef void __compute_norm(long nsize, double * vecarr, double * outarr) noexcept nogil
cdef void __compute_rho_and_R(long nsize, double * xyz, double * rhoarr, double * Rarr) noexcept nogil
cdef void compute_lamppost_distance(long nsize, double * outdist, double * xyz0, double * rho, double H_lamp) noexcept nogil
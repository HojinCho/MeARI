cdef void assign_slim_free_log_hexa(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double p, 
    double * azm1d, double * R1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil
cdef void assign_slim_free_log_rect(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double p, 
    double * azm1d, double * R1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil
cdef void assign_slim_free_lin_hexa(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double p, 
    double * azm1d, double * R1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil
cdef void assign_slim_free_lin_rect(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double p, 
    double * azm1d, double * R1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil
cdef double slim_free_marginal_area( # Fixing Rout and finding the incremental area at Rout (or decreasing).
    double Rout, double cosi, 
    double c_f, double p,
) noexcept nogil
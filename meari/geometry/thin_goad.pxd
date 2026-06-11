cdef void assign_thin_goad_log_hexa(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double alpha, 
    double * azm1d, double * rho1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil
cdef void assign_thin_goad_log_rect(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double alpha, 
    double * azm1d, double * rho1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil
cdef void assign_thin_goad_lin_hexa(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double alpha, 
    double * azm1d, double * rho1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil
cdef void assign_thin_goad_lin_rect(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double alpha, 
    double * azm1d, double * rho1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil
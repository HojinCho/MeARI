# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython
from cython.parallel import prange

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, exp, cos, sin, pow,
    acos, asin,
    fmin, fmax, fabs,
)

from .diskgeom cimport assign_radial_lin, assign_radial_log, __compute_rho_and_R


cdef void __coord_slim_free_rect(
    long nrad, long nazm, double * R1d, double * azm, double * z1d, 
    double * coords,
) noexcept nogil:
    cdef long i, j, J
    cdef double c, s, rho
    for j in prange(nazm, nogil=True):
        c = cos(azm[j])
        s = sin(azm[j])
        J = j*(nrad+1)
        for i in range(nrad+1):
            rho = sqrt(R1d[i]*R1d[i] - z1d[i]*z1d[i])
            coords[3*(J+i)    ] = rho*c
            coords[3*(J+i) + 1] = rho*s
            coords[3*(J+i) + 2] = z1d[i]

# cdef void __coord_slim_free_hexa(
#     long nrad, long nazm, double * R1d, double * azm, double * z1d, 
#     double * coords,
# ) noexcept nogil:
#     cdef long i, j, J
#     cdef double rho, delta
#     cdef double dazm = -0.5*(azm[1] - azm[0])
#     for i in prange(nrad+1, nogil=True):
#         rho = sqrt(R1d[i]*R1d[i] - z1d[i]*z1d[i])
#         delta = i*dazm
#         for j in range(nazm):
#             J = 3*(j*(nrad+1)+i)
#             coords[J    ] = rho*cos(azm[j]+delta)
#             coords[J + 1] = rho*sin(azm[j]+delta)
#             coords[J + 2] = z1d[i]

cdef void __coord_slim_free_hexa(
    long nrad, long nazm, double * R1d, double * azm, double * z1d, 
    double * coords,
) noexcept nogil:
    cdef long i, j, J
    cdef double rho, delta
    cdef double dazm = -0.5*(azm[1] - azm[0])
    for i in prange(nrad+1, nogil=True):
        rho = sqrt(R1d[i]*R1d[i] - z1d[i]*z1d[i])
        delta = i*dazm
        for j in range(nazm):
            J = 3*(j*(nrad+1)+i)
            coords[J    ] = rho*cos(azm[j]+delta)
            coords[J + 1] = rho*sin(azm[j]+delta)
            coords[J + 2] = z1d[i]

cdef void __slim_free_normals(
    long nv, double alpha, 
    double * R,
    double * xyz, 
    double * normal,
) noexcept nogil:
    cdef long i, I
    cdef double t, denom
    for i in prange(nv, nogil=True):
        I = 3*i
        t = xyz[I+2]/R[i]
        denom = 1./sqrt(1. - alpha*(2. - alpha)*t*t)
        normal[I+2] = (1-alpha*t*t)*denom
        denom = - denom*alpha*t/(R[i])
        normal[I  ] = denom*xyz[I  ]
        normal[I+1] = denom*xyz[I+1]

cdef void assign_slim_free_log_hexa(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double p, 
    double * azm1d, double * R1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil:
    cdef double alpha = 0.5*(3.-p)
    cdef double beta = c_f*pow(R_out, 1.-alpha)
    rho_out[0] = R_out*sqrt(1. - c_f*c_f)
    assign_radial_log(nrad, R_in, R_out, R1d)
    for i in prange(nrad+1, nogil=True):
        z1d[i] = beta*pow(R1d[i], alpha)
    __coord_slim_free_hexa( nrad, nazm, R1d, azm1d, z1d, xyz)
    __compute_rho_and_R(nv, xyz, rho, R)
    __slim_free_normals(nv, alpha, R, xyz, normal)
    

cdef void assign_slim_free_log_rect(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double p, 
    double * azm1d, double * R1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil:
    cdef double alpha = 0.5*(3.-p)
    cdef double beta = c_f*pow(R_out, 1.-alpha)
    rho_out[0] = R_out*sqrt(1. - c_f*c_f)
    assign_radial_log(nrad, R_in, R_out, R1d)
    for i in prange(nrad+1, nogil=True):
        z1d[i] = beta*pow(R1d[i], alpha)
    __coord_slim_free_rect( nrad, nazm, R1d, azm1d, z1d, xyz)
    __compute_rho_and_R(nv, xyz, rho, R)
    __slim_free_normals(nv, alpha, R, xyz, normal)

cdef void assign_slim_free_lin_hexa(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double p, 
    double * azm1d, double * R1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil:
    cdef double alpha = 0.5*(3.-p)
    cdef double beta = c_f*pow(R_out, 1.-alpha)
    rho_out[0] = R_out*sqrt(1. - c_f*c_f)
    assign_radial_lin(nrad, R_in, R_out, R1d)
    for i in prange(nrad+1, nogil=True):
        z1d[i] = beta*pow(R1d[i], alpha)
    __coord_slim_free_hexa( nrad, nazm, R1d, azm1d, z1d, xyz)
    __compute_rho_and_R(nv, xyz, rho, R)
    __slim_free_normals(nv, alpha, R, xyz, normal)

cdef void assign_slim_free_lin_rect(
    # Outputs
    double * rho_out, double * rho, double * R, double * xyz, double * normal,
    # Inputs
    long nazm, long nrad, long nv, double R_out, double R_in, double c_f, double p, 
    double * azm1d, double * R1d, double * z1d, 
    void * geom_model_packet,
) noexcept nogil:
    cdef double alpha = 0.5*(3.-p)
    cdef double beta = c_f*pow(R_out, 1.-alpha)
    rho_out[0] = R_out*sqrt(1. - c_f*c_f)
    assign_radial_lin(nrad, R_in, R_out, R1d)
    for i in prange(nrad+1, nogil=True):
        z1d[i] = beta*pow(R1d[i], alpha)
    __coord_slim_free_rect( nrad, nazm, R1d, azm1d, z1d, xyz)
    __compute_rho_and_R(nv, xyz, rho, R)
    __slim_free_normals(nv, alpha, R, xyz, normal)



cdef double slim_free_marginal_area( # Fixing Rout and finding the incremental area at Rout (or decreasing).
    double Rout, double cosi, 
    double c_f, double p,
) noexcept nogil:
    cdef double alpha, rhoout, zout, half_theta, dz, drho, dl, term
    alpha = 0.5*(3.-p)
    zout = c_f*Rout
    rhoout = sqrt(Rout*Rout - zout*zout)

    dz = alpha*c_f # d R^alpha / dR  = alpha*R^(alpha-1)
    drho = (Rout - zout*dz)/rhoout # d sqrt(r*r - z*z) /dr = (r - z*dz)/sqrt(r*r - z*z) = (r - z*alpha*t)/rho
    dl = sqrt(drho*drho + dz*dz)
    term = Rout*(1-alpha*c_f*c_f)*cosi/ (sqrt(1-cosi*cosi)*rhoout*alpha*c_f) # mostly positive.
    half_theta = M_PI - acos(term) if fabs(term) <= 1 else (M_PI if term > 1 else 0.)
    return 2*half_theta*rhoout*dl
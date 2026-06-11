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


from ..utils.algorithms cimport find_root

from .diskgeom cimport assign_radial_lin, assign_radial_log, __compute_rho_and_R

cdef double[3] unit_z = [0., 0., 1.]
cdef struct par_radial:
    double alpha
    double beta
    double R

# Computing function for single values

cdef double __z_bowl_planar(double rho, double alpha, double beta) noexcept nogil:
    return beta*pow(rho,alpha)

cdef void __z_bowl_planar_arr(long n, double * rho, double alpha, double beta, double * outarr) noexcept nogil:
    cdef long i
    for i in range(n):
        outarr[i] = __z_bowl_planar(rho[i], alpha, beta)

cdef double __R_bowl_planar(double rho, double alpha, double beta) noexcept nogil:
    cdef double z = __z_bowl_planar(rho, alpha, beta) # height.
    return sqrt(z*z + rho*rho)

cdef double __radial_root_finding_function(double rho, void *params) noexcept nogil:
    cdef par_radial *p = <par_radial *> params
    return __R_bowl_planar(rho, p.alpha, p.beta) - p.R
     
cdef double __radial_bowl_planar(double R, double alpha, double beta) noexcept nogil:
    cdef double minimum_bound = fmin(0.01, (0.33)*sqrt(fmax(2.-2.*R*pow(beta, (1./(alpha - 1))), 0))) # radians
    cdef par_radial *params = <par_radial *>malloc(sizeof(par_radial))
    cdef double radial
    params.alpha = alpha
    params.beta = beta
    params.R = R
    radial = find_root(__radial_root_finding_function, <void *>params, R*minimum_bound, 1.2*R)
    free(params)
    return radial

cdef double __radial_conic_planar(double R, double beta) noexcept nogil:
    return R/sqrt(1+beta*beta)

cdef void __thin_normals(
    long nv, double alpha, 
    double * rho,
    double * xyz, 
    double * normal,
) noexcept nogil:
    cdef long i, I
    cdef double norm
    for i in prange(nv, nogil=True):
        I = 3*i
        norm = alpha*xyz[I+2]/rho[i] # is never 0, as long as rho[i]>0
        norm = 1./sqrt(1. + norm*norm)
        normal[I+2] = norm
        norm = -norm*alpha*xyz[I+2]/(rho[i]*rho[i])
        normal[I  ] = norm*xyz[I  ]
        normal[I+1] = norm*xyz[I+1]

# cdef void __thin_normals_curv(
#     long nv, double alpha, 
#     double * rho,
#     double * xyz, 
#     double * normal,
# ) noexcept nogil:
#     cdef long i, I
#     cdef double nz, normfact
#     for i in prange(nv, nogil=True):
#         I = 3*i
#         nz = rho[i]/(alpha*xyz[I+2])
#         normfact = 1/sqrt(1+nz*nz)
#         normal[I+2] =  normfact*nz
#         normfact = (-normfact/rho[i])
#         normal[I  ] = normfact*xyz[I  ]
#         normal[I+1] = normfact*xyz[I+1]
        
# cdef void __thin_normals_flat(long nv, double * normal) noexcept nogil:
#     cdef long i
#     cdef size_t s = 3*sizeof(double)
#     for i in prange(nv, nogil=True): # True Parallel
#         memcpy(<void *>(normal+3*i), <void *>(unit_z), s)


cdef void __coord_thin_rect(
    long nrad, long nazm, double * rho, double * azm, double * z, 
    double * coords,
) noexcept nogil:
    cdef long i, j, J
    cdef double c, s
    for j in prange(nazm, nogil=True):
        c = cos(azm[j])
        s = sin(azm[j])
        J = j*(nrad+1)
        for i in range(nrad+1):
            coords[3*(J+i)    ] = rho[i]*c
            coords[3*(J+i) + 1] = rho[i]*s
            coords[3*(J+i) + 2] = z[i]

cdef void __coord_thin_hexa(
    long nrad, long nazm, double * rho, double * azm, double * z, 
    double * coords,
) noexcept nogil:
    cdef long i, j, J
    cdef double c, s
    cdef double dazm = -0.5*(azm[1] - azm[0])
    for j in prange(nazm, nogil=True):
        J = j*(nrad+1)
        for i in range(nrad+1):
            coords[3*(J+i)    ] = rho[i]*cos(azm[j]+i*dazm)
            coords[3*(J+i) + 1] = rho[i]*sin(azm[j]+i*dazm)
            coords[3*(J+i) + 2] = z[i]

cdef void assign_thin_goad_log_hexa(
    # Outputs
    # scalar
    double * rho_out, 
    # 1-d vectors
    double * rho, double * R,
    # 2-d vectors
    double * xyz, double * normal,
    # Inputs
    # Scalars
    long nazm, long nrad, long nv, # since it is already computed, why not.
    double R_out, double R_in, double c_f, double alpha, 
    # Vectors (but note that azm1d is immutable, while rest 2 are buffers)
    # in general, it is always azm first, and two more 1-d buffers.
    # Naming convention can be different from function to function.
    double * azm1d, 
    double * rho1d, double * z1d, 
    # Here packs the pointer for the functions and so on.
    void * additional_data, 
) noexcept nogil:
    rho_out[0] = R_out*sqrt(1. - c_f*c_f)
    cdef double beta = R_out*c_f/pow(rho_out[0], alpha)
    cdef double rho_in
    if alpha == 1:
        rho_in  = __radial_conic_planar(R_in, beta)
    else:
        rho_in  = __radial_bowl_planar(R_in, alpha, beta)
    assign_radial_log(nrad, rho_in, rho_out[0], rho1d)
    __z_bowl_planar_arr(  nrad+1, rho1d, alpha, beta, z1d)
    __coord_thin_hexa( nrad, nazm, rho1d, azm1d, z1d, xyz)
    __compute_rho_and_R(nv, xyz, rho, R)
    __thin_normals(nv, alpha, rho, xyz, normal)

cdef void assign_thin_goad_log_rect(
    # Outputs
    # scalar
    double * rho_out, 
    # 1-d vectors
    double * rho, double * R,
    # 2-d vectors
    double * xyz, double * normal,
    # Inputs
    # Scalars
    long nazm, long nrad, long nv, # since it is already computed, why not.
    double R_out, double R_in, double c_f, double alpha, 
    # Vectors (but note that azm1d is immutable, while rest 2 are buffers)
    # in general, it is always azm first, and two more 1-d buffers.
    # Naming convention can be different from function to function.
    double * azm1d, 
    double * rho1d, double * z1d, 
    # Here packs the pointer for the functions and so on.
    void * additional_data, 
) noexcept nogil:
    rho_out[0] = R_out*sqrt(1. - c_f*c_f)
    cdef double beta = R_out*c_f/pow(rho_out[0], alpha)
    cdef double rho_in
    if alpha == 1:
        rho_in  = __radial_conic_planar(R_in, beta)
    else:
        rho_in  = __radial_bowl_planar(R_in, alpha, beta)
    assign_radial_log(nrad, rho_in, rho_out[0], rho1d)
    __z_bowl_planar_arr(  nrad+1, rho1d, alpha, beta, z1d)
    __coord_thin_rect(nrad, nazm, rho1d, azm1d, z1d, xyz)
    __compute_rho_and_R(nv, xyz, rho, R)
    __thin_normals(nv, alpha, rho, xyz, normal)
    
cdef void assign_thin_goad_lin_hexa(
    # Outputs
    # scalar
    double * rho_out, 
    # 1-d vectors
    double * rho, double * R,
    # 2-d vectors
    double * xyz, double * normal,
    # Inputs
    # Scalars
    long nazm, long nrad, long nv, # since it is already computed, why not.
    double R_out, double R_in, double c_f, double alpha, 
    # Vectors (but note that azm1d is immutable, while rest 2 are buffers)
    # in general, it is always azm first, and two more 1-d buffers.
    # Naming convention can be different from function to function.
    double * azm1d, 
    double * rho1d, double * z1d, 
    # Here packs the pointer for the functions and so on.
    void * additional_data, 
) noexcept nogil:
    rho_out[0] = R_out*sqrt(1. - c_f*c_f)
    cdef double beta = R_out*c_f/pow(rho_out[0], alpha)
    cdef double rho_in
    if alpha == 1:
        rho_in  = __radial_conic_planar(R_in, beta)
    else:
        rho_in  = __radial_bowl_planar(R_in, alpha, beta)
    assign_radial_lin(nrad, rho_in, rho_out[0], rho1d)
    __z_bowl_planar_arr(  nrad+1, rho1d, alpha, beta, z1d)
    __coord_thin_hexa( nrad, nazm, rho1d, azm1d, z1d, xyz)
    __compute_rho_and_R(nv, xyz, rho, R)
    __thin_normals(nv, alpha, rho, xyz, normal)
    
cdef void assign_thin_goad_lin_rect(
    # Outputs
    # scalar
    double * rho_out, 
    # 1-d vectors
    double * rho, double * R,
    # 2-d vectors
    double * xyz, double * normal,
    # Inputs
    # Scalars
    long nazm, long nrad, long nv, # since it is already computed, why not.
    double R_out, double R_in, double c_f, double alpha, 
    # Vectors (but note that azm1d is immutable, while rest 2 are buffers)
    # in general, it is always azm first, and two more 1-d buffers.
    # Naming convention can be different from function to function.
    double * azm1d, 
    double * rho1d, double * z1d, 
    # Here packs the pointer for the functions and so on.
    void * additional_data, 
) noexcept nogil:
    rho_out[0] = R_out*sqrt(1. - c_f*c_f)
    cdef double beta = R_out*c_f/pow(rho_out[0], alpha)
    cdef double rho_in
    if alpha == 1:
        rho_in  = __radial_conic_planar(R_in, beta)
    else:
        rho_in  = __radial_bowl_planar(R_in, alpha, beta)
    assign_radial_lin(nrad, rho_in, rho_out[0], rho1d)
    __z_bowl_planar_arr(  nrad+1, rho1d, alpha, beta, z1d)
    __coord_thin_rect(nrad, nazm, rho1d, azm1d, z1d, xyz)
    
    __compute_rho_and_R(nv, xyz, rho, R)
    # It should also compute vertex normals, given it is model-dependent.
    __thin_normals(nv, alpha, rho, xyz, normal)
    
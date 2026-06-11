# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy, memset

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, exp, cos, sin, pow,
    acos, asin,
    fmin, fmax, fabs,
)

from ..utils.algorithms cimport (
    kahan_sum_iterator, 
    mergesort_ptr, mergesort_ptr_buf, 
    countingsort_ptr, countingsort_uchar_ptr,
    count_each_items_ptr, count_each_items_uchar_ptr,
    flip_index, flip_index_b,
    assign_sorted_vector_inplace_lng_ptr,
    assign_sorted_scalar_inplace_lng_ptr,
    assign_sorted_vector_inplace_dbl_ptr,
    assign_sorted_scalar_inplace_dbl_ptr,
    assign_sorted_scalar_inplace_uchar_ptr,
    assign_sorted_scalar_dbl_ptr,
    assign_sorted_vector_dbl_ptr,
)

from ..extern.lia cimport ( # BLAS and LAPACK
    # Unary
    lia_v_norm, lia_v_norm_A_i, lia_v_comp,
    lia3d_v_rho_R,
    # Binary
    lia_sv_scalar_inplace, lia_sv_scalar_A_ii_inplace, lia_sv_scalar, lia_sv_scalar_A_ii,
    lia_vv_dot, lia_vv_dot_A_0i, lia_vv_dot_A_ii, lia_vv_cpy, lia_vv_cpy_A_ii, lia_vv_swp, lia_vv_swp_A_ii,
    lia_vv_add_inplace, lia_vv_add, lia_vv_add_A_i0_inplace, lia_vv_add_A_i0, lia_vv_add_A_ii_inplace, lia_vv_add_A_ii,
    lia_vv_sub_inplace, lia_vv_sub, lia_vv_sub_A_i0_inplace, lia_vv_sub_A_i0, lia_vv_sub_A_ii_inplace, lia_vv_sub_A_ii,
    lia_vv_mul_inplace, lia_vv_mul, lia_vv_mul_A_i0_inplace, lia_vv_mul_A_i0, lia_vv_mul_A_ii_inplace, lia_vv_mul_A_ii,
    lia_vv_div_inplace, lia_vv_div, lia_vv_div_A_i0_inplace, lia_vv_div_A_i0, lia_vv_div_A_ii_inplace, lia_vv_div_A_ii,
    lia_mv_mul, lia_mv_mul_A_0i, lia_mv_mul_A_ii,
    lia2d_mv_rot_inplace, lia2d_mv_rot_A_0i_inplace, lia2d_mv_rot, lia2d_mv_rot_A_0i, lia3d_mv_rot_inplace, lia3d_mv_rot_A_0i_inplace, lia3d_mv_rot, lia3d_mv_rot_A_0i,
    # Ternary
    lia_vmv_quadform_bf, lia_vmv_quadform_A_i0i_bf, lia_vmv_quadform_A_iii_bf, 
    lia_vmv_quadform, lia_vmv_quadform_A_i0i, lia_vmv_quadform_A_iii, 
    # Quarternary 
    lia_svsv_lincomb, lia_svsv_lincomb_A_0i0i, lia_svsv_lincomb_A_iiii,
)

# TODO: Change long that are used for indices to size_t.

from ..utils.types cimport Numeric, Integer, Real, func_t_double_ternary

cdef void quad_annulus_connectivity(
    long nrad, long nazm, long * fvi, 
    long * vfi, long * vfn,
) noexcept nogil:
    # Disk or one side of torus have annulus fviectivity
    cdef long i, j, k, nv
    cdef long I, J, K
    cdef long nvpf = 4
    cdef long nfpv = 4
    nv = (nazm-1)*nrad
    for i in prange((nrad+1)*nazm, nogil=True): # MUST INITIALIZE!
        vfn[i] = 0
    for i in prange(nv, nogil=True):
        j = i + (i // nrad)
        k = j + nrad + 1
        I = nvpf*i
        J = nfpv*j
        K = nfpv*k
        # Concave ordering: Bowl is concave
        fvi[I  ] = j       # p1 = i * nrad + j
        fvi[I+1] = j + 1   # p2 = (i + 1) * nrad + j
        fvi[I+2] = k + 1   # p3 = (i + 1) * nrad + j + 1
        fvi[I+3] = k       # p4 = i * nrad + j + 1
        vfi[J      + vfn[j  ]] = i
        vfi[J+nfpv + vfn[j+1]] = i
        vfi[K+nfpv + vfn[k+1]] = i
        vfi[K      + vfn[k  ]] = i
        vfn[j  ] += 1
        vfn[j+1] += 1
        vfn[k+1] += 1
        vfn[k  ] += 1

    for k in prange(nrad, nogil=True): # Stitch the last row to the first row to make annulus topology
        j = (nazm-1)*(nrad+1) + k
        i = k + nv
        I = nvpf*i
        J = nfpv*j
        K = nfpv*k
        # Concave ordering: Bowl is concave
        fvi[I  ] = j       # p1 = i * nrad + j
        fvi[I+1] = j + 1   # p2 = (i + 1) * nrad + j
        fvi[I+2] = k + 1   # p3 = (i + 1) * nrad + j + 1
        fvi[I+3] = k       # p4 = i * nrad + j + 1
        vfi[J      + vfn[j  ]] = i
        vfi[J+nfpv + vfn[j+1]] = i
        vfi[K+nfpv + vfn[k+1]] = i
        vfi[K      + vfn[k  ]] = i
        vfn[j  ] += 1
        vfn[j+1] += 1
        vfn[k+1] += 1
        vfn[k  ] += 1

cdef void tri_annulus_connectivity(
    long nrad, long nazm, long * fvi, 
    long * vfi, long * vfn,
) noexcept nogil:
    # Disk or one side of torus have annulus fviectivity
    cdef long i, j, k, l, n
    cdef long I, J, K, L
    cdef long nvpf = 3
    cdef long nfpv = 6
    for i in prange((nrad+1)*nazm, nogil=True): # MUST INITIALIZE!
        vfn[i] = 0
    n = (nazm-1)*nrad
    for i in prange(n, nogil=True):
        j = i + (i // nrad)
        k = j + nrad + 1
        l = 2*i
        L = nvpf*l
        J = nfpv*j
        K = nfpv*k
        # Concave ordering: Bowl is concave
        # Lower Triangle
        fvi[L  ] = j            # p1 = i * nrad + j
        fvi[L+1] = j + 1        # p2 = (i + 1) * nrad + j
        fvi[L+2] = k + 1        # p3 = (i + 1) * nrad + j + 1
        # Upper Triangle
        fvi[L+3] = j            # p1 = i * nrad + j
        fvi[L+4] = k + 1        # p3 = (i + 1) * nrad + j + 1
        fvi[L+5] = k            # p4 = i * nrad + j + 1
        vfi[J      + vfn[j  ]]     = l
        vfi[J+nfpv + vfn[j+1]]     = l
        vfi[K+nfpv + vfn[k+1]]     = l
        vfi[J      + vfn[j  ] + 1] = l+1
        vfi[K+nfpv + vfn[k+1] + 1] = l+1
        vfi[K      + vfn[k  ]]     = l+1
        vfn[j  ] += 2
        vfn[j+1] += 1
        vfn[k+1] += 2
        vfn[k  ] += 1

    for k in prange(nrad, nogil=True): # Stitch the last row to the first row to make annulus topology
        j = (nazm-1)*(nrad+1) + k
        i = k + n
        l = 2*i
        L = nvpf*l
        J = nfpv*j
        K = nfpv*k
        # Concave ordering: Bowl is concave
        # Lower Triangle
        fvi[L  ] = j            # p1 = i * nrad + j
        fvi[L+1] = j + 1        # p2 = (i + 1) * nrad + j
        fvi[L+2] = k + 1        # p3 = (i + 1) * nrad + j + 1
        # Upper Triangle
        fvi[L+3] = j            # p1 = i * nrad + j
        fvi[L+4] = k + 1        # p3 = (i + 1) * nrad + j + 1
        fvi[L+5] = k            # p4 = i * nrad + j + 1
        vfi[J      + vfn[j  ]]     = l
        vfi[J+nfpv + vfn[j+1]]     = l
        vfi[K+nfpv + vfn[k+1]]     = l
        vfi[J      + vfn[j  ] + 1] = l+1
        vfi[K+nfpv + vfn[k+1] + 1] = l+1
        vfi[K      + vfn[k  ]]     = l+1
        vfn[j  ] += 2
        vfn[j+1] += 1
        vfn[k+1] += 2
        vfn[k  ] += 1

cdef double __triangle_area(
    double ux, double uy, double uz,
    double vx, double vy, double vz,
) noexcept nogil:
    cdef double nx = uy*vz - uz*vy
    cdef double ny = uz*vx - ux*vz
    cdef double nz = ux*vy - uy*vx
    return 0.5*sqrt(nx*nx + ny*ny + nz*nz)

cdef void __assign_position_difference_vectors(
    long nf, double * coords, long * fvi, double * vecarr
) noexcept nogil: # True parallel
    # This is for triangular mesh
    cdef long i, I9, I6, p0, p1, p2, p5
    for i in prange(nf//2, nogil=True):
        I9 = 9*i
        I6 = 6*i
        p0 = fvi[I6  ]*3
        p1 = fvi[I6+1]*3
        p2 = fvi[I6+2]*3
        p5 = fvi[I6+5]*3
        vecarr[I9     + 0] = coords[p1+0] - coords[p0+0]
        vecarr[I9     + 1] = coords[p1+1] - coords[p0+1]
        vecarr[I9     + 2] = coords[p1+2] - coords[p0+2]
        vecarr[I9 + 3 + 0] = coords[p2+0] - coords[p0+0]
        vecarr[I9 + 3 + 1] = coords[p2+1] - coords[p0+1]
        vecarr[I9 + 3 + 2] = coords[p2+2] - coords[p0+2]
        vecarr[I9 + 6 + 0] = coords[p5+0] - coords[p0+0]
        vecarr[I9 + 6 + 1] = coords[p5+1] - coords[p0+1]
        vecarr[I9 + 6 + 2] = coords[p5+2] - coords[p0+2]

cdef void __assign_triangle_area(
    long nf, double * vecarr, double * outarr
) noexcept nogil: # True parallel
    cdef long i, I, J, I2
    for i in prange(nf//2, nogil=True): # USE BLAS - LEVEL 1?
        I = 9*i
        J = 9*i + 3
        I2 = 2*i
        outarr[I2] = __triangle_area(
            vecarr[I  ], vecarr[I+1], vecarr[I+2],
            vecarr[J  ], vecarr[J+1], vecarr[J+2],
        ) 
        I = 9*i + 6
        outarr[I2+1] = __triangle_area(
            vecarr[J  ], vecarr[J+1], vecarr[J+2],
            vecarr[I  ], vecarr[I+1], vecarr[I+2],
        )

cdef void __check_region(
    long nv, long * vfi, long * vfn,
    double * test_scalar, double threshold, double direction,
    unsigned char * out_bool_points, unsigned char * out_cell_n_good_points,
):
    cdef long i, j, I
    for i in range(nv):
        if out_bool_points[i] == 0:
            continue
        if (test_scalar[i]-threshold) * direction < 0:
            I = 6*i
            out_bool_points[i] = 0
            for j in range(vfn[i]):
                out_cell_n_good_points[vfi[I+j]] -= 1

# cdef void __filter_by_mask( # Old version.
#     long nv, long nf, 
#     long * fvi, long * vfi, long * vfn,
#     double * mask,
#     unsigned char * out_bool_points, unsigned char * out_cell_n_good_points,
# ) noexcept nogil:
#     cdef long i, j, I
#     for i in range(nv):
#         if out_bool_points[i] == 0:
#             continue
#         if mask[i] < 0:
#             I = 6*i
#             out_bool_points[i] = 0
#             for j in range(vfn[i]):
#                 out_cell_n_good_points[vfi[I+j]] -= 1

cdef void __filter_by_mask( # Serial Version.
    long nv, long nf, 
    long * fvi, long * vfi, long * vfn,
    double * mask,
    unsigned char * out_bool_points, # Either 1 or 0
    unsigned char * out_cell_n_good_points, # 0, 1, 2, or 3.
) noexcept nogil:
    cdef long i, j
    cdef long * id_ptr
    for i in range(nv):
        if out_bool_points[i] == 0:
            continue
        if mask[i] < 0:
            out_bool_points[i] = 0
            id_ptr = vfi + 6*i
            for j in range(vfn[i]):
                out_cell_n_good_points[id_ptr[j]] -= 1

# cdef void __filter_by_mask( # Parallel Version, this is actually slower.
#     long nv, long nf, 
#     long * fvi, long * vfi, long * vfn,
#     double * mask,
#     unsigned char * out_bool_points, # Either 1 or 0
#     unsigned char * out_cell_n_good_points, # 0, 1, 2, or 3.
# ) noexcept nogil:
#     cdef long i
#     cdef long * id_ptr
#     for i in prange(nv, nogil=True):
#         out_bool_points[i] = 0 if mask[i] < 0 else 1
#     for i in prange(nf, nogil=True):
#         id_ptr = fvi + 3*i
#         out_cell_n_good_points[i] = (
#             out_bool_points[id_ptr[0]]
#           + out_bool_points[id_ptr[1]]
#           + out_bool_points[id_ptr[2]]
#         )

cdef void __composite_mask(
    long nv, 
    double * inmask, double * scalar, double threshold, double direction,
    double * outmask,
) noexcept nogil:
    cdef long i
    for i in prange(nv, nogil=True):
        outmask[i] = fmin(inmask[i], (scalar[i]-threshold)*direction)
    
cdef void __composite_mask_inplace(
    long nv, 
    double * mask, double * scalar, double threshold, double direction,
) noexcept nogil:
    cdef long i
    for i in prange(nv, nogil=True):
        mask[i] = fmin(mask[i], (scalar[i]-threshold)*direction)

cdef void __replace_mask(
    long nv, 
    double * mask, double * scalar, double threshold, double direction,
) noexcept nogil:
    cdef long i
    for i in prange(nv, nogil=True):
        mask[i] = (scalar[i]-threshold)*direction

cdef void make_obscuration_mask(
    long nv, long nf, long nvpf,
    long * fvi, long * vfi, long * vfn, 
    double * obsc_mask, double * normal_z, 
    unsigned char * mask_v, unsigned char * mask_f, double * mask_obs,
) noexcept nogil:
    cdef long i, j
    memset(<void *> mask_v,    1, nv*sizeof(unsigned char))
    memset(<void *> mask_f, nvpf, nf*sizeof(unsigned char))
    # Filter by Obscuration
    __replace_mask(nv, mask_obs, obsc_mask, 0, -1,)
    __filter_by_mask(nv, nf, fvi, vfi, vfn, mask_obs, mask_v, mask_f,)
    # Filter by Normal vector z-direction sign
    __composite_mask_inplace(nv, mask_obs, normal_z, 0, 1,)
    __filter_by_mask(nv, nf, fvi, vfi, vfn, mask_obs, mask_v, mask_f,)



cdef void __interpolate_to_zero(
    double x1, double y1, double z1, double f1, double m1,
    double x2, double y2, double z2, double f2, double m2,
    double *x, double *y, double *z, double *f,
) noexcept nogil:
    cdef double lam
    lam = - m1/(m2-m1)
    x[0] = x1 + lam*(x2-x1)
    y[0] = y1 + lam*(y2-y1)
    z[0] = z1 + lam*(z2-z1)
    f[0] = f1 + lam*(f2-f1)

cdef double __barycentric_integration(double f1, double f2, double f3,) noexcept nogil:
    # Volume of truncated triangular prism. 
    # Should be modified accordingly when incorporating the curvature
    # Also: https://everything2.com/title/Numerical+Quadrature+Over+Triangles One-point quadrature.
    return (f1 + f2 + f3)/3 # Should be multiplied by area!

cdef double __integrate__normal(
    double x1, double y1, double z1, double f1,
    double x2, double y2, double z2, double f2,
    double x3, double y3, double z3, double f3,
    double area,
) noexcept nogil:
    return area*__barycentric_integration(f1, f2, f3)

cdef double __integrate_1_missing(
    double x1, double y1, double z1, double f1, double m1,
    double x2, double y2, double z2, double f2, double m2,
    double xm, double ym, double zm, double fm, double mm,
    double area,
) noexcept nogil:
    cdef double x3, y3, z3, f3, x4, y4, z4, f4, negarea
    __interpolate_to_zero(
        x1, y1, z1, f1, m1, 
        xm, ym, zm, fm, mm, 
        &x3, &y3, &z3, &f3,
    )
    __interpolate_to_zero(
        x2, y2, z2, f2, m2, 
        xm, ym, zm, fm, mm, 
        &x4, &y4, &z4, &f4,
    )

    negarea = __triangle_area((xm-x3), (ym-y3), (zm-z3), (xm-x4), (ym-y4), (zm-z4))
    return (__integrate__normal(x1, y1, z1, f1, x2, y2, z2, f2, xm, ym, zm, fm, area) 
          - __integrate__normal(xm, ym, zm, fm, x3, y3, z3, f3, x4, y4, z4, f4, negarea))

cdef double __integrate_2_missing(
    double xp, double yp, double zp, double fp, double mp,
    double x1, double y1, double z1, double f1, double m1,
    double x2, double y2, double z2, double f2, double m2,
) noexcept nogil:
    cdef double x3, y3, z3, f3, x4, y4, z4, f4, posarea
    __interpolate_to_zero(
        x1, y1, z1, f1, m1,
        xp, yp, zp, fp, mp,
        &x3, &y3, &z3, &f3,
    )
    __interpolate_to_zero(
        x2, y2, z2, f2, m2,
        xp, yp, zp, fp, mp,
        &x4, &y4, &z4, &f4,
    )

    posarea = __triangle_area((xp-x3), (yp-y3), (zp-z3), (xp-x4), (yp-y4), (zp-z4))
    return __integrate__normal(xp, yp, zp, fp, x3, y3, z3, f3, x4, y4, z4, f4, posarea)

cdef void __reorder_based_on_dblscalar(
    long nv, long nf, double * coord, long * fvi, 
    long * vfi, long * vfn,
    double * potential, long * idx,
    long * lb_nv, # buffer
) noexcept nogil:
    cdef long i
    # mergesort_ptr(potential, idx, 0, nv-1) # 20% faster than quicksort.
    # mergesort_ptr_buf_prealloc(potential, idx, 0, nv-1)
    mergesort_ptr_buf(potential, idx, 0, nv-1, lb_nv)
    _c_fvi_redirect(nv, nf, fvi, idx, lb_nv)
    assign_sorted_vector_inplace_dbl_ptr(nv, 3, idx, coord)
    assign_sorted_vector_inplace_lng_ptr(nv, 6, idx, vfi)
    assign_sorted_scalar_inplace_lng_ptr(nv, idx, vfn)
    
cdef void __two_point_interp(
    double x0, double y0, double z0, double m0,
    double x1, double y1, double z1, double m1,
    double x2, double y2, double z2, double m2,
    double * x, double * y, double * z,
) noexcept nogil:
    cdef double _x1, _y1, _z1, _x2, _y2, _z2, temp
    __interpolate_to_zero(
        x1, y1, z1, 1, m1,
        x0, y0, z0, 1, m0,
        &_x1, &_y1, &_z1, &temp,
    )
    __interpolate_to_zero(
        x2, y2, z2, 1, m2,
        x0, y0, z0, 1, m0,
        &_x2, &_y2, &_z2, &temp,
    )
    x[0] = 0.5*(_x1+_x2)
    y[0] = 0.5*(_y1+_y2)
    z[0] = 0.5*(_z1+_z2)

cdef long ___c_trim_mesh(
    long nv, long nf, 
    double * coords,        # nv, 3
    long * fvi,   # nf, 3
    long * vfi, # nv, 6
    long * vfn,    # nv
    double * areaarr,         # nf
    unsigned char * mask_v,   # nv
    unsigned char * mask_f,   # nf
    double * potential,       # nv
    long nf_good, long nf3, 
    long * n_interp_pairs,    # nv    # 1 or 2; interpolation scheme.
    long * idx_interp_pairs,  # nv, 3 # [i, 0] and [i, 1], [i, 0] and [i, 2] should be interpolated. [i,0] is shifted.
    double * interp_weight,   # nv, 2 # lam=-m1/(m2-m1) factor for each direction.
) noexcept nogil:
    cdef long i, j, k, i0, i1, i2, nreturn
    cdef long I, J, K, I0, I1, I2, NR2, NR3
    # Assume cells are sorted.
    for k in range(nf_good, nf): # Removing "Invalid" cells: All vertices are null.
        for j in range(3):
            i0 = fvi[3*k+j] # Current vertex: i0
            I0 = 6*i0
            if vfn[i0]>0: # i0 has more than 1 cell remaining.
                for i in range(vfn[i0]): # Find which index i corresponds to bad cell k.
                    if vfi[I0+i]==k: # id_member_cell[i0, i] points to the bad cell k.
                        i1=i # name i as i1
                        break
                for i in range(i1+1, vfn[i0]):
                    vfi[I0+i-1] = vfi[I0+i]
                vfi[I0+vfn[i0]-1] = -1
                vfn[i0] -= 1
    nreturn = 0
    for i in range(nv):
        I = 6*i
        NR3 = 3*nreturn
        if mask_v[i]==0 and vfn[i]>0: # Point is outside the region but in valid cells. Needs to be shifted.
            k = -1
            for j in range(vfn[i]): # First find a cell that can use 2-point interpolation.
                if mask_f[vfi[I+j]]==2: 
                    k = vfi[I+j]
                    K = 3*k
                    break
            if k==-1: # If no 2-point interpolation is possible, use 1-point interpolation.
                for k in range(vfn[i]):
                    if mask_f[vfi[I+k]]==1: 
                        i0 = vfi[I+k]
                        I0 = 3*i0
                        # Averaging would be better, but because of compilation issue, just use one cell.
                        for j in range(3):
                            if mask_v[fvi[I0+j]]==1:
                                n_interp_pairs[nreturn] = 1
                                idx_interp_pairs[NR3  ] = i
                                idx_interp_pairs[NR3+1] = fvi[I0+j]
                                nreturn += 1
                                break
                        break
            else: # 2-point interpolation candidate is found.
                for j in range(3):
                    if fvi[K+j]==i:
                        n_interp_pairs[nreturn] = 2
                        idx_interp_pairs[NR3  ] = i
                        idx_interp_pairs[NR3+1] = fvi[K+(j+1)%3]
                        idx_interp_pairs[NR3+2] = fvi[K+(j+2)%3]
                        nreturn += 1
                        break

    for k in prange(nreturn, nogil=True): # True parallel?
        K = 3*k
        if n_interp_pairs[k]==1:
            i  = idx_interp_pairs[K  ]
            i1 = idx_interp_pairs[K+1]
            I  = 3*i
            I1 = 3*i1
            __interpolate_to_zero(
                coords[I1  ], coords[I1+1], coords[I1+2], 1, potential[i1],
                coords[ I  ], coords[ I+1], coords[ I+2], 1, potential[ i],
                &coords[I  ], &coords[I+1], &coords[I+2], &potential[i],
            )
            interp_weight[2*k] = -potential[i]/(potential[i1]-potential[i])
            potential[i] = 0
            mask_v[i] = 1
            for j in range(vfn[i]):
                I0 = vfi[6*i+j]
                if mask_f[I0]<3:
                    mask_f[I0] +=1
        else: # n_inter_pairs[i]==2:
            i  = idx_interp_pairs[K  ]
            i1 = idx_interp_pairs[K+1]
            i2 = idx_interp_pairs[K+2]
            I  = 3*i
            I1 = 3*i1
            I2 = 3*i2
            __two_point_interp(
                coords[I   ], coords[I +1], coords[I +2], potential[i ],
                coords[I1  ], coords[I1+1], coords[I1+2], potential[i1],
                coords[I2  ], coords[I2+1], coords[I2+2], potential[i2],
                &coords[I  ], &coords[I+1], &coords[I+2],
            )
            interp_weight[2*k  ] = -0.5*potential[i]/(potential[i1]-potential[i])
            interp_weight[2*k+1] = -0.5*potential[i]/(potential[i2]-potential[i])
            potential[i] = 0
            mask_v[i] = 1
            for j in range(vfn[i]):
                I0 = vfi[6*i+j]
                if mask_f[I0]<3:
                    mask_f[I0] +=1

    for i in prange(nf3, nf_good, nogil=True): # True parallel
        I0 = fvi[3*i  ]*3
        I1 = fvi[3*i+1]*3
        I2 = fvi[3*i+2]*3
    # For all cells that "had" less than 3 valid verticies.
        areaarr[i] = __triangle_area(
            coords[I0  ]-coords[I1  ], 
            coords[I0+1]-coords[I1+1], 
            coords[I0+2]-coords[I1+2], 
            coords[I0  ]-coords[I2  ], 
            coords[I0+1]-coords[I2+1], 
            coords[I0+2]-coords[I2+2], 
        )
    return nreturn

cdef void _c_fvi_redirect(
    long nv, long nf, long * fvi, long * idx_v, 
    # buffer
    long * lb_nv, # nv
) noexcept nogil:
    cdef long i, I
    # cdef long * lb_nv = <long *> malloc(nv * sizeof(long)) # keep, or make long type buffer, size nv
    for i in prange(nv, nogil=True): # True parallel
        lb_nv[idx_v[i]] = i
    for i in prange(nf, nogil=True): # True parallel
        I = 3*i
        fvi[I  ] = lb_nv[fvi[I  ]]
        fvi[I+1] = lb_nv[fvi[I+1]]
        fvi[I+2] = lb_nv[fvi[I+2]]
    # free(lb_nv)

cdef void _c_vfi_redirect(
    long nv, long nf, long * vfi, long * vfn, long * idx_f, long * lb_nf,
) noexcept nogil:
    cdef long i, j, I
    for i in prange(nf, nogil=True): # True parallel
        lb_nf[idx_f[i]] = i
    for i in prange(nv, nogil=True): # True parallel
        I = 6*i
        for j in range(vfn[i]):
            vfi[I+j] = lb_nf[vfi[I+j]]
        for j in range(vfn[i], 6):
            vfi[I+j] = -1

cdef void assign_face_value_by_vert_values(
    long nv, long nf, func_t_double_ternary f,
    double * val_f, double * val_v, long * fvi,
) noexcept nogil:
    cdef long i, I
    for i in prange(nf, nogil=True):
        I = 3*i
        val_f[i] = f(
            val_v[fvi[I  ]],
            val_v[fvi[I+1]],
            val_v[fvi[I+2]],
        )

cdef void assign_faceval_from_vertval(
    double * faceval, long nf, 
    long ndim, long * fvi, double * vertval, double * areaarr, 
) noexcept nogil:
    cdef long i, i3
    cdef long I, I0, I1, I2
    cdef long j
    for i in prange(nf, nogil=True):
        i3  = 3*i
        I  = ndim*i
        I0 = ndim*fvi[i3  ]
        I1 = ndim*fvi[i3+1]
        I2 = ndim*fvi[i3+2]
        for j in range(ndim):
            faceval[I+j] = areaarr[i]*__barycentric_integration(
                vertval[I0+j],
                vertval[I1+j],
                vertval[I2+j],
            )

cdef void sort_faces_by_vert_val(
    long nv, long nf, 
    double * val_v, 
    func_t_double_ternary f, # double (double, double, double)
    long * fvi,              # nf, nvpf; F |-> V: Need to be sorted
    long * vfi,              # nv, nfpv; V |-> F: Need to be redirected
    long * vfn,              # nv; NO need to be sorted
    double * cell_areas,     # nf; Need to be sorted
    # buffer
    long * idx_f,            # nf; Sorting index for f
    long * lb_nf,            # nf; Sorting index for f
    # output
    double * val_f,          # return value
) noexcept nogil:
    cdef long i
    # cdef long * idx_f = <long *>malloc(nf*sizeof(long)) # keep, or make long type buffer, size nf
    assign_face_value_by_vert_values(nv, nf, f, val_f, val_v, fvi)
    for i in prange(nf, nogil=True):
        idx_f[i] = i
    # mergesort_ptr(val_f, idx_f, 0, nf-1) # 20% faster than quicksort.
    # mergesort_ptr_buf_prealloc(val_f, idx_f, 0, nf-1)
    mergesort_ptr_buf(val_f, idx_f, 0, nf-1, lb_nf)

    assign_sorted_vector_inplace_lng_ptr(nf, 3, idx_f, fvi)
    _c_vfi_redirect(nv, nf, vfi, vfn, idx_f, lb_nf)
    assign_sorted_scalar_inplace_dbl_ptr(nf, idx_f, cell_areas)
    assign_sorted_scalar_inplace_dbl_ptr(nf, idx_f, val_f)
    # free(idx_f)

cdef void _c_trim_mesh(
    long nv, long nf,
    double * coords_in,          # nv, 3 (x,y,z)
    long * fvi_in,     # nf, 3 (v1,v2,v3)
    long * vfi_in,   # nv, 6 (f1,f2,f3,f4,f5,f6) # meaningful at most vfn
    long * vfn_in,    # nv
    double * areaarr_in,         # nf
    unsigned char * mask_v,   # nv
    unsigned char * mask_f,   # nf
    double * mask_pot,  # nv
    # Buffers
    long * lb_nv,               # nv
    long * lb_nf,               # nf
    # # Buffers for masks
    # unsigned char * mask_v,   # nv
    # unsigned char * mask_f,   # nf
    # double * mask_pot,        # nv
    # Output values; preallocated to their own dimensions.
    long * out_nv_good, long * out_nf_good, long * out_ninterp,
    double * coords,         # nv, 3
    long * fvi,    # nf, 3
    long * vfi,  # nv, 6
    long * vfn,   # nv
    double * areaarr,        # nf
    long * idx_v,            # nv; Sorting index for v and f (for sorting other arrays, all output arrays are sorted.)
    long * idx_f,            # nf; Sorting index for v and f (for sorting other arrays, all output arrays are sorted.)
    long * n_interp_pairs,   # nv; Only valid until :out_ninterp, which is at most ~ 6*nv//7. (a point has 6 true neighbor)
    long * idx_interp_pairs, # nv, 3
    double * interp_weight,  # nv, 2
) noexcept nogil:  
    cdef long i, I, nf_good, ninterp
    
    # Local variables
    cdef long * good_v = <long *>calloc(7,sizeof(long))
    cdef long * good_f = <long *>calloc(4,sizeof(long))

    # Assigning values
    for i in prange(nv, nogil=True):
        idx_v[i] = i
        n_interp_pairs[i] = 0
        I = 2*i
        interp_weight[I  ] = 0.
        interp_weight[I+1] = 0.
        I = 3*i
        idx_interp_pairs[I  ] = -1
        idx_interp_pairs[I+1] = -1
        idx_interp_pairs[I+2] = -1
    for i in prange(nf, nogil=True):
        idx_f[i] = i
    memcpy(<void*>coords, <void*>coords_in, nv*3*sizeof(double))
    memcpy(<void*>fvi, <void*>fvi_in, nf*3*sizeof(long))
    memcpy(<void*>vfi, <void*>vfi_in, nv*6*sizeof(long))
    memcpy(<void*>vfn, <void*>vfn_in, nv*sizeof(long))
    memcpy(<void*>areaarr, <void*>areaarr_in, nf*sizeof(double))
    # # These are mutated, but originals need not to be kept as they are not used again
    # memcpy(<void*>mask_v, <void*>mask_v_in, nv*sizeof(unsigned char))
    # memcpy(<void*>mask_f, <void*>mask_f_in, nf*sizeof(unsigned char))
    # memcpy(<void*>mask_pot, <void*>mask_potential_in, nv*sizeof(double))

    # Sorting Faces
    countingsort_uchar_ptr(mask_f, idx_f, nf, 3)
    count_each_items_uchar_ptr(mask_f, good_f, nf, 3)
    nf_good = nf - good_f[0]
    # ...to descending order of good pts.
    flip_index_b(nf, idx_f, lb_nf)                                              # uses buffer long[nf] (in util.algorithms)
    assign_sorted_vector_inplace_lng_ptr(nf, 3, idx_f, fvi)
    assign_sorted_scalar_inplace_dbl_ptr(nf, idx_f, areaarr)
    assign_sorted_scalar_inplace_uchar_ptr(nf, idx_f, mask_f)
    _c_vfi_redirect(nv, nf, vfi, vfn, idx_f, lb_nf) # uses buffer long[nf]

    # Trimming Mesh
    ninterp = ___c_trim_mesh(nv, nf, 
        coords, fvi, vfi, vfn, 
        areaarr,
        mask_v, mask_f, mask_pot, 
        nf_good, good_f[3], n_interp_pairs, idx_interp_pairs, interp_weight
    )
    
    # Sorting Vertices
    countingsort_ptr(vfn, idx_v, nv, 6) # 6+1
    count_each_items_ptr(vfn, good_v, nv, 6) # 6+1

    flip_index_b(nv, idx_v, lb_nv)                                 # uses buffer long[nv] (in util.algorithms)
    _c_fvi_redirect(nv, nf, fvi, idx_v, lb_nv) # uses buffer long[nv]

    assign_sorted_vector_inplace_dbl_ptr(nv, 3, idx_v, coords)
    assign_sorted_vector_inplace_lng_ptr(nv, 6, idx_v, vfi)
    assign_sorted_scalar_inplace_lng_ptr(nv, idx_v, vfn)

    # Valid lengths for outputs
    out_nv_good[0] = nv-good_v[0]
    out_nf_good[0] = nf_good
    out_ninterp[0] = ninterp

    # Free all locally allocated memories.
    free(good_v)
    free(good_f)

cdef void _c_reorder_scalar_after_trim(
    long nv,
    double * outarr, double * inarr, 
    long n_good, long * idx_order,
    long ninterp, long * nip, long * iip, double * iw,
    # buffer
    double * buffer,
) noexcept nogil:
    memcpy(<void *>buffer, <void *>inarr, nv*sizeof(double))
    __interpolate_scalar_via_weight(ninterp, nip, iip, iw, buffer)
    assign_sorted_scalar_dbl_ptr(n_good, idx_order, buffer, outarr)

cdef void _c_reorder_vector_after_trim(
    long nv, long ndim,
    double * outarr, double * inarr, 
    long n_good, long * idx_order,
    long ninterp, long * nip, long * iip, double * iw,
    # buffer
    double * buffer,
) noexcept nogil:
    cdef long i
    memcpy(<void *>buffer, <void *>inarr, nv*ndim*sizeof(double))
    __interpolate_vector_via_weight(ninterp, ndim, nip, iip, iw, buffer)
    assign_sorted_vector_dbl_ptr(n_good, ndim, idx_order, buffer, outarr)

cdef void __integrate_over_good( # Maybe need to update not to duplicate the computation
    long ntfs, double * out, 
    long nf, double * coords, long * fvi, 
    long * vfi, long * vfn,
    unsigned char * mask_v, unsigned char * mask_f,
    double * integrand, double * areaarr, double * precompute,
    long * sorted_idx_f, long id1, long id2, long id3,
    double * potential,
    # buffer
    double * buffer_ntfs,    # minimum size ntfs
    double * buffer_ntfs_nf, # minimum size nf*ntfs
) noexcept nogil:
    cdef long i, j, k, i0, i1, i2
    cdef long I,       I0, I1, I2
    cdef long I_,    I0_, I1_, I2_
    # Maybe this needs to be preallocated and kept until the Psi(tau) is computed for all tau.
    # cdef double * buffer = <double *> malloc((nf-id1) * sizeof(double)) # keep, or make double type buffer, size nf

    for k in prange(id1, id2, nogil=True):
        i = sorted_idx_f[k]
        I = 3*i
        for j in range(3):
            if mask_v[fvi[I+j]]==1:
                i0 = fvi[I+j]
                i1 = fvi[I+(j+1)%3]
                i2 = fvi[I+(j+2)%3]
                I0 = 3*i0
                I1 = 3*i1
                I2 = 3*i2
                break
        I_ = (k-id1)*ntfs
        I0_ = ntfs*i0
        I1_ = ntfs*i1
        I2_ = ntfs*i2
        for j in range(ntfs):
            buffer_ntfs_nf[I_+j] = __integrate_2_missing(
                coords[I0  ], coords[I0+1], coords[I0+2], integrand[I0_+j], potential[i0],
                coords[I1  ], coords[I1+1], coords[I1+2], integrand[I1_+j], potential[i1],
                coords[I2  ], coords[I2+1], coords[I2+2], integrand[I2_+j], potential[i2],
            )
    for k in prange(id2, id3, nogil=True):
        i = sorted_idx_f[k]
        I = 3*i
        for j in range(3):
            if mask_v[fvi[I+j]]==0:
                i0 = fvi[I+j]
                i1 = fvi[I+(j+1)%3]
                i2 = fvi[I+(j+2)%3]
                I0 = 3*i0
                I1 = 3*i1
                I2 = 3*i2
                break
        I_ = (k-id1)*ntfs
        I0_ = ntfs*i0
        I1_ = ntfs*i1
        I2_ = ntfs*i2
        for j in range(ntfs):
            buffer_ntfs_nf[I_+j] = __integrate_1_missing(
                coords[I1  ], coords[I1+1], coords[I1+2], integrand[I1_+j], potential[i1],
                coords[I2  ], coords[I2+1], coords[I2+2], integrand[I2_+j], potential[i2],
                coords[I0  ], coords[I0+1], coords[I0+2], integrand[I0_+j], potential[i0],
                areaarr[i],
            )
    if id3<nf:
        for k in prange(id3, nf, nogil=True):
            I = ntfs*sorted_idx_f[k]
            I_ = (k-id1)*ntfs
            for j in range(ntfs):
                buffer_ntfs_nf[I_+j] = precompute[I+j]
    
    # Replace with Kahan_Sum function
    for j in range(ntfs):
        buffer_ntfs[j] = 0.0
        out[j] = 0.0
    # c = 0.0 # Kahan sum
    # out = 0.0
    for i in range(nf-id1):
        I = i*ntfs
        for j in range(ntfs):
            kahan_sum_iterator(out+j, buffer_ntfs_nf[I+j], buffer_ntfs+j)
    # free(buffer)
    # return integral

cdef void __interpolate_scalar_via_weight(
    long nruns, long * n_interp_pairs, long * idx_inter_pairs, 
    double * interp_weight, double * arr_to_interp,
) noexcept nogil: # True parallel
    cdef long j, i
    cdef long J2, J3
    cdef double f0, f1, f2, w1, w2
    for j in prange(nruns, nogil=True):
        J2 = 2*j
        J3 = 3*j
        if n_interp_pairs[j]==1:
            i = idx_inter_pairs[J3  ]
            f0 = arr_to_interp[i]
            f1 = arr_to_interp[idx_inter_pairs[J3+1]]
            w1 = interp_weight[J2  ]
            arr_to_interp[i] = f0 + w1*(f1-f0)
        else: # n_interp_pairs[j]==2:
            i = idx_inter_pairs[J3  ]
            f0 = arr_to_interp[i]
            f1 = arr_to_interp[idx_inter_pairs[J3+1]]
            f2 = arr_to_interp[idx_inter_pairs[J3+2]]
            w1 = interp_weight[J2  ]
            w2 = interp_weight[J2+1]
            arr_to_interp[i] = f0 + w1*(f1-f0) + w2*(f2-f0)

cdef void __interpolate_vector_via_weight(
    long nruns, long ndim, long * n_interp_pairs, long * idx_inter_pairs, 
    double * interp_weight, double * vec_to_interp,
) noexcept nogil: # True parallel
    cdef long k, j
    cdef long J2, J3, I0, I1, I2
    cdef double f0, f1, f2, w1, w2
    for j in prange(nruns, nogil=True):
        J2 = 2*j
        J3 = 3*j
        if n_interp_pairs[j]==1:
            I0 = ndim*idx_inter_pairs[J3  ]
            I1 = ndim*idx_inter_pairs[J3+1]
            for k in range(ndim):
                f0 = vec_to_interp[I0+k]
                f1 = vec_to_interp[I1+k]
                w1 = interp_weight[J2  ]
                vec_to_interp[I0+k] = f0 + w1*(f1-f0)
        else: # n_interp_pairs[j]==2:
            I0 = ndim*idx_inter_pairs[J3  ]
            I1 = ndim*idx_inter_pairs[J3+1]
            I2 = ndim*idx_inter_pairs[J3+2]
            for k in range(ndim):
                f0 = vec_to_interp[I0+k]
                f1 = vec_to_interp[I1+k]
                f2 = vec_to_interp[I2+k]
                w1 = interp_weight[J2  ]
                w2 = interp_weight[J2+1]
                vec_to_interp[I0+k] = f0 + w1*(f1-f0) + w2*(f2-f0)
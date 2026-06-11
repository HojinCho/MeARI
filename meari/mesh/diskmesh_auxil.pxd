from ..utils.types cimport func_t_double_ternary

cdef void quad_annulus_connectivity(
    long nrad, long nazm, long * fvi, 
    long * vfi, long * vfn,
) noexcept nogil
cdef void tri_annulus_connectivity(
    long nrad, long nazm, long * fvi, 
    long * vfi, long * vfn,
) noexcept nogil
cdef double __triangle_area(
    double ux, double uy, double uz,
    double vx, double vy, double vz,
) noexcept nogil
cdef void __assign_position_difference_vectors(
    long nf, double * coords, long * fvi, double * vecarr
) noexcept nogil
cdef void __assign_triangle_area(
    long nf, double * vecarr, double * outarr
) noexcept nogil
cdef void __check_region(
    long nv, long * vfi, long * vfn,
    double * test_scalar, double threshold, double direction,
    unsigned char * out_bool_points, unsigned char * out_cell_n_good_points,
)
cdef void __composite_mask(
    long nv, 
    double * inmask, double * scalar, double threshold, double direction,
    double * outmask,
) noexcept nogil
cdef void __composite_mask_inplace(
    long nv, 
    double * mask, double * scalar, double threshold, double direction,
) noexcept nogil
cdef void __replace_mask(
    long nv, 
    double * mask, double * scalar, double threshold, double direction,
) noexcept nogil
cdef void __filter_by_mask(
    long nv, long nf, 
    long * fvi, long * vfi, long * vfn,
    double * mask,
    unsigned char * out_bool_points, unsigned char * out_cell_n_good_points,
) noexcept nogil
cdef void make_obscuration_mask(long nv, long nf, long nvpf, 
    long * fvi, long * vfi, long * vfn, double * obsc_mask, double * normal_z, 
    unsigned char * mask_v, unsigned char * mask_f, double * mask_obs) noexcept nogil
cdef void __interpolate_to_zero(
    double x1, double y1, double z1, double f1, double m1,
    double x2, double y2, double z2, double f2, double m2,
    double *x, double *y, double *z, double *f,
) noexcept nogil
cdef double __barycentric_integration(double f1, double f2, double f3,) noexcept nogil
cdef double __integrate__normal(
    double x1, double y1, double z1, double f1,
    double x2, double y2, double z2, double f2,
    double x3, double y3, double z3, double f3,
    double area,
) noexcept nogil
cdef double __integrate_1_missing(
    double x1, double y1, double z1, double f1, double m1,
    double x2, double y2, double z2, double f2, double m2,
    double xm, double ym, double zm, double fm, double mm,
    double area,
) noexcept nogil
cdef double __integrate_2_missing(
    double xp, double yp, double zp, double fp, double mp,
    double x1, double y1, double z1, double f1, double m1,
    double x2, double y2, double z2, double f2, double m2,
) noexcept nogil
cdef void assign_faceval_from_vertval(
    double * faceval, long nf, 
    long ndim, long * fvi, double * vertval, double * areaarr, 
) noexcept nogil
cdef void __reorder_based_on_dblscalar(
    long nv, long nf, double * coord, long * fvi, 
    long * vfi, long * vfn,
    double * potential, long * idx, long * lb_nv,
) noexcept nogil
cdef void __two_point_interp(
    double x0, double y0, double z0, double m0,
    double x1, double y1, double z1, double m1,
    double x2, double y2, double z2, double m2,
    double * x, double * y, double * z,
) noexcept nogil
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
) noexcept nogil
cdef void _c_fvi_redirect(
    long nv, long nf, long * fvi, long * idx_v, long * lb_nv,
) noexcept nogil
cdef void _c_vfi_redirect(
    long nv, long nf, long * vfi, long * vfn, long * idx_f, long * lb_nf,
) noexcept nogil
cdef void assign_face_value_by_vert_values(
    long nv, long nf, func_t_double_ternary f,
    double * val_f, double * val_v, long * fvi,
) noexcept nogil
cdef void sort_faces_by_vert_val(
    long nv, long nf, 
    double * val_v, func_t_double_ternary f,  
    long * fvi,    # nf, nvpf; F |-> V: Need to be sorted
    long * vfi,  # nv, nfpv; V |-> F: Need to be redirected
    long * vfn,   # nv; NO need to be sorted
    double * cell_areas,     # nf; Need to be sorted
    long * idx_f,            # nf; buffer
    long * lb_nf,            # nf; buffer
    double * val_f           # return value
) noexcept nogil
cdef void _c_trim_mesh(
    long nv, long nf,
    double * coords_in,          # nv, 3 (x,y,z)
    long * fvi_in, # nf, 3 (v1,v2,v3)
    long * vfi_in, # nv, 6 (f1,f2,f3,f4,f5,f6) # meaningful at most vfn
    long * vfn_in, # nv
    double * areaarr_in,         # nf
    unsigned char * mask_v,   # nv
    unsigned char * mask_f,   # nf
    double * mask_pot,        # nv
    # Buffers
    long * lb_nv,   # nv
    long * lb_nf,   # nf
    # Output values; preallocated to their own dimensions.
    long * out_nv_good, long * out_nf_good, long * out_ninterp,
    double * coords,         # nv, 3
    long * fvi, # nf, 3
    long * vfi, # nv, 6
    long * vfn, # nv
    double * areaarr,        # nf
    long * idx_v,            # nv; Sorting index for v and f (for sorting other arrays, all output arrays are sorted.)
    long * idx_f,            # nf; Sorting index for v and f (for sorting other arrays, all output arrays are sorted.)
    long * n_interp_pairs,   # nv; Only valid until :out_ninterp, which is at most ~ 6*nv//7. (a point has 6 true neighbor)
    long * idx_interp_pairs, # nv, 3
    double * interp_weight,  # nv, 2
) noexcept nogil
cdef void _c_reorder_scalar_after_trim(
    long nv,
    double * outarr, double * inarr, 
    long n_good, long * idx_order,
    long ninterp, long * nip, long * iip, double * iw,
    # buffer
    double * buffer,
) noexcept nogil
cdef void _c_reorder_vector_after_trim(
    long nv, long ndim,
    double * outarr, double * inarr, 
    long n_good, long * idx_order,
    long ninterp, long * nip, long * iip, double * iw,
    # buffer
    double * buffer,
) noexcept nogil
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
) noexcept nogil
cdef void __interpolate_scalar_via_weight(
    long nruns, long * n_interp_pairs, long * idx_inter_pairs, 
    double * interp_weight, double * arr_to_interp,
) noexcept nogil
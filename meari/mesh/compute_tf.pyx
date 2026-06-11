# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False


cimport cython
from cython.parallel import prange

from libc.stdio cimport printf

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, cos, sin, pow,
    fmin, fmax, fabs,
)

from ..extern.lia cimport lia_sv_scalar_inplace

from ..utils.algorithms cimport (
    find_min_max, max_ternary,
    binary_search_ptr_L, binary_search_ptr_R, 
    binary_search_ptr_R_desc, 
    countingsort_uchar_ptr_subdivide, 
    count_each_items_uchar_ptr,
    assign_sorted_vector_inplace_dbl_ptr,
    assign_sorted_scalar_inplace_dbl_ptr,
    find_root, finite_difference_f, finite_difference_b,
    kahan_sum_iterator,
)

from .diskmesh_auxil cimport (
    assign_faceval_from_vertval,
    sort_faces_by_vert_val,
    __filter_by_mask, __replace_mask,
    __integrate_over_good, 
)

### Auxiliary Functions to compute Transfer Functions

cdef void __assign_cumulsum(
    long nf, long ntfs, double *precompute, double * cumulsum,
    double * s, double * c, # use preallocated memory.
) noexcept nogil:
    cdef long i, j, I
    # cdef double * s = <double *> calloc(ntfs, sizeof(double))
    # cdef double * c = <double *> calloc(ntfs, sizeof(double))
    for j in range(ntfs):
        s[j] = 0
        c[j] = 0
    # s = 0
    # c = 0
    # for j in prange(ntfs, nogil=True): # RACE CONDITION, DO NOT USE THIS!
    #     for i in range(nf):
    #         kahan_sum_iterator(s+j, precompute[i*nf+j], c+j)
    #         cumulsum[i*nf+j] = s[j]
    for i in range(nf):
        I = i*ntfs
        for j in range(ntfs):
            kahan_sum_iterator(s+j, precompute[I+j], c+j)
            cumulsum[I+j] = s[j]

cdef void compute_normalizing_factors(
    double * out, long nf, long ntfs, double * avg_emission, double * c
) noexcept nogil:
    cdef long i, j
    cdef double * ptr
    for j in range(ntfs):
        out[j] = 0
        c[j] = 0
    for i in range(nf):
        ptr = avg_emission + i*ntfs
        for j in range(ntfs):
            kahan_sum_iterator(out+j, ptr[j], c+j)
    for j in range(ntfs):
        out[j] = 1/out[j]

# CDF precomputing logic.
# Step 1. Sort faces according to the "maximum"(minimum?) tau value of vertices.
# Step 2. Precompute the response CDF along sorted faces.
# Step 3. Given tau_edge[i], flag all vertices having larger lags than tau_edge[i] (refer to Step 1.)
# Step 4. Compute masks for vertices and faces as usual.
# Step 5. Find first k where mask_f[k]<3 and let k0=k
# Step 6. Isolate the region between 0:k0 by letting all mask_f in this range to 0.
# Step 7. Run the usual computation code to compute resps_1d_cdf[i].
# Step 8. Add cumulsum[k0-1] to resps_1d_cdf[i] if k0>0.
cdef void compute_tf_cdf_internal(
    long n_resp_tau, long ntfs, double * resps_1d_cdf, 
    long index_start, long index_stop, 
    double * tauarr, double * tau_resps,
    long nv, long nf, long nvpf, double * coords, long * fvi, long * vfi, long * vfn, double * cell_areas, 
    double * integrand, double * precomp, double * cumulsum, double * tau_f,
    # buffer inputs
    long * b_idx_f, double * db_nf, # long/double [nf]
    double * db_nfntfs, # double [nf*ntfs]
    double * db_ntfs,   # double [ntfs]
    unsigned char * mask_v, unsigned char * mask_f, double * mask_metric, # (nv, nf, nv)
) noexcept nogil:
    # locals
    # cdef long * good_counts = <long*>malloc(4*sizeof(long))
    cdef long[4] good_counts
    cdef long i, j, J
    cdef long k, k0
    cdef double * dbptr
    cdef double * current_cdf
    cdef double val
    
    for i in range(index_start, index_stop): # index on time.
        current_cdf = resps_1d_cdf + i
        # memset only works proprerly for unsigned char.
        memset(<void *> mask_v,    1, nv*sizeof(unsigned char))
        memset(<void *> mask_f, nvpf, nf*sizeof(unsigned char))
        # Step 3. Given tau_edge[i], flag all vertices having larger lags than tau_edge[i] (refer to Step 1.)
        __replace_mask(
            nv, mask_metric, tauarr, tau_resps[i], -1,
        )
        # Step 4. Compute masks for vertices and faces as usual.
        __filter_by_mask(
            nv, nf, fvi, vfi, vfn, mask_metric, mask_v, mask_f,
        )
        # Step 5. Find first k where mask_f[k]<3 and let k0=k
        # # 5-1. Linear search; Best: O(1), Worst: O(nf), Average: O(nf/2)
        # 5-2. Binary search method; Best: O(log(nf)), Worst: O(log(nf)), Average: O(log(nf))
        k0 = binary_search_ptr_R(tau_resps[i], tau_f, nf)
        # k0 = binary_search_ptr_R_desc(tau_resps[i], tau_f, nf)
        # Possible outputs:
        # k0 = 0      :              tau_bin_edges[i]<tau_f[ 0] , mask_f[:] <3
        # 0 < k0 < nf : tau_f[k0-1]<=tau_bin_edges[i]<tau_f[k0] , mask_f[0:k0-1] = 3, mask_f[k0:nf-1] < 3
        # k0 = nf     : tau_f[nf-1]<=tau_bin_edges[i]           , mask_f[:] = 3
        if k0==nf:
            dbptr = cumulsum + (nf-1)*ntfs
            for j in range(ntfs):
                current_cdf[j*n_resp_tau] = dbptr[j]
        else: # if k0<nf:
            # Step 6. Isolate the region between 0:k0 by letting all mask_f in this range to 0.
            memset(<void *> mask_f, 0, k0*sizeof(unsigned char))
            # Step 7. Run the usual computation code to compute resps_1d_cdf[i].
            for k in prange(nf, nogil=True):
                b_idx_f[k] = k
            for k in range(4):
                good_counts[k]=0
            countingsort_uchar_ptr_subdivide(mask_f, b_idx_f, k0, nf, 3)
            count_each_items_uchar_ptr((mask_f+k0), good_counts, nf-k0, 3)
            good_counts[0] += k0
            # printf("%d %d, %d %d %d %d\n",i, k0, good_counts[0],good_counts[1],good_counts[2],good_counts[3])
            for k in range(1, 4):
                good_counts[k] += good_counts[k-1]
            __integrate_over_good(
                ntfs, db_ntfs,
                nf, coords, fvi, vfi, vfn, mask_v, mask_f, 
                integrand, cell_areas, precomp, b_idx_f, 
                good_counts[0], good_counts[1], good_counts[2], mask_metric, 
                db_nf, db_nfntfs,
            )
            for j in range(ntfs):
                current_cdf[j*n_resp_tau] = db_ntfs[j]
            # Step 8. Add cumulsum[k0-1] to resps_1d_cdf[i] if k0>0.
            if k0>0:
                dbptr = cumulsum + (k0-1)*ntfs
                for j in range(ntfs):
                    current_cdf[j*n_resp_tau] += dbptr[j]
    if index_start>0:
        # resps_1d_cdf[j*n_resp_tau + i] = dbptr[j]
        dbptr = resps_1d_cdf + index_start
        for j in prange(ntfs, nogil=True):
            J = j*n_resp_tau
            current_cdf = resps_1d_cdf + J
            val = resps_1d_cdf[J+index_start]
            for i in range(index_start):
                current_cdf[i] = val # dbptr[J]
    if index_stop<n_resp_tau:
        dbptr = resps_1d_cdf + (index_stop-1)
        for j in prange(ntfs, nogil=True):
            J = j*n_resp_tau
            current_cdf = resps_1d_cdf + J
            val = resps_1d_cdf[J+index_stop-1]
            for i in range(index_stop, n_resp_tau):
                current_cdf[i] = val # dbptr[J]
    # free(good_counts)
    
cdef void compute_tf_cdf(
    # Response
    long n_resp_tau, long ntfs,
    double * resps_1d_cdf, # n_resp_tau
    # persistent
    double * tau_resps,    # n_resp_tau
    # Mesh Properties
    long nv, long nf, long nvpf, 
    double * coords,     # nv, 3
    long * fvi,         # nf, nvpf=3, F|->V
    long * vfi,         # nv, nfpv=6, V|->F
    long * vfn,          # nv
    double * cell_areas, # nf
    double * integrand,  # nv, ntfs
    double * normalizer, # nv, ntfs
    double * tauarr,     # nv
    # buffers
    double * precomp,      # nf*ntfs
    double * average,      # nf*ntfs
    double * cumulsum,     # nf*ntfs
    double * mult,         # ntfs
    double * tau_f,        # nf
    long * idx_f,          # nf
    long * lb_nf,          # nf
    double * db_nf,        # nf
    double * db_nfntfs,    # nf*ntfs
    double * db_ntfs,      # ntfs
    # mask buffers
    unsigned char * mask_v,# nv
    unsigned char * mask_f,# nf
    double * mask_metric,  # nv
) noexcept nogil:
    cdef long i, index_start, index_stop
    cdef double tmin, tmax
    # Step 1. Sort faces according to the "maximum"(minimum?) tau value of vertices.
    # ("Reorder by Tau for Faces")
    sort_faces_by_vert_val(nv, nf, tauarr, max_ternary, fvi, vfi, vfn, cell_areas, idx_f, lb_nf, tau_f) # uses buffer long[nf]
    # Step 2. Precompute the response CDF along sorted faces.
    # __assign_precompute(nf, ntfs, coords, fvi, integrand, cell_areas, precomp)
    assign_faceval_from_vertval(precomp, nf, ntfs, fvi, integrand,  cell_areas)
    assign_faceval_from_vertval(average, nf, ntfs, fvi, normalizer, cell_areas)
    # But don't apply mult until the end, given how small it gets once it is applied.
    __assign_cumulsum(nf, ntfs, precomp, cumulsum, db_nfntfs, db_nfntfs+ntfs)
    
    find_min_max(tauarr, nv, &tmin, &tmax)
    # index_start = binary_search_ptr_L(tmin, tau_resps, n_resp_tau) -1
    # index_stop  = binary_search_ptr_R(tmax, tau_resps, n_resp_tau) +1
    index_start = binary_search_ptr_R(tmin, tau_resps, n_resp_tau) -1
    index_stop  = binary_search_ptr_L(tmax, tau_resps, n_resp_tau) +1

    # Guard Clause: This only happens if the mesh covers more range than the given tau.
    if index_start < 0:
        index_start = 0
    if index_stop > n_resp_tau:
        index_stop = n_resp_tau
    # End Guard Clause
    compute_tf_cdf_internal(
        n_resp_tau, ntfs, resps_1d_cdf, index_start, index_stop, 
        tauarr, tau_resps,
        nv, nf, nvpf, coords, fvi, vfi, vfn, cell_areas, 
        integrand, precomp, cumulsum, tau_f, lb_nf, db_nf, db_nfntfs, db_ntfs,
        mask_v, mask_f, mask_metric,
    )
    compute_normalizing_factors(mult, nf, ntfs, average, db_ntfs)
    for i in range(ntfs):
        lia_sv_scalar_inplace(                    # CDF[i]
            mult[i], resps_1d_cdf + i*n_resp_tau, # = mult[i] * CDF[i]
            n_resp_tau,                           # in total, ntau elements
            stride_x=1                            # 
        )

cdef void compute_tf_pdf(
    # Response
    long n_resp_tau, long ntfs,
    double * resps_1d_pdf, # n_resp_tau*ntfs
    double * resps_1d_cdf, # n_resp_tau*ntfs
    # persistent
    double * tau_resps,    # n_resp_tau
    double * d_tau_bins,   # n_resp_tau
    # Mesh Properties
    long nv, long nf, long nvpf, 
    double * coords,     # nv, 3
    long * fvi,         # nf, nvpf=3, F|->V
    long * vfi,         # nv, nfpv=6, V|->F
    long * vfn,          # nv
    double * cell_areas, # nf
    double * integrand,  # nv, ntfs
    double * normalizer, # nv, ntfs
    double * tauarr,     # nv
    # buffers
    double * precomp,      # nf*ntfs
    double * average,      # nf*ntfs
    double * cumulsum,     # nf*ntfs
    double * mult,         # ntfs
    double * tau_f,        # nf
    long * idx_f,          # nf
    long * lb_nf,          # nf
    double * db_nf,        # nf
    double * db_nfntfs,    # nf*ntfs
    double * db_ntfs,      # ntfs
    # mask buffers
    unsigned char * mask_v,# nv
    unsigned char * mask_f,# nf
    double * mask_metric,  # nv
) noexcept nogil:
    cdef long i, I, j, Im1
    compute_tf_cdf(
        n_resp_tau, ntfs, resps_1d_cdf, tau_resps,
        nv, nf, nvpf, coords, fvi, vfi, vfn, cell_areas, integrand, normalizer, tauarr,
        precomp, average, cumulsum, mult, tau_f, idx_f, lb_nf, db_nf, db_nfntfs, db_ntfs,
        mask_v, mask_f, mask_metric,
    )

    # for i in range(n_resp_tau):
    #     I = i*ntfs
    #     for j in range(ntfs):
    #         resps_1d_pdf[I+j] = resps_1d_cdf[I+j]

    # Backward difference is now implemented.
    # For time-fast alignment,
    for i in prange(ntfs, nogil=True):
        I = i*n_resp_tau
        resps_1d_pdf[I] = 0
        for j in range(1, n_resp_tau):
            resps_1d_pdf[I+j] = (resps_1d_cdf[I+j] - resps_1d_cdf[I+j-1])/d_tau_bins[j]
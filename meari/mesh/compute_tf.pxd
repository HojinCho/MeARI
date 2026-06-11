cdef void compute_tf_cdf_internal(
    long n_resp_tau, long ntfs, double * resps_1d_cdf, 
    long index_start, long index_stop, 
    double * tauarr, double * tau_resps,
    long nv, long nf, long nvpf, double * coords, long * fvi, long * vfi, long * vfn, double * cell_areas, 
    double * integrand, double * precomp, double * cumulsum, double * tau_f,
    # buffer inputs
    long * b_idx_f, double * db_nf, # long/double [nf]
    double * db_nfntfs, # double [nf*ntfs]
    double * db_ntfs,      # ntfs
    unsigned char * mask_v, unsigned char * mask_f, double * mask_metric, # (nv, nf, nv)
) noexcept nogil
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
) noexcept nogil
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
) noexcept nogil
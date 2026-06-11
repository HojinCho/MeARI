# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cimport cython

@cython.final
cdef class LightCurves:
    cdef:
        # Response LCs can be more than 1. NLCs = (# of driving) + (# of response) = 1 + (# of response)
        public long NLCs
        long _NResps
        # Number of points in each LC
        long * _npts
        long * _offset
        public object NResp
        public long n
        public bint has_driving
        public bint has_response
        public bint has_cov
        public bint has_corr
        public bint lc_in_mag
        # internal storage for LCs
        double * _t
        double * _y
        double * _e
        double * _cov
        double * _corr
        unsigned char * _lcid
        long ** _idxs

        # Debug Purpose! 
        public object t1d
        public object y1d
        public object e1d
        public object cov1d
        public object corr1d
        public object lcid1d
    

    cdef void _c_allocate_mem(self, long NLCs, long NDriv, long * NResp, 
        bint store_cov=*, bint store_corr=*, bint lc_in_mag=*) noexcept nogil
    cdef void _c_single_LC(self, 
        unsigned char lcid, double * t, double * y, double * e, 
        bint is_magnitude=*, double magzp=*,
    ) noexcept nogil
    cdef void _c_corr_pair(self, unsigned char lcid1, unsigned char lcid2, double * corr) noexcept nogil
    cdef void _c_cov_pair( self, unsigned char lcid1, unsigned char lcid2, double * cov ) noexcept nogil

    cdef void _c_expose_packet(self, packet_LC * packet) noexcept nogil

    cpdef void memalloc_arr(self, long [:] npts, bint has_driving=*, 
        bint store_cov=*, bint store_corr=*, bint lc_in_mag=*)
    cpdef void set_single_LC(self, 
        unsigned char lcid, double [:] t, double [:] y, double [:] e,
        bint is_magnitude=*, double magzp=*
    )
    cpdef void set_corr(self, unsigned char lcid1, unsigned char lcid2, double [:,:] corr)
    cpdef void set_cov( self, unsigned char lcid1, unsigned char lcid2, double [:,:] cov )
    
cdef struct packet_LC:
    long NLCs
    long NResps
    bint has_driving
    bint lc_in_mag

    size_t n
    double * t
    double * y
    double * e

    bint has_cov
    bint has_corr
    double * cov
    double * corr

    long * npts
    long * offset

cdef void free_packet_LC(packet_LC * packet) noexcept nogil
# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False

cdef void vel_field_disk_thick( 
    # Goad et al. 2012 MNRAS 426, 3086-3111, eq.2
    # Reduces to Keplerian if coords[3*i+2]==0 for all i.
    long n, 
    double logM, # Mass of the central object in solar masses, in common log.
    double * coords, double * R, # in lt-days
    double * vel, # output pointer in km/s
) noexcept nogil
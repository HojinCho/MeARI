# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

cimport cython
from cython.parallel import prange

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

from ..utils.algorithms cimport (
    quicksort_ptr, mergesort_ptr,
    assign_sorted_scalar_inplace_dbl_ptr, assign_sorted_scalar_inplace_dbl_ptr_bf,
    assign_sorted_scalar_dbl_ptr,
)

from ..utils.numpy_interface cimport (
    numpy_dbl_1d, numpy_dbl_2d,
    numpy_lng_1d, numpy_lng_2d,
    numpy_uch_1d
)

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, M_LOG10E,
    sqrt, log, exp, cos, sin, pow,
    fmin, fmax, fabs,
)

# Magnitude-Flux conversion functions
# maybe need to separate into other module?
cdef double magfact = 2.5*M_LOG10E
cdef double maginvf = 0.4*M_LN10

cdef double flux2mag(double flux, double zp) noexcept nogil:
    return -magfact*log(flux) + zp
cdef double ferr2merr(double ferr, double flux) noexcept nogil:
    return magfact*(ferr/flux)
cdef double mag2flux(double mag, double zp) noexcept nogil:
    return exp(-maginvf*(mag-zp))
cdef double merr2ferr(double merr, double flux) noexcept nogil:
    return maginvf*merr*flux
cdef void mag2flux_err_arr(
    long n, double * flux, double * ferr, double * mag, double * merr, double zp,
) noexcept nogil:
    cdef long i
    for i in prange(n, nogil=True):
        flux[i] = mag2flux( mag[i], zp)
        ferr[i] = merr2ferr(merr[i], flux[i])
cdef void flux2mag_err_arr(
    long n, double * mag, double * merr, double * flux, double * ferr, double zp,
) noexcept nogil:
    cdef long i
    for i in prange(n, nogil=True):
        merr[i] = ferr2merr(ferr[i], flux[i])
        mag[i]  = flux2mag( flux[i], zp)
cdef void mag2flux_err_arr_inplace(long n, double * y, double * e, double zp) noexcept nogil:
    cdef long i
    for i in prange(n, nogil=True):
        y[i] = mag2flux(y[i], zp)
        e[i] = merr2ferr(e[i], y[i])
cdef void flux2mag_err_arr_inplace(long n, double * y, double * e, double zp) noexcept nogil:
    cdef long i
    for i in prange(n, nogil=True):
        e[i] = ferr2merr(e[i], y[i])
        y[i] = flux2mag(y[i], zp)

cdef void free_packet_LC(packet_LC * packet) noexcept nogil:
    packet.t      = NULL
    packet.y      = NULL
    packet.e      = NULL
    packet.cov    = NULL
    packet.corr   = NULL
    packet.npts   = NULL
    packet.offset = NULL
    free(packet)

@cython.final
cdef class LightCurves():
    def __cinit__(self):
        pass
    
    cdef void _c_allocate_mem(self,
        long NLCs,
        long NDriv, long * NResp,
        bint store_cov=False,
        bint store_corr=False,
        bint lc_in_mag=False,
    ) noexcept nogil:
        # allocate memory and assign invariant topology (mesh fviectivity)
        # Don't assign other values
        cdef long i, j
        if NLCs<1:
            raise ValueError("NLCs must be at least 1, with 0 or 1 driving LC.")
        self.NLCs = NLCs
        self._npts = <long *>malloc(self.NLCs*sizeof(long))
        self._offset = <long *>malloc(self.NLCs*sizeof(long))
        # self.NDriv = NDriv
        if NDriv>0:
            self.has_driving = True
            self._npts[0] = NDriv
            self._NResps = self.NLCs-1
            if self._NResps>0:
                self.has_response = True
                memcpy(<void *>(&self._npts[1]), <void *>NResp, self._NResps*sizeof(long))
                with gil:
                    self.NResp = numpy_lng_1d((&self._npts[1]), self._NResps, False)
            else:
                self.has_response = False
        else:
            self.has_driving = False
            self.has_response = True
            self._NResps = self.NLCs
            memcpy(<void *>self._npts, <void *>NResp, self._NResps*sizeof(long))
            with gil:
                self.NResp = numpy_lng_1d(self._npts, self._NResps, False)
        
        self.n = 0
        for i in range(self.NLCs):
            self._offset[i] = self.n
            self.n += self._npts[i]
            
        self._t = <double *>malloc(self.n*sizeof(double))
        self._y = <double *>malloc(self.n*sizeof(double))
        self._e = <double *>malloc(self.n*sizeof(double))
        self._lcid = <unsigned char *>malloc(self.n*sizeof(unsigned char))
        self._idxs = <long **>malloc(self.NLCs*sizeof(long *))
        if self._idxs==NULL:
            raise MemoryError("Memory allocation failed.")
        for i in range(self.NLCs):
            self._idxs[i] = <long *>malloc(self._npts[i]*sizeof(long))
            if self._idxs[i]==NULL:
                raise MemoryError("Memory allocation failed.")
            for j in range(self._npts[i]):
                self._idxs[i][j] = j

        if lc_in_mag:
            self.lc_in_mag = True
        else:
            self.lc_in_mag = False
        
        if store_cov:
            self.has_cov = True
            self._cov = <double *>calloc(self.n*self.n,sizeof(double)) # zero-initialized
        else:
            self.has_cov = False
            self._cov = NULL
        if store_corr:
            self.has_corr = True
            self._corr = <double *>calloc(self.n*self.n,sizeof(double)) # zero-initialized
            for i in prange(self.n, nogil=True):
                self._corr[i*(self.n +1)] = 1.0 # Initialize as identity matrix
        else:
            self.has_corr = False
            self._corr = NULL
        if (
            self._t == NULL or self._y == NULL or self._e == NULL or self._lcid == NULL
            or self._npts == NULL or self._offset == NULL
        ):
            raise MemoryError("Memory allocation failed.")

        # Debug Purpose! 
        with gil:
            self.t1d    = numpy_dbl_1d(self._t,    self.n, False)
            self.y1d    = numpy_dbl_1d(self._y,    self.n, False)
            self.e1d    = numpy_dbl_1d(self._e,    self.n, False)
            self.lcid1d = numpy_uch_1d(self._lcid, self.n, False)
            if self.has_cov:
                self.cov1d = numpy_dbl_2d(self._cov, self.n, self.n, False)
            if self.has_corr:
                self.corr1d = numpy_dbl_2d(self._corr, self.n, self.n, False)
            

    def __dealloc__(self):
        if self._idxs!=NULL:
            for i in range(self.NLCs):
                if self._idxs[i]!=NULL:
                    free(self._idxs[i])
            free(self._idxs)
        if self._t!=NULL:
            free(self._t)
        if self._y!=NULL:
            free(self._y)
        if self._e!=NULL:
            free(self._e)
        if self._lcid!=NULL:
            free(self._lcid)
        if self._npts!=NULL:
            free(self._npts)
        if self._offset!=NULL:
            free(self._offset)
        
        if self.has_cov and self._cov!=NULL:
            free(self._cov)
        if self.has_corr and self._corr!=NULL:
            free(self._corr)

    @property
    def NDriv(self):
        if self.has_driving:
            return self._npts[0]
        else:
            return 0

    cdef void _c_single_LC(self,
        unsigned char lcid,
        double * t, double * y, double * e,
        bint is_magnitude=False, double magzp=0,
    ) noexcept nogil:
        # memset only works proprerly for unsigned char.
        cdef long offset = self._offset[lcid]
        cdef long nlc    = self._npts[lcid]
        cdef size_t i, I, j, J
        memset(<void*> (&self._lcid[offset]), lcid, nlc*sizeof(unsigned char))
        # memcpy(<void*> (&self._t[offset]), <void*> t, nlc*sizeof(double))
        # memcpy(<void*> (&self._y[offset]), <void*> y, nlc*sizeof(double))
        # memcpy(<void*> (&self._e[offset]), <void*> e, nlc*sizeof(double))
        mergesort_ptr(t, self._idxs[lcid], 0, nlc-1)
        assign_sorted_scalar_dbl_ptr(nlc, self._idxs[lcid], t, self._t+offset)
        assign_sorted_scalar_dbl_ptr(nlc, self._idxs[lcid], y, self._y+offset)
        assign_sorted_scalar_dbl_ptr(nlc, self._idxs[lcid], e, self._e+offset)

        if self.lc_in_mag: # in Mag space
            if not is_magnitude: # Convert to magnitude if flux
                flux2mag_err_arr_inplace(nlc, &self._y[offset], &self._e[offset], magzp)
            if self.has_cov:
                for i in prange(nlc, nogil=True): # previous cov is already invalidated, because we gave e directly.
                    I = i+offset
                    self._cov[I*(self.n+1)] = self._e[I]*self._e[I]
                if self.has_corr: # but if corr is stored, it is still OK.
                    for i in prange(nlc, nogil=True):
                        I = (i+offset)
                        for j in range(i+1, nlc):
                            J = (j+offset)
                            self._cov[I*self.n + J] = self._e[I]*self._e[J]*self._corr[I*self.n + J]
                            self._cov[J*self.n + I] = self._cov[I*self.n + J]
        else: # in Flux space
            if is_magnitude: # Convert to flux if magnitude
                mag2flux_err_arr_inplace(nlc, &self._y[offset], &self._e[offset], magzp)
            if self.has_cov:
                for i in prange(nlc, nogil=True): # previous cov is already invalidated, because we gave e directly.
                    I = i+offset
                    self._cov[I*(self.n+1)] = self._e[I]*self._e[I]
                if self.has_corr: # but if corr is stored, it is still OK.
                    for i in prange(nlc, nogil=True):
                        I = (i+offset)
                        for j in range(i+1, nlc):
                            J = (j+offset)
                            self._cov[I*self.n + J] = self._e[I]*self._e[J]*self._corr[I*self.n + J]
                            self._cov[J*self.n + I] = self._cov[I*self.n + J]

    cdef void _c_cov_pair(self, unsigned char lcid1, unsigned char lcid2, double * cov) noexcept nogil:
        # Maybe it's better to use packed array?
        cdef long i1, i2, j, I2, I1
        cdef long offset1 = self._offset[lcid1]
        cdef long offset2 = self._offset[lcid2]
        cdef long nlc1 = self._npts[lcid1]
        cdef long nlc2 = self._npts[lcid2]
        cdef long * idx1 = self._idxs[lcid1]
        cdef long * idx2 = self._idxs[lcid2]
        if self.has_cov:
            for I2 in prange(nlc2, nogil=True):
                i2 = idx2[I2]
                j = (offset2 + i2)
                memcpy(
                    <void *> &self._cov[j*self.n + offset1], 
                    <void *> &cov[i2*nlc1], 
                    nlc1*sizeof(double)
                )
                for I1 in range(nlc1):
                    i1 = idx1[I1]
                    self._cov[(offset1 + i1)*self.n + j] = self._cov[j*self.n + (offset1 + i1)]
        if self.has_corr:
            for I2 in prange(nlc2, nogil=True):
                i2 = idx2[I2]
                j = (offset2 + i2)
                for I1 in range(nlc1):
                    i1 = idx1[I1]
                    self._corr[j*self.n + (offset1 + i1)] = cov[i2*nlc1 + i1]/(self._e[j]*self._e[offset1 + i1])
                    self._corr[(offset1 + i1)*self.n + j] = self._corr[j*self.n + (offset1 + i1)]

    cdef void _c_corr_pair(self, unsigned char lcid1, unsigned char lcid2, double * corr) noexcept nogil:
        # Maybe it's better to use packed array?
        cdef long i1, i2, j, I2, I1
        cdef long offset1 = self._offset[lcid1]
        cdef long offset2 = self._offset[lcid2]
        cdef long nlc1 = self._npts[lcid1]
        cdef long nlc2 = self._npts[lcid2]
        cdef long * idx1 = self._idxs[lcid1]
        cdef long * idx2 = self._idxs[lcid2]
        if self.has_cov:
            for I2 in prange(nlc2, nogil=True):
                i2 = idx2[I2]
                j = (offset2 + i2)
                for I1 in range(nlc1):
                    i1 = idx1[I1]
                    self._cov[j*self.n + (offset1 + i1)] = corr[i2*nlc1 + i1]*self._e[j]*self._e[offset1 + i1]
                    self._cov[(offset1 + i1)*self.n + j] = self._cov[j*self.n + (offset1 + i1)]
        
        if self.has_corr:
            for I2 in prange(nlc2, nogil=True):
                i2 = idx2[I2]
                j = (offset2 + i2)
                memcpy(
                    <void *> &self._corr[j*self.n + offset1], 
                    <void *> &corr[i2*nlc1], 
                    nlc1*sizeof(double)
                )
                for I1 in range(nlc1):
                    i1 = idx1[I1]
                    self._corr[(offset1 + i1)*self.n + j] = self._corr[j*self.n + (offset1 + i1)]
        
    cdef void _c_expose_packet(self, packet_LC * packet) noexcept nogil:
        packet.lc_in_mag   = self.lc_in_mag
        packet.NLCs        = self.NLCs
        packet.NResps      = self._NResps
        packet.has_driving = self.has_driving
        packet.has_cov     = self.has_cov
        packet.has_corr    = self.has_corr
        packet.n           = self.n
        # Pointers
        packet.t           = self._t
        packet.y           = self._y
        packet.e           = self._e
        packet.cov         = self._cov
        packet.corr        = self._corr
        packet.npts        = self._npts
        packet.offset      = self._offset


    # Python-accessible routines
    cpdef void memalloc_arr(self, 
        long [:] npts, 
        bint has_driving=True, 
        bint store_cov=False, bint store_corr=False, 
        bint lc_in_mag=False,
    ):
        cdef long NLCs = npts.size
        with nogil:
            if has_driving and NLCs>1:
                self._c_allocate_mem(NLCs, npts[0], &npts[1], store_cov, store_corr, lc_in_mag)
            elif not has_driving and NLCs>0:
                self._c_allocate_mem(NLCs,       0, &npts[0], store_cov, store_corr, lc_in_mag)
            else:
                raise NotImplemented
    
    cpdef void set_single_LC(self, 
        unsigned char lcid, double [:] t, double [:] y, double [:] e,
        bint is_magnitude=False, double magzp=0
    ):
        with nogil:
            self._c_single_LC(lcid, &t[0], &y[0], &e[0], is_magnitude=is_magnitude, magzp=magzp)
    
    cpdef void set_cov(self, unsigned char lcid1, unsigned char lcid2, double [:,:] cov):
        cdef:
            long n1 = self._npts[lcid1]
            long n2 = self._npts[lcid2]
            long cn1, cn2
            double * c
        cn2 = cov.shape[0]
        cn1 = cov.shape[1]
        c = &cov[0,0]
        with nogil:
            if (cn1==n1) and (cn2==n2):
                self._c_cov_pair(lcid1, lcid2, c)
            elif (cn1==n2) and (cn2==n1):
                self._c_cov_pair(lcid2, lcid1, c)
            # else: # Invalid layout!
            #     pass
    cpdef void set_corr(self, unsigned char lcid1, unsigned char lcid2, double [:,:] corr):
        cdef:
            long n1 = self._npts[lcid1]
            long n2 = self._npts[lcid2]
            long cn1, cn2
            double * c
        cn2 = corr.shape[0]
        cn1 = corr.shape[1]
        c = &corr[0,0]
        with nogil:
            if (cn1==n1) and (cn2==n2):
                self._c_corr_pair(lcid1, lcid2, c)
            elif (cn1==n2) and (cn2==n1):
                self._c_corr_pair(lcid2, lcid1, c)
            # else: # Invalid layout!
            #     pass
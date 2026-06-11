# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: cdivision=True

cimport cython
from cython cimport view
from cython.parallel import prange
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy

from .numpy_interface cimport numpy_ascontiguousarray, numpy_ravel_C, numpy_dbl_1d, numpy_dbl_2d, numpy_dbl_3d

from libc.math cimport ( # https://github.com/cython/cython/blob/master/Cython/Includes/libc/math.pxd
    INFINITY, NAN,
    M_PI, M_E, M_LN10, 
    sqrt, log, cos, sin, pow, exp,
    fmin, fmax, fabs, ceil, log2,
)
from .algorithms cimport binary_search_ptr_R

from ..extern.lia cimport ( # BLAS and LAPACK
    lia_mv_solve,
)

#####################################################
# Generalized BSpline Construction
cdef double bspline_basis(long nx, long k, long j, double x_i, double * t) noexcept nogil:
    cdef double denom1, denom2, term1, term2
    if k == 0:
        if (t[j] <= x_i < t[j+1]):
            return 1.0
        elif (x_i == t[nx+k] and t[j+1] == t[nx+k]):
            # base case: include right endpoint of the entire domain
            return 1.0
        else:
            return 0.0
    denom1 = t[j+k  ] - t[j  ]
    denom2 = t[j+k+1] - t[j+1]
    term1 = 0.0
    if denom1 > 0:
        term1 = ((x_i - t[j]) / denom1) * bspline_basis(nx, k-1, j, x_i, t)
    term2 = 0.0
    if denom2 > 0:
        term2 = ((t[j+k+1] - x_i) / denom2) * bspline_basis(nx, k-1, j+1, x_i, t)
    return term1 + term2

cdef void not_a_knot(double * t, long nx, long k, double * x) noexcept nogil:
    # Reverse-engineered scipy.interpolate.make_interp_spline(..., bc_type='not-a-knot')
    # Necessity for reverse engineering: SciPy and other libraries
    #   tend to change their API, while what need to be done here is extremely simple.
    #   Given SciPy is MIT-licensed (which this code is also licensed under as well), 
    #   it is better to implement the necessary function ourself.
    # Reference: 
    #     https://github.com/scipy/scipy/blob/v1.15.2/scipy/interpolate/_bsplines.py#L1008
    cdef long i, k2
    if k % 2 == 1:
        k2 = (k + 1) // 2 # k2 = m
        memcpy(<void*> (t+k+1), <void*> (x+k2), (nx-k-1) * sizeof(double))
    else: # k = 2m
        k2 = k // 2
        for i in range(nx-k-1):
            t[i+k+1] = 0.5*(x[i+k2+1] + x[i+k2])
    for i in range(k+1):
        t[i    ] = x[0  ]
    for i in range(k+1):
        t[nx+k-i] = x[nx-1]
    
cdef void build_full_collocation_matrix(
    double * M, long nx, long k, double * x, double * t
) noexcept nogil:
    cdef long i, j, I
    for i in range(nx): # nbasis = n for this case.
        I = i*nx
        for j in range(nx):
            M[I+j] = bspline_basis(nx, k, j, x[i], t)

cdef void solve_bspline(
    double * t, double * c, 
    long nx, long k, double * x, double * y
) noexcept nogil:
    # t: size n+k+1
    # c: size n
    cdef double * M = <double*> malloc(nx * nx * sizeof(double))
    not_a_knot(t, nx, k, x)
    build_full_collocation_matrix(M, nx, k, x, t)
    lia_mv_solve(c, M, y, nx) # Technically, we can solve for all filters.
    # But we only need this to be run once, so it doesn't need to be efficient.
    free(M)

#####################################################
# Piecewise Polynomial Interpolator (Cubic)
cdef void ppi3_coeff_one(double * coef, 
    # These are taken as input to avoid race conditions. 
    double t1, double t2, double t3, double t4, double t5, double t6,
    double c0, double c1, double c2, double c3,
) noexcept nogil:
    # valid for x in [t3,t4)
    cdef double r41 = 1/(t4-t1)
    cdef double r42 = 1/(t4-t2)
    cdef double r43 = 1/(t4-t3)
    cdef double r52 = 1/(t5-t2)
    cdef double r53 = 1/(t5-t3)
    cdef double r63 = 1/(t6-t3)
    # coef: size 4
    coef[0] = (# order 0
        +c0*(r41*r42*r43*t4*t4*t4)       # basis0
        -c1*(r43*(                       # basis1
            (r41*r42*t1*t4*t4)
          + (r52*t5*(t2*t4*r42+t3*t5*r53))
        ))
        +c2*(                            # basis2
            r42*r43*r52*t4*t2*t2
          + r43*r52*r53*t2*t3*t5
          + r43*r53*r63*t6*t3*t3
        )
        -c3*(r43*r53*r63*t3*t3*t3)       # basis3
    )# end order 0
    coef[1] = (# order 1
        -c0*(3*r41*r42*r43*t4*t4)        # basis0
        -c1*((r41*r42*r43*r52*r53)*(     # basis1
            3*(
                t1*t3*t4*t5 
              + t4*t5*(t2*t3-t4*t5)
              + t1*t2*(t4*t5-t3*(t4+t5)))
        ))
        -c2*(3*r42*r43*r52*r53*r63*(     # basis2
            t2*t2*t3*t3 
          + t2*t4*t5*t6 
          + t3*t4*t5*t6 
          - t2*t3*(t5*t6+t4*(t5+t6))
        ))
        +c3*(3*r43*r53*r63*t3*t3)        # basis3
    )# end order 1
    coef[2] = (# order 2
        +c0*(3*r41*r42*r43*t4)           # basis0
        -c1*((r41*r42*r43*r52*r53)*(     # basis1
            3*t1*t2*t3 
          - 3*(t1+t2+t3-t4)*t4*t5
          + 3*t4*t5*t5
        ))
        +c2*(3*r42*r43*r52*r53*r63*(     # basis2
            t2*t3*(t2+t3-t4-t5) 
          - t2*t3*t6 
          + t4*t5*t6
        ))
        -c3*(3*r43*r53*r63*t3)           # basis3
    )# end order 2
    coef[3] = (# order 3
        -c0*(r41*r42*r43)                # basis0
        +c1*(                            # basis1
            r41*r42*r43 
          + r42*r43*r52
          + r43*r52*r53
        )
        +c2*(r43*r53*(                   # basis2
            r42*r52*(t2+t3-t4-t5)
          - r63
        ))
        +c3*(r43*r53*r63)                # basis3
    )# end order 3

cdef void ppi3_coeff_bulk(double * coef, long nx, double * t, double * c) noexcept nogil:
    # accepts t, c constructed from n interpolating points.
    # k=3.
    # t: n+k+1 = n+4
    # c: n
    # coef: (k+1, n-k) = (4, n-3)
    cdef long i
    for i in prange(nx-3, nogil=True):
        ppi3_coeff_one(
            coef + 4*i, # output memory block
            t[i+1], t[i+2], t[i+3], t[i+4], t[i+5], t[i+6],
            c[i  ], c[i+1], c[i+2], c[i+3],
        )
cdef void ppi3_rightbound(double * x_rb, long nx, double * t) noexcept nogil:
    # n-k-1 = n-4
    memcpy(<void*> (x_rb), <void*> (t+4), (nx-4)*sizeof(double))

cdef void ppi3_coeffs(
    double * x_rb, double * coef, 
    long nx, double *x, double *y
) noexcept nogil:
    # x: size n
    # y: size n
    # coef: size 4*(n-3)
    # x_rb  : size n-4
    cdef double * t = <double*> malloc((nx+4)*sizeof(double))
    cdef double * c = <double*> malloc( nx   *sizeof(double))
    solve_bspline(t, c, nx, 3, x, y)
    ppi3_rightbound(x_rb, nx, t   )
    ppi3_coeff_bulk(coef, nx, t, c)
    free(t)
    free(c)

#####################################################
# Computing Interpolation

cdef double ppi3_eval(double x, double * x_rb, double * coef, long n_rb) noexcept nogil:
    # n_rb = nx-4, the size of x_rb. This should be the target size.
    # x_rb: size n_rb
    # coef: size 4*(n_rb+1)
    cdef double * a = coef + 4*binary_search_ptr_R(x, x_rb, n_rb)
    cdef double x2 = x*x
    cdef double x3 = x*x2
    return a[0] + a[1]*x + a[2]*x2 + a[3]*x3

cdef void ppi3_eval_bulk(double * out, long ndim, double x, double * x_rb, double * coef, long n_rb) noexcept nogil:
    # n_rb = nx-4, the size of x_rb. This should be the target size.
    # x_rb: size n_rb
    # coef: size 4*ndim*(n_rb+1): numpy shape should be (n_rb+1, ndim, 4) for this layout
    cdef double * a = coef + 4*ndim*binary_search_ptr_R(x, x_rb, n_rb)
    cdef double x2 = x*x
    cdef double x3 = x*x2
    cdef long i
    for i in range(ndim):
        out[i] = a[0] + a[1]*x + a[2]*x2 + a[3]*x3
        a = a + 4

cdef long ppi3_eval_bulk_and_idx(double * out, long ndim, double x, double * x_rb, double * coef, long n_rb) noexcept nogil:
    # n_rb = nx-4, the size of x_rb. This should be the target size.
    # x_rb: size n_rb
    # coef: size 4*ndim*(n_rb+1): numpy shape should be (n_rb+1, ndim, 4) for this layout
    cdef long idx = 4*ndim*binary_search_ptr_R(x, x_rb, n_rb)
    cdef double * a = coef + idx
    cdef double x2 = x*x
    cdef double x3 = x*x2
    cdef long i
    for i in range(ndim):
        out[i] = a[0] + a[1]*x + a[2]*x2 + a[3]*x3
        a = a + 4
    return idx

cdef void ppi3_eval_bulk_recycle_idx(double * out, long ndim, double x, double * x_rb, double * coef, long n_rb, long idx) noexcept nogil:
    # n_rb = nx-4, the size of x_rb. This should be the target size.
    # x_rb: size n_rb
    # coef: size 4*ndim*(n_rb+1): numpy shape should be (n_rb+1, ndim, 4) for this layout
    cdef double * a = coef + idx
    cdef double x2 = x*x
    cdef double x3 = x*x2
    cdef long i
    for i in range(ndim):
        out[i] = a[0] + a[1]*x + a[2]*x2 + a[3]*x3
        a = a + 4

#####################################################
# Numpy API

cpdef PPI3_param(object x_in, object y_in): # Make this accept multiple (x, y) simultaneously?
    # make contiguous memory (?)
    xcont = numpy_ascontiguousarray(x_in)
    ycont = numpy_ascontiguousarray(y_in)
    cdef double [:] x = xcont
    cdef double [:] y = ycont
    cdef long nx = x.size
    cdef double * x_rb = <double *> malloc((nx-4)  *sizeof(double))
    cdef double * coef = <double *> malloc((nx-3)*4*sizeof(double))
    with nogil:
        ppi3_coeffs(x_rb, coef, nx, &x[0], &y[0])
    return numpy_dbl_1d(x_rb, nx-4, True),  numpy_dbl_2d(coef, nx-3, 4, True)

cpdef PPI3_coalesce_params(object coef):
    if len(coef)==1:
        # if hasattr(coef, '__iter__') and not hasattr(coef, '__len__'):
        #     y = numpy_ascontiguousarray(tuple(coef)) # Need better solution than this.
        # else:
            raise ValueError("Cannot understand the type of coef, expected a list or tuple of 2d numpy arrays.")
    else:
        y = numpy_ascontiguousarray(coef)
    out = y[0]
    for i in range(1, len(y)):
        out = PPI3_append_params(out, y[i])
    return out

cpdef PPI3_append_params(object this, object other):
    this_c = numpy_ascontiguousarray(this)
    other_c = numpy_ascontiguousarray(other)
    if len(other.shape)!=2:
        raise ValueError("Cannot append to PPI3 coefficients, expected a 2D array for the second argument.")
    if len(this_c.shape)>3 or len(this_c.shape)<2:
        raise ValueError("Cannot append to PPI3 coefficients, expected a 2D or 3D array for the first argument.")
    if len(this_c.shape)==2:
        return __PPI3_append_params_new(this_c, other_c)
    else:
        return __PPI3_append_params_existing(this_c, other_c)

cpdef __PPI3_append_params_new(object this, object other):
    cdef double [:,:] t_c = this
    cdef double [:,:] o_c = other
    cdef long n1 = t_c.shape[0]
    cdef long n2 = 2 # For new array
    cdef long n3 = o_c.shape[1]
    cdef long n21 = n2 - 1
    cdef long i, j
    cdef double * out = <double *> malloc(n1*n2*n3*sizeof(double))
    with nogil:
        for i in range(n1):
            memcpy(<void*> (out + n3*n2*i),    <void*> (&t_c[i, 0]), n21*n3*sizeof(double))
            memcpy(<void*> (out + n3*n2*(i+1)-n3), <void*> (&o_c[i, 0]), n3*sizeof(double))
    return numpy_dbl_3d(out, n1, n2, n3, True)

cpdef __PPI3_append_params_existing(object this, object other):
    cdef double [:,:,:] t_c = this
    cdef double [:,:] o_c = other
    cdef long n1  = t_c.shape[0]
    cdef long n21 = t_c.shape[1]
    cdef long n2  = n21+1
    cdef long n3  = o_c.shape[1]
    cdef long i, j
    cdef double * out = <double *> malloc(n1*n2*n3*sizeof(double))
    with nogil:
        for i in range(n1):
            memcpy(<void*> (out + n3*n2*i), <void*> (&t_c[i, 0, 0]), n21*n3*sizeof(double))
            memcpy(<void*> (out + n3*n2*(i+1)-n3), <void*> (&o_c[i, 0]), n3*sizeof(double))
    return numpy_dbl_3d(out, n1, n2, n3, True)

cpdef PPI3_eval(object x_in, object x_rb, object coef):
    rbc = numpy_ascontiguousarray(x_rb)
    cfc = numpy_ascontiguousarray(coef)
    cdef long ncase = len(cfc.shape)
    if ncase==2:
        return __PPI3_eval_1dim(x_in, rbc, cfc)
    elif ncase== 3:
        return __PPI3_eval_Ndim(x_in, rbc, cfc)
    else:
        raise ValueError("Invalid number of dimensions in coefficients. Expected 2 or 3.")
    
cpdef __PPI3_eval_1dim(object x_in, object x_rb, object coef):
    rbc = numpy_ravel_C(x_rb)
    cfc = numpy_ravel_C(coef)
    cdef double [:] rb = rbc
    cdef double [:] cf = cfc
    cdef long nppi = rb.size
    cdef double xo
    if not hasattr(x_in, '__len__'):
        xo = x_in
        with nogil:
            xo = ppi3_eval(xo, &rb[0], &cf[0], nppi)
        return xo
    x_in_c = numpy_ravel_C(x_in)
    cdef double [:] x = x_in_c
    cdef long nx = x.size
    cdef double * y = <double *> malloc(nx*sizeof(double))
    cdef long i
    with nogil:
        for i in range(nx):
            y[i] = ppi3_eval(x[i], &rb[0], &cf[0], nppi)
    return numpy_dbl_1d(y, nx, True)

cpdef __PPI3_eval_Ndim(object x_in, object x_rb, object coef):
    rbc = numpy_ravel_C(x_rb)
    cfc = numpy_ravel_C(coef)
    cdef double [:] rb = rbc
    cdef double [:] cf = cfc
    cdef long nppi = rb.size
    cdef long ndim = coef.shape[1]
    cdef long nout = ndim
    cdef double * y
    cdef double xo
    if not hasattr(x_in, '__len__'):
        y = <double *> malloc(nout*sizeof(double))
        xo = x_in
        with nogil:
            ppi3_eval_bulk(y, ndim, xo, &rb[0], &cf[0], nppi)
        return numpy_dbl_1d(y, nout, True)
    x_in_c = numpy_ravel_C(x_in)
    cdef double [:] x = x_in_c
    cdef long nx = x.size
    y = <double *> malloc(nout*sizeof(double))
    cdef long i
    with nogil:
        for i in range(nx):
            ppi3_eval_bulk(y+ndim*i, ndim, x[i], &rb[0], &cf[0], nppi)
    return numpy_dbl_2d(y, nx, ndim, True)
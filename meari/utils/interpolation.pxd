
#########################################################
# Interpolator Routines

## Generalized BSpline Construction
cdef double bspline_basis(long nx, long k, long j, double x_i, double * t) noexcept nogil
cdef void solve_bspline(double * t, double * c, long nx, long k, double * x, double * y) noexcept nogil

## Piecewise Cubic Polynomial Interpolator (PPI3) 
# Basically, it unwraps the cubic spline into a piecewise polynomial for more efficient evaluation.
cdef void ppi3_coeffs(double * x_rb, double * coef, long nx, double *x, double *y) noexcept nogil
## Computing PPI3
cdef double ppi3_eval(double x, double * x_rb, double * coef, long N) noexcept nogil
cdef void ppi3_eval_bulk(double * out, long ndim, double x, double * x_rb, double * coef, long n_rb) noexcept nogil
cdef long ppi3_eval_bulk_and_idx(double * out, long ndim, double x, double * x_rb, double * coef, long n_rb) noexcept nogil
cdef void ppi3_eval_bulk_recycle_idx(double * out, long ndim, double x, double * x_rb, double * coef, long n_rb, long idx) noexcept nogil

## Numpy API
cpdef PPI3_append_params(object this, object other)
cpdef PPI3_coalesce_params(object coef)
cpdef PPI3_param(object x_in, object y_in)
cpdef PPI3_eval(object x_in, object x_rb, object coef)
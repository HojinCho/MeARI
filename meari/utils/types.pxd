from libc.stdint import uint32_t

ctypedef float complex complex_float
ctypedef double complex complex_double

ctypedef fused Numeric:
    char
    short
    int
    long
    long long
    float
    double
    long double

ctypedef fused Integer:
    char
    short
    int
    long
    long long

ctypedef fused Real:
    float
    double
    long double

ctypedef double (*func_t_double_par)(double x, void *params) noexcept nogil
ctypedef double (*func_t_double2_par)(double x, double y, void *params) noexcept nogil
ctypedef double (*func_t_double_ternary)(double x, double y, double z) noexcept nogil

ctypedef void (*func_t_void_dblarr)(long n, double * x, void *params) noexcept nogil
ctypedef void (*func_t_void_dblarr_auxarr)(long n, double * x, double **aux, void *params) noexcept nogil
# How to use void ** correctly: https://stackoverflow.com/a/9040946/4755229
ctypedef void (*func_t_void_arr)(long n, void * x, void *params) noexcept nogil
ctypedef void (*func_t_void_arr_auxarr)(long n, void * x, void **aux, void *params, void *auxpar) noexcept nogil

ctypedef void (*func_t_void_void_ptr)(void * x) noexcept nogil
ctypedef double (*func_t_double_void_ptr)(void * x) noexcept nogil

ctypedef void (* func_t_param_2)(void * par, double x1, double x2) noexcept nogil
ctypedef void (* func_t_param_dblptr2)(void * par, double * x1, double * x2) noexcept nogil
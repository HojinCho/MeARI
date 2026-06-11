from libc.stdint cimport int_fast32_t

cdef extern from "lapack.h":
    # This shouldn't necessarily be a cdef extern, lapack.h should exist anyway.
    # ctypedef int32_t lapack_int
    ctypedef int_fast32_t lapack_int # Maybe this is safer?
#     ctypedef lapack_logical (*LAPACK_S_SELECT2) (float*,  float* )
#     ctypedef lapack_logical (*LAPACK_S_SELECT3) (float*,  float*,  float* )
#     ctypedef lapack_logical (*LAPACK_D_SELECT2) (double*, double* )
#     ctypedef lapack_logical (*LAPACK_D_SELECT3) (double*, double*, double* )
#     ctypedef lapack_logical (*LAPACK_C_SELECT1) (lapack_complex_float* )
#     ctypedef lapack_logical (*LAPACK_C_SELECT2) (lapack_complex_float*, lapack_complex_float* )
#     ctypedef lapack_logical (*LAPACK_Z_SELECT1) (lapack_complex_double* )
#     ctypedef lapack_logical (*LAPACK_Z_SELECT2) (lapack_complex_double*,lapack_complex_double* )

cdef extern from "lapacke.h":
    cdef const int LAPACK_ROW_MAJOR
    cdef const int LAPACK_COL_MAJOR
    cdef const int LAPACK_WORK_MEMORY_ERROR
    cdef const int LAPACK_TRANSPOSE_MEMORY_ERROR
    # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pptri.html
    # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pptrf.html
    lapack_int LAPACKE_dpptri(int matrix_layout, char uplo, lapack_int n, double* ap) noexcept nogil
    lapack_int LAPACKE_dpptrf(int matrix_layout, char uplo, lapack_int n, double* ap) noexcept nogil
    lapack_int LAPACKE_dpotrf(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda) noexcept nogil
    lapack_int LAPACKE_dpotri(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda) noexcept nogil
    lapack_int LAPACKE_dsytrf(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda, lapack_int * ipiv) noexcept nogil
    lapack_int LAPACKE_dsytri(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda, lapack_int * ipiv) noexcept nogil
    lapack_int LAPACKE_dsytri2_work(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda, const lapack_int * ipiv, double * work, lapack_int lwork) noexcept nogil
    lapack_int LAPACKE_dgesv (int matrix_layout , lapack_int n , lapack_int nrhs , double * a , lapack_int lda , lapack_int * ipiv , double * b , lapack_int ldb ) noexcept nogil

ctypedef enum LAPACK_LAYOUT:
    LapackRowMajor = LAPACK_ROW_MAJOR
    LapackColMajor = LAPACK_COL_MAJOR


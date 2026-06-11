from libc.stdint cimport int32_t, int_fast32_t

from ...utils.types cimport complex_float, complex_double
# ctypedef float complex complex_float
# ctypedef double complex complex_double

# Sometimes it's better to simply copy and paste and use regex

cdef extern from "cblas.h":
    ctypedef size_t CBLAS_INDEX
    ctypedef enum CBLAS_ORDER:
        CblasRowMajor
        CblasColMajor

    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans
        CblasTrans
        CblasConjTrans
        CblasConjNoTrans

    ctypedef enum CBLAS_UPLO:
        CblasUpper
        CblasLower

    ctypedef enum CBLAS_DIAG:
        CblasNonUnit
        CblasUnit

    ctypedef enum CBLAS_SIDE:
        CblasLeft
        CblasRight

    float  cblas_sdsdot(int n, float alpha, float *x, int incx, float *y, int incy) noexcept nogil
    double cblas_dsdot (int n, float *x, int incx, float *y, int incy) noexcept nogil
    float  cblas_sdot(int n, float  *x, int incx, float  *y, int incy) noexcept nogil
    double cblas_ddot(int n, double *x, int incx, double *y, int incy) noexcept nogil

    complex_float cblas_cdotu(int n, void  *x, int incx, void  *y, int incy) noexcept nogil
    complex_float  cblas_cdotc(int n, void  *x, int incx, void  *y, int incy) noexcept nogil
    complex_double cblas_zdotu(int n, void *x, int incx, void *y, int incy) noexcept nogil
    complex_double cblas_zdotc(int n, void *x, int incx, void *y, int incy) noexcept nogil

    void  cblas_cdotu_sub(int n, void  *x, int incx, void  *y, int incy, void  *ret) noexcept nogil
    void  cblas_cdotc_sub(int n, void  *x, int incx, void  *y, int incy, void  *ret) noexcept nogil
    void  cblas_zdotu_sub(int n, void *x, int incx, void *y, int incy, void *ret) noexcept nogil
    void  cblas_zdotc_sub(int n, void *x, int incx, void *y, int incy, void *ret) noexcept nogil

    float  cblas_sasum(int n, float  *x, int incx) noexcept nogil
    double cblas_dasum(int n, double *x, int incx) noexcept nogil
    float  cblas_scasum(int n, void *x, int incx) noexcept nogil
    double cblas_dzasum(int n, void *x, int incx) noexcept nogil

    float  cblas_ssum (int n, float  *x, int incx) noexcept nogil
    double cblas_dsum (int n, double *x, int incx) noexcept nogil
    float  cblas_scsum(int n, void  *x, int incx) noexcept nogil
    double cblas_dzsum(int n, void *x, int incx) noexcept nogil

    float  cblas_snrm2(int N, float  *X, int incX) noexcept nogil
    double cblas_dnrm2(int N, double *X, int incX) noexcept nogil
    float  cblas_scnrm2(int N, void  *X, int incX) noexcept nogil
    double cblas_dznrm2(int N, void  *X, int incX) noexcept nogil

    CBLAS_INDEX cblas_isamax(int n, float  *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_idamax(int n, double *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_icamax(int n, void *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_izamax(int n, void *x, int incx) noexcept nogil

    CBLAS_INDEX cblas_isamin(int n, float  *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_idamin(int n, double *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_icamin(int n, void *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_izamin(int n, void *x, int incx) noexcept nogil

    float cblas_samax(int n, float  *x, int incx) noexcept nogil
    double cblas_damax(int n, double *x, int incx) noexcept nogil
    float cblas_scamax(int n, void  *x, int incx) noexcept nogil
    double cblas_dzamax(int n, void *x, int incx) noexcept nogil

    float cblas_samin(int n, float  *x, int incx) noexcept nogil
    double cblas_damin(int n, double *x, int incx) noexcept nogil
    float cblas_scamin(int n, void  *x, int incx) noexcept nogil
    double cblas_dzamin(int n, void *x, int incx) noexcept nogil

    CBLAS_INDEX cblas_ismax(int n, float  *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_idmax(int n, double *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_icmax(int n, void *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_izmax(int n, void *x, int incx) noexcept nogil

    CBLAS_INDEX cblas_ismin(int n, float  *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_idmin(int n, double *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_icmin(int n, void *x, int incx) noexcept nogil
    CBLAS_INDEX cblas_izmin(int n, void *x, int incx) noexcept nogil

    void cblas_saxpy(int n, float alpha, float *x, int incx, float *y, int incy) noexcept nogil
    void cblas_daxpy(int n, double alpha, double *x, int incx, double *y, int incy) noexcept nogil
    void cblas_caxpy(int n, void *alpha, void *x, int incx, void *y, int incy) noexcept nogil
    void cblas_zaxpy(int n, void *alpha, void *x, int incx, void *y, int incy) noexcept nogil

    void cblas_caxpyc(int n, void *alpha, void *x, int incx, void *y, int incy) noexcept nogil
    void cblas_zaxpyc(int n, void *alpha, void *x, int incx, void *y, int incy) noexcept nogil

    void cblas_scopy(int n, float *x, int incx, float *y, int incy) noexcept nogil
    void cblas_dcopy(int n, double *x, int incx, double *y, int incy) noexcept nogil
    void cblas_ccopy(int n, void *x, int incx, void *y, int incy) noexcept nogil
    void cblas_zcopy(int n, void *x, int incx, void *y, int incy) noexcept nogil

    void cblas_sswap(int n, float *x, int incx, float *y, int incy) noexcept nogil
    void cblas_dswap(int n, double *x, int incx, double *y, int incy) noexcept nogil
    void cblas_cswap(int n, void *x, int incx, void *y, int incy) noexcept nogil
    void cblas_zswap(int n, void *x, int incx, void *y, int incy) noexcept nogil

    void cblas_srot(int N, float *X, int incX, float *Y, int incY, float c, float s) noexcept nogil
    void cblas_drot(int N, double *X, int incX, double *Y, int incY, double c, double  s) noexcept nogil
    void cblas_csrot(int n, void *x, int incx, void *y, int incY, float c, float s) noexcept nogil
    void cblas_zdrot(int n, void *x, int incx, void *y, int incY, double c, double s) noexcept nogil

    void cblas_srotg(float *a, float *b, float *c, float *s) noexcept nogil
    void cblas_drotg(double *a, double *b, double *c, double *s) noexcept nogil
    void cblas_crotg(void *a, void *b, float *c, void *s) noexcept nogil
    void cblas_zrotg(void *a, void *b, double *c, void *s) noexcept nogil

    void cblas_srotm(int N, float *X, int incX, float *Y, int incY, float *P) noexcept nogil
    void cblas_drotm(int N, double *X, int incX, double *Y, int incY, double *P) noexcept nogil

    void cblas_srotmg(float *d1, float *d2, float *b1, float b2, float *P) noexcept nogil
    void cblas_drotmg(double *d1, double *d2, double *b1, double b2, double *P) noexcept nogil

    void cblas_sscal(int N, float alpha, float *X, int incX) noexcept nogil
    void cblas_dscal(int N, double alpha, double *X, int incX) noexcept nogil
    void cblas_cscal(int N, void *alpha, void *X, int incX) noexcept nogil
    void cblas_zscal(int N, void *alpha, void *X, int incX) noexcept nogil
    void cblas_csscal(int N, float alpha, void *X, int incX) noexcept nogil
    void cblas_zdscal(int N, double alpha, void *X, int incX) noexcept nogil

    void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,  int m, int n,
            float alpha, float  *a, int lda,  float  *x, int incx,  float beta,  float  *y, int incy) noexcept nogil
    void cblas_dgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,  int m, int n,
            double alpha, double  *a, int lda,  double  *x, int incx,  double beta,  double  *y, int incy) noexcept nogil
    void cblas_cgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,  int m, int n,
            void *alpha, void  *a, int lda,  void  *x, int incx,  void *beta,  void  *y, int incy) noexcept nogil
    void cblas_zgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,  int m, int n,
            void *alpha, void  *a, int lda,  void  *x, int incx,  void *beta,  void  *y, int incy) noexcept nogil

    void cblas_sger (CBLAS_ORDER order, int M, int N, float   alpha, float  *X, int incX, float  *Y, int incY, float  *A, int lda) noexcept nogil
    void cblas_dger (CBLAS_ORDER order, int M, int N, double  alpha, double *X, int incX, double *Y, int incY, double *A, int lda) noexcept nogil
    void cblas_cgeru(CBLAS_ORDER order, int M, int N, void  *alpha, void  *X, int incX, void  *Y, int incY, void  *A, int lda) noexcept nogil
    void cblas_cgerc(CBLAS_ORDER order, int M, int N, void  *alpha, void  *X, int incX, void  *Y, int incY, void  *A, int lda) noexcept nogil
    void cblas_zgeru(CBLAS_ORDER order, int M, int N, void *alpha, void *X, int incX, void *Y, int incY, void *A, int lda) noexcept nogil
    void cblas_zgerc(CBLAS_ORDER order, int M, int N, void *alpha, void *X, int incX, void *Y, int incY, void *A, int lda) noexcept nogil

    void cblas_strsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, float *A, int lda, float *X, int incX) noexcept nogil
    void cblas_dtrsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, double *A, int lda, double *X, int incX) noexcept nogil
    void cblas_ctrsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *A, int lda, void *X, int incX) noexcept nogil
    void cblas_ztrsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *A, int lda, void *X, int incX) noexcept nogil

    void cblas_strmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, float *A, int lda, float *X, int incX) noexcept nogil
    void cblas_dtrmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, double *A, int lda, double *X, int incX) noexcept nogil
    void cblas_ctrmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *A, int lda, void *X, int incX) noexcept nogil
    void cblas_ztrmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *A, int lda, void *X, int incX) noexcept nogil

    void cblas_ssyr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *X, int incX, float *A, int lda) noexcept nogil
    void cblas_dsyr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *X, int incX, double *A, int lda) noexcept nogil
    void cblas_cher(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, void *X, int incX, void *A, int lda) noexcept nogil
    void cblas_zher(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, void *X, int incX, void *A, int lda) noexcept nogil

    void cblas_ssyr2(CBLAS_ORDER order, CBLAS_UPLO Uplo,int N, float alpha, float *X,
                    int incX, float *Y, int incY, float *A, int lda) noexcept nogil
    void cblas_dsyr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *X,
                    int incX, double *Y, int incY, double *A, int lda) noexcept nogil
    void cblas_cher2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *X, int incX,
                    void *Y, int incY, void *A, int lda) noexcept nogil
    void cblas_zher2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *X, int incX,
                    void *Y, int incY, void *A, int lda) noexcept nogil

    void cblas_sgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N,
                    int KL, int KU, float alpha, float *A, int lda, float *X, int incX, float beta, float *Y, int incY) noexcept nogil
    void cblas_dgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N,
                    int KL, int KU, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY) noexcept nogil
    void cblas_cgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N,
                    int KL, int KU, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY) noexcept nogil
    void cblas_zgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N,
                    int KL, int KU, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY) noexcept nogil

    void cblas_ssbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, int K, float alpha, float *A,
                    int lda, float *X, int incX, float beta, float *Y, int incY) noexcept nogil
    void cblas_dsbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, int K, double alpha, double *A,
                    int lda, double *X, int incX, double beta, double *Y, int incY) noexcept nogil

    void cblas_stbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, int K, float *A, int lda, float *X, int incX) noexcept nogil
    void cblas_dtbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, int K, double *A, int lda, double *X, int incX) noexcept nogil
    void cblas_ctbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, int K, void *A, int lda, void *X, int incX) noexcept nogil
    void cblas_ztbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, int K, void *A, int lda, void *X, int incX) noexcept nogil

    void cblas_stbsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, int K, float *A, int lda, float *X, int incX) noexcept nogil
    void cblas_dtbsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, int K, double *A, int lda, double *X, int incX) noexcept nogil
    void cblas_ctbsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, int K, void *A, int lda, void *X, int incX) noexcept nogil
    void cblas_ztbsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, int K, void *A, int lda, void *X, int incX) noexcept nogil

    void cblas_stpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, float *Ap, float *X, int incX) noexcept nogil
    void cblas_dtpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, double *Ap, double *X, int incX) noexcept nogil
    void cblas_ctpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, void *Ap, void *X, int incX) noexcept nogil
    void cblas_ztpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, void *Ap, void *X, int incX) noexcept nogil

    void cblas_stpsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, float *Ap, float *X, int incX) noexcept nogil
    void cblas_dtpsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, double *Ap, double *X, int incX) noexcept nogil
    void cblas_ctpsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, void *Ap, void *X, int incX) noexcept nogil
    void cblas_ztpsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                    int N, void *Ap, void *X, int incX) noexcept nogil

    void cblas_ssymv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *A,
                    int lda, float *X, int incX, float beta, float *Y, int incY) noexcept nogil
    void cblas_dsymv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *A,
                    int lda, double *X, int incX, double beta, double *Y, int incY) noexcept nogil
    void cblas_chemv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *A,
                    int lda, void *X, int incX, void *beta, void *Y, int incY) noexcept nogil
    void cblas_zhemv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *A,
                    int lda, void *X, int incX, void *beta, void *Y, int incY) noexcept nogil

    void cblas_sspmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *Ap,
                    float *X, int incX, float beta, float *Y, int incY) noexcept nogil
    void cblas_dspmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *Ap,
                    double *X, int incX, double beta, double *Y, int incY) noexcept nogil

    void cblas_sspr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *X, int incX, float *Ap) noexcept nogil
    void cblas_dspr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *X, int incX, double *Ap) noexcept nogil

    void cblas_chpr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, void *X, int incX, void *A) noexcept nogil
    void cblas_zhpr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, void *X,int incX, void *A) noexcept nogil

    void cblas_sspr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *X, int incX, float *Y, int incY, float *A) noexcept nogil
    void cblas_dspr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *X, int incX, double *Y, int incY, double *A) noexcept nogil
    void cblas_chpr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *X, int incX, void *Y, int incY, void *Ap) noexcept nogil
    void cblas_zhpr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *X, int incX, void *Y, int incY, void *Ap) noexcept nogil

    void cblas_chbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, int K,
            void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY) noexcept nogil
    void cblas_zhbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, int K,
            void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY) noexcept nogil

    void cblas_chpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N,
            void *alpha, void *Ap, void *X, int incX, void *beta, void *Y, int incY) noexcept nogil
    void cblas_zhpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N,
            void *alpha, void *Ap, void *X, int incX, void *beta, void *Y, int incY) noexcept nogil

    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K,
            float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc) noexcept nogil
    void cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K,
            double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) noexcept nogil
    void cblas_cgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K,
            void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil
    void cblas_cgemm3m(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K,
            void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil
    void cblas_zgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K,
            void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil
    void cblas_zgemm3m(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K,
            void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil

    void cblas_sgemmt(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int K,
            float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc) noexcept nogil
    void cblas_dgemmt(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int K,
            double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) noexcept nogil
    void cblas_cgemmt(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int K,
            void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil
    void cblas_zgemmt(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int K,
            void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil

    void cblas_ssymm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N,
                    float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc) noexcept nogil
    void cblas_dsymm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N,
                    double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) noexcept nogil
    void cblas_csymm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N,
                    void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil
    void cblas_zsymm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N,
                    void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil

    void cblas_ssyrk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
            int N, int K, float alpha, float *A, int lda, float beta, float *C, int ldc) noexcept nogil
    void cblas_dsyrk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
            int N, int K, double alpha, double *A, int lda, double beta, double *C, int ldc) noexcept nogil
    void cblas_csyrk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
            int N, int K, void *alpha, void *A, int lda, void *beta, void *C, int ldc) noexcept nogil
    void cblas_zsyrk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
            int N, int K, void *alpha, void *A, int lda, void *beta, void *C, int ldc) noexcept nogil

    void cblas_ssyr2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
            int N, int K, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc) noexcept nogil
    void cblas_dsyr2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
            int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) noexcept nogil
    void cblas_csyr2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
            int N, int K, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil
    void cblas_zsyr2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
            int N, int K, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil

    void cblas_strmm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                    CBLAS_DIAG Diag, int M, int N, float alpha, float *A, int lda, float *B, int ldb) noexcept nogil
    void cblas_dtrmm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                    CBLAS_DIAG Diag, int M, int N, double alpha, double *A, int lda, double *B, int ldb) noexcept nogil
    void cblas_ctrmm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                    CBLAS_DIAG Diag, int M, int N, void *alpha, void *A, int lda, void *B, int ldb) noexcept nogil
    void cblas_ztrmm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                    CBLAS_DIAG Diag, int M, int N, void *alpha, void *A, int lda, void *B, int ldb) noexcept nogil

    void cblas_strsm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                    CBLAS_DIAG Diag, int M, int N, float alpha, float *A, int lda, float *B, int ldb) noexcept nogil
    void cblas_dtrsm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                    CBLAS_DIAG Diag, int M, int N, double alpha, double *A, int lda, double *B, int ldb) noexcept nogil
    void cblas_ctrsm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                    CBLAS_DIAG Diag, int M, int N, void *alpha, void *A, int lda, void *B, int ldb) noexcept nogil
    void cblas_ztrsm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                    CBLAS_DIAG Diag, int M, int N, void *alpha, void *A, int lda, void *B, int ldb) noexcept nogil

    void cblas_chemm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N,
                    void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil
    void cblas_zhemm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N,
                    void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc) noexcept nogil

    void cblas_cherk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K,
                    float alpha, void *A, int lda, float beta, void *C, int ldc) noexcept nogil
    void cblas_zherk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K,
                    double alpha, void *A, int lda, double beta, void *C, int ldc) noexcept nogil

    void cblas_cher2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K,
                    void *alpha, void *A, int lda, void *B, int ldb, float beta, void *C, int ldc) noexcept nogil
    void cblas_zher2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K,
                    void *alpha, void *A, int lda, void *B, int ldb, double beta, void *C, int ldc) noexcept nogil

    # void cblas_xerbla(int p, char *rout, char *form, ...) noexcept nogil
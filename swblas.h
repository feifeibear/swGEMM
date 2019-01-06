#ifndef _SWBLAS_H_
#define _SWBLAS_H_ 
#include "cblas.h"
#ifdef __cplusplus
extern "C" {
#endif
void sw_sgemm_trans(float* input, float* weight, float* output, int M, int N, int K, int blkM, int blkN, int blkK);
void sw_sgemm(float* input, float* weight, float* output, int M, int N, int K, int blkM);

void sw_cblas_sgemm_unit(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

void sw_cblas_sgemm_nopad(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);



void sw_cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);


void sw_blas_init();
void sw_blas_stop();
void *_aligned_malloc(size_t size, size_t align_size);
void _aligned_free(void *ptr);

void zeropad_matrix(const float* A, int ld, int ld_pad, int hd, int hd_pad, float* A_zeropad);
void sw_zeropad_matrix(const float* A, int ld, int ld_pad, int hd, int hd_pad, float* A_zeropad);
void lw_zeropad_matrix(const float* A, int ld, int ld_valid, int ld_pad, int hd, int hd_valid, int hd_pad, float* A_zeropad);
void depad_matrix(float* A, int ld, int ld_pad, int hd, int hd_pad, const float* A_zeropad);
void sw_depad_matrix(float* A, int ld, int ld_pad, int hd, int hd_pad, const float* A_zeropad);
#ifdef __cplusplus
}
#endif




#endif

/* ****
 * Implement a GEMM
 * Jerry Fang
 * 2018.9.15
 * ****/
#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include "simd.h"
#include "swblas.h"
#include <cblas.h>

#define TIME(a,b) (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))


/*********
 * A standard interface unitest function
 * check the correctness with cblas interface
 * currently support Trans, NoTrans case 
 * *******/
void test_sw_dgemm_Atrans_std(int M, int N, int K) {
  int i;
  int cM, cN, cK, cT;
  struct timeval t1, t2;
  int ld = 15;
  int M2 = M + ld;
  int N2 = N + ld;
  int K2 = K + ld;

  int lda = M2;
  int ldb = N2;
  int ldc = N2;
#undef _MEM_128BALIGN_
#ifdef _MEM_128BALIGN_
  double* C = (double*)_aligned_malloc(sizeof(double)*N2*M2, 128);
  double* C_blas = (double*)_aligned_malloc(sizeof(double)*N2*M2, 128);
  double* A = (double*)_aligned_malloc(sizeof(double)*K2*M2, 128);
  double* B = (double*)_aligned_malloc(sizeof(double)*K2*N2, 128);
  //double* Ap = (double*)_aligned_malloc(sizeof(double)*K2*M2*4, 128);
  //double* Bp = (double*)_aligned_malloc(sizeof(double)*K2*N2*4, 128);
  //double* Cp = (double*)_aligned_malloc(sizeof(double)*M2*N2*4, 128);
#else
  double* C = (double*)malloc(sizeof(double)*N2*M2);
  double* C_blas = (double*)malloc(sizeof(double)*N2*M2);
  double* A = (double*)malloc(sizeof(double)*K2*M2);
  double* B = (double*)malloc(sizeof(double)*K2*N2);
  //double* Ap = (double*)malloc(sizeof(double)*K2*M2*4);
  //double* Bp = (double*)malloc(sizeof(double)*K2*N2*4);
  //double* Cp = (double*)malloc(sizeof(double)*M2*N2*4);
#endif

  srand((unsigned int) time(NULL));
  for(i=0; i < K2*M2; i++){
    A[i] = 1.0; //rand()*1.0/RAND_MAX;
  }
  for(i=0; i < K2*N2; i++){
    B[i] = 1.0; //rand()*1.0/RAND_MAX;
  }
  for(i=0; i < N2*M2; i++){
    C[i] = 0.0;
    C_blas[i] = 0.0;
  }
  double gflop = (double)2*K/1024*N/1024*M/1024;

  //(M*T, K) * (K, N)
#ifdef USE_COMP
  printf("TIME compute part\n");
#else
  printf("no TIME compute part\n");
#endif

  double alpha = 1.0;
  double beta = 0.0;
  gettimeofday(&t1, NULL);
  //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
  //cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
  char TRANSA = 'N';
  char TRANSB = 'T';
  //sgemm_(&TRANSA, &TRANSB, &N, &M, &K, &alpha, B, &N, A, &M, &beta, C, &N);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);

  gettimeofday(&t2, NULL);
  double tt = TIME(t1,t2);
  double gflops = gflop / tt;
  printf("xMath gflops is %.2lf time %lf sec\n", gflops, tt);

  gettimeofday(&t1, NULL);
  //sw_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C_blas, N);
  sw_cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C_blas, ldc);
  gettimeofday(&t2, NULL);
  tt = TIME(t1,t2);
  gflops = gflop / tt;
  printf("dgemm gflops is %.2lf time %lf sec\n", gflops, tt);

  printf("check res\n");
#ifdef CHECK_RES
  int cnt = 0;
  long double sum1 = 0., sum2 = 0.;
  cnt = 0;
  printf("C length : %d\n", N*M);

  int ii, j;
  for(i = 0; i < M2; ++i) for(j = 0; j < N; j++) {
    ii = i*ldc + j;
    //printf("now %d\n", i);
    //printf("now cblas %.1f xmath %f\n", C_blas[i], C[i]);
    if(fabs(C[ii] - C_blas[ii]) > 1e-3) {
      if(cnt < 1000) printf("error @ (%d %d), %lf vs %lf\n", i, j, C[ii], C_blas[ii]);
      cnt++;
    }
    sum1 += C[ii];
    sum2 += C_blas[ii];
  }
  printf("pass validation test! xmath sum1: %.2lf dgemm sum2: %.2lf, error_cnt: %d\n", sum1, sum2, cnt);
  fflush(stdout);
#endif

#ifdef _MEM_128BALIGN_
  _aligned_free(A);
  _aligned_free(B);
  _aligned_free(C);
  _aligned_free(C_blas);
  //_aligned_free(Ap);
  //_aligned_free(Bp);
  //_aligned_free(Cp);
#else
  free(A);
  free(B);
  free(C);
  free(C_blas);
  //free(Ap);
  //free(Bp);
  //free(Cp);
#endif
  return;
}

void test_sw_sgemm_Atrans_std(int M, int N, int K) {
  int i;
  int cM, cN, cK, cT;
  struct timeval t1, t2;
#ifdef _MEM_128BALIGN_
  float* C = (float*)_aligned_malloc(sizeof(float)*N*M, 128);
  float* C_blas = (float*)_aligned_malloc(sizeof(float)*N*M, 128);
  float* A = (float*)_aligned_malloc(sizeof(float)*K*M, 128);
  float* B = (float*)_aligned_malloc(sizeof(float)*K*N, 128);
  float* Ap = (float*)_aligned_malloc(sizeof(float)*K*M*4, 128);
  float* Bp = (float*)_aligned_malloc(sizeof(float)*K*N*4, 128);
  float* Cp = (float*)_aligned_malloc(sizeof(float)*M*N*4, 128);
#else
  float* C = (float*)malloc(sizeof(float)*N*M);
  float* C_blas = (float*)malloc(sizeof(float)*N*M);
  float* A = (float*)malloc(sizeof(float)*K*M);
  float* B = (float*)malloc(sizeof(float)*K*N);
  float* Ap = (float*)malloc(sizeof(float)*K*M*4);
  float* Bp = (float*)malloc(sizeof(float)*K*N*4);
  float* Cp = (float*)malloc(sizeof(float)*M*N*4);
#endif

  srand((unsigned int) time(NULL));
  for(i=0; i < K*M; i++){
    A[i] = 1.0; //rand()*1.0/RAND_MAX;
  }
  for(i=0; i < K*N; i++){
    B[i] = 1.0; //rand()*1.0/RAND_MAX;
  }
  for(i=0; i < N*M; i++){
    C[i] = 0.0;
    C_blas[i] = 1.0;
  }
  double gflop = (double)2*K/1024*N/1024*M/1024;

  //(M*T, K) * (K, N)
#ifdef USE_COMP
  printf("TIME compute part\n");
#else
  printf("no TIME compute part\n");
#endif

  double alpha = 1.0;
  double beta = 0.0;
  gettimeofday(&t1, NULL);
  //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
  //cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
  char TRANSA = 'N';
  char TRANSB = 'T';
  //sgemm_(&TRANSA, &TRANSB, &N, &M, &K, &alpha, B, &N, A, &M, &beta, C, &N);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, M, K, alpha, B, N, A, M, beta, C, N);

  gettimeofday(&t2, NULL);
  double tt = TIME(t1,t2);
  double gflops = gflop / tt;
  printf("xMath gflops is %.2lf time %lf sec\n", gflops, tt);

  gettimeofday(&t1, NULL);
  //sw_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C_blas, N);
  sw_cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C_blas, N);
  gettimeofday(&t2, NULL);
  tt = TIME(t1,t2);
  gflops = gflop / tt;
  printf("sgemm gflops is %.2lf time %lf sec\n", gflops, tt);

  printf("check res\n");
#ifdef CHECK_RES
  int cnt = 0;
  long double sum1 = 0., sum2 = 0.;
  cnt = 0;
  printf("C length : %d\n", N*M);
  for(i = 0; i < N*M; ++i) {
    //printf("now %d\n", i);
    //printf("now cblas %.1f xmath %f\n", C_blas[i], C[i]);
    if(fabs(C[i] - C_blas[i]) > 1e-3 && cnt < 10) {
      printf("error @ %d, %lf vs %lf\n", i, C[i], C_blas[i]);
      cnt++;
    }
    sum1 += C[i];
    sum2 += C_blas[i];
  }
  printf("pass validation test! xmath sum1: %.2lf sgemm sum2: %.2lf\n", sum1, sum2);
  fflush(stdout);
#endif

#ifdef _MEM_128BALIGN_
  _aligned_free(A);
  _aligned_free(B);
  _aligned_free(C);
  _aligned_free(C_blas);
  _aligned_free(Ap);
  _aligned_free(Bp);
  _aligned_free(Cp);
#else
  free(A);
  free(B);
  free(C);
  free(C_blas);
  free(Ap);
  free(Bp);
  free(Cp);
#endif
  return;
}

/*********
 * A standard interface unitest function
 * check the correctness with cblas interface
 * Padding
 * *******/
void test_sw_sgemm_Anotrans_std(int M, int N, int K) {
  int i;
  int cM, cN, cK, cT;
  struct timeval t1, t2;
#ifdef _MEM_128BALIGN_
  float* C = (float*)_aligned_malloc(sizeof(float)*N*M, 128);
  float* C_blas = (float*)_aligned_malloc(sizeof(float)*N*M, 128);
  float* A = (float*)_aligned_malloc(sizeof(float)*K*M, 128);
  float* B = (float*)_aligned_malloc(sizeof(float)*K*N, 128);
  float* Ap = (float*)_aligned_malloc(sizeof(float)*K*M*4, 128);
  float* Bp = (float*)_aligned_malloc(sizeof(float)*K*N*4, 128);
  float* Cp = (float*)_aligned_malloc(sizeof(float)*M*N*4, 128);
#else
  float* C = (float*)malloc(sizeof(float)*N*M);
  float* C_blas = (float*)malloc(sizeof(float)*N*M);
  float* A = (float*)malloc(sizeof(float)*K*M);
  float* B = (float*)malloc(sizeof(float)*K*N);
  float* Ap = (float*)malloc(sizeof(float)*K*M*4);
  float* Bp = (float*)malloc(sizeof(float)*K*N*4);
  float* Cp = (float*)malloc(sizeof(float)*M*N*4);
#endif

  srand((unsigned int) time(NULL));
  for(i=0; i < K*M; i++){
    A[i] = 1.0; //rand()*1.0/RAND_MAX;
  }
  for(i=0; i < K*N; i++){
    B[i] = 1.0; //rand()*1.0/RAND_MAX;
  }
  for(i=0; i < N*M; i++){
    C[i] = 0.0;
    C_blas[i] = 0.0;
  }
  double gflop = (double)2*K/1024*N/1024*M/1024;

  //(M*T, K) * (K, N)
#ifdef USE_COMP
  printf("TIME compute part\n");
#else
  printf("no TIME compute part\n");
#endif

  double alpha = 1.0;
  double beta = 0.0;
  gettimeofday(&t1, NULL);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
  gettimeofday(&t2, NULL);
  double tt = TIME(t1,t2);
  double gflops = gflop / tt;
  printf("xMath gflops is %.2lf time %lf sec\n", gflops, tt);

  gettimeofday(&t1, NULL);
  sw_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C_blas, N);
  gettimeofday(&t2, NULL);
  tt = TIME(t1,t2);
  gflops = gflop / tt;
  printf("sgemm gflops is %.2lf time %lf sec\n", gflops, tt);

  printf("check res\n");
#ifdef CHECK_RES
  int cnt = 0;
  long double sum1 = 0., sum2 = 0.;
  cnt = 0;
  printf("C length : %d\n", N*M);
  for(i = 0; i < N*M; ++i) {
    //printf("now %d\n", i);
    //printf("now cblas %.1f xmath %f\n", C_blas[i], C[i]);
    if(fabs(C[i] - C_blas[i]) > 1e-3 && cnt < 10) {
      printf("error @ %d, %lf vs %lf\n", i, C[i], C_blas[i]);
      cnt++;
    }
    sum1 += C[i];
    sum2 += C_blas[i];
  }
  printf("pass validation test! swblas sum1: %.2lf xmath sum2: %.2lf\n", sum1, sum2);
  fflush(stdout);
#endif

#ifdef _MEM_128BALIGN_
  _aligned_free(A);
  _aligned_free(B);
  _aligned_free(C);
  _aligned_free(C_blas);
  _aligned_free(Ap);
  _aligned_free(Bp);
  _aligned_free(Cp);
#else
  free(A);
  free(B);
  free(C);
  free(C_blas);
  free(Ap);
  free(Bp);
  free(Cp);
#endif
  return;
}

/*********
 * A depracated interface function
 * check the correctness with cblas interface
 * support Trans, NoTrans case
 * No padding N 128x M 32x K > 16 8x
 * *******/
void test_sw_sgemm_Atrans_nopad_std(int M, int N, int K) {
  int i;
  int cM, cN, cK, cT;
  struct timeval t1, t2;
#ifdef _MEM_128BALIGN_
  float* C_blas = (float*)_aligned_malloc(sizeof(float)*N*M, 128);
  float* C = (float*)_aligned_malloc(sizeof(float)*N*M, 128);
  float* A = (float*)_aligned_malloc(sizeof(float)*K*M, 128);
  float* B = (float*)_aligned_malloc(sizeof(float)*K*N, 128);
#else
  float* C_blas = (float*)malloc(sizeof(float)*N*M);
  float* C = (float*)malloc(sizeof(float)*N*M);
  float* A = (float*)malloc(sizeof(float)*K*M);
  float* B = (float*)malloc(sizeof(float)*K*N);
#endif

  srand((unsigned int) time(NULL));
  for(i=0; i < K*M; i++){
    A[i] = rand()*1.0/RAND_MAX;
  }
  for(i=0; i < K*N; i++){
    B[i] = rand()*1.0/RAND_MAX;
  }
  for(i=0; i < N*M; i++){
    C[i] = 0.0;
    C_blas[i] = 0.0;
  }
  double gflop = (double)2*K/1024*N/1024*M/1024;
  double alpha = 1.0;
  double beta = 0.0;

  gettimeofday(&t1, NULL);
  sw_cblas_sgemm_nopad(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C_blas, N);
  gettimeofday(&t2, NULL);
  double tt = TIME(t1,t2);
  double gflops = gflop / tt;
  printf("sgemm gflops is %.2lf time %lf sec\n", gflops, tt);


  gettimeofday(&t1, NULL);
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
  gettimeofday(&t2, NULL);
  tt = TIME(t1,t2);
  gflops = gflop / tt;
  printf("xMath gflops is %.2lf time %lf sec\n", gflops, tt);

#ifdef CHECK_RES
  int cnt = 0;
  double sum1 = 0., sum2 = 0.;
  cnt = 0;
  for(i = 0; i < N*M; ++i) {
    if(fabs(C[i] - C_blas[i]) > 1e-3 && cnt < 10) {
      printf("error @ %d, %lf vs %lf\n", i, C[i], C_blas[i]);
      cnt++;
    }
    sum1 += C[i];
    sum2 += C_blas[i];
  }
  printf("pass validation test! swblas sum1: %.2lf xmath sum2: %.2lf\n", sum1, sum2);
#endif

#ifdef _MEM_128BALIGN_
  _aligned_free(A);
  _aligned_free(B);
  _aligned_free(C);
  _aligned_free(C_blas);
#else
  free(A);
  free(B);
  free(C);
  free(C_blas);
#endif
  return;
}

int main(int argc, char **argv) {
  int M, K, N;
  M  = atoi(argv[1]);
  K = atoi(argv[2]);
  N = atoi(argv[3]);
  //test_sw_sgemm_small(M, N, K);
  //test_sw_sgemm_Anotrans_std(M, N, K);
  //test_sw_sgemm_Atrans_std(M, N, K);
  //test_sw_sgemm_Atrans_nopad_std(M, N, K);
  test_sw_dgemm_Atrans_std(M, N, K);
  return 0;
}

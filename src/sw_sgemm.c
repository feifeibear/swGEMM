#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include "swblas.h"
#include "./include/common_slave.h"
#include <cblas.h>

extern SLAVE_FUN(FJR_blas_sgemm)();
extern SLAVE_FUN(FJR_blas_sgemm_float)();
extern SLAVE_FUN(FJR_blas_sgemm_trans_rank)();
/******
 * N should not be blocked
**** */
void sw_sgemm(float* input, float* weight, float* output, int M, int N, int K, int blkM) {

  ConvData* params = (ConvData*)malloc(sizeof(ConvData));
  params->input = input;
  params->weight = weight;
  params->output = output;
  params->blkK = K;
  params->blkN = N;
  params->blkM = blkM;
  params->M = M;
  //params->numK = (K + blkK - 1)/blkK;

  //int ldm_use = sizeof(double)*(K*N*2 + K*blkM*2 + N*blkM)/64;
  //printf("K %d N %d M %d, blkM %d, LDM %d KM\n", K, N, M, blkM, ldm_use/1024);
  //if(ldm_use > 56*1024)
  //  return;

  if(params->blkN%128 == 0 && params->blkM%32 == 0 && params->blkK%8 == 0){
    athread_spawn(FJR_blas_sgemm, params);
    athread_join();
  } else {
    float alpha = 1.;
    float beta = 0.;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, input, K, weight, N, beta, output, N);
  }
  free(params);
  return;
}



#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include "swblas.h"
#include "./include/common_slave.h"
#include <cblas.h>

extern SLAVE_FUN(FJR_blas_sgemm_small)();
/******
 * Jiarui Fang
 * 2018.Dec.9th
 * input (M, K)
 * weight (N, K)
 * output (M, N)
 * block on M dimension
**** */
void sw_sgemp(float* input, float* weight, float* output, int M, int N, int K, int blkM) {

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
  athread_spawn(FJR_blas_sgemm_small, params);
  athread_join();

  free(params);
  return;
}

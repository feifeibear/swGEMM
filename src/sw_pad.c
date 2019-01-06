#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include "./include/common_slave.h"
#include <cblas.h>
extern void SLAVE_FUN(FJR_zeropad_matrix());
extern void SLAVE_FUN(FJR_depad_matrix());
/*******
 * deprecated!
 * Old interface for Padding
 * Copy entire data, so time consumming
 * *****/

void sw_zeropad_matrix(const float* A, int ld, int ld_pad, int hd, int hd_pad, float* A_zeropad) {
  /*
  int i, j;
  float* A_zeropad2 = (float*)malloc(sizeof(float)*ld_pad*hd_pad);
  memset(A_zeropad2, 0, ld_pad*hd_pad*sizeof(float));
  for(i = 0; i < hd; ++i)
    for(j = 0; j < ld; ++j) {
      A_zeropad2[i*ld_pad + j] = A[i*ld + j];
    }
    */

  ZeropadStruct* params = (ZeropadStruct*)malloc(sizeof(ZeropadStruct));
  params->A = A;
  params->ld = ld;
  params->ld_pad = ld_pad;
  params->hd = hd;
  params->hd_pad = hd_pad;
  params->A_zeropad = A_zeropad;
  athread_spawn(FJR_zeropad_matrix, params);
  athread_join();
  free(params);
/*
  int cnt = 0;
  for(i = 0; i < ld_pad*hd_pad; ++i)
    if(A_zeropad2[i] != A_zeropad[i])
      printf("%d %d %lf vs %lf\n", i/ld_pad, i%ld_pad, A_zeropad2[i], A_zeropad[i]);

  free(A_zeropad2);
      */
  return;
}


void sw_depad_matrix(float* A, int ld, int ld_pad, int hd, int hd_pad, const float* A_zeropad) {
  ZeropadStruct* params = (ZeropadStruct*)malloc(sizeof(ZeropadStruct));
  params->A = A;
  params->ld = ld;
  params->ld_pad = ld_pad;
  params->hd = hd;
  params->hd_pad = hd_pad;
  params->A_zeropad = A_zeropad;

  athread_spawn(FJR_depad_matrix, params);
  athread_join();
  free(params);
  return;
}


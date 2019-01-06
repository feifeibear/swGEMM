#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "include/common_slave.h"


/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2016.Sep.28th
 * On good if Water
 *
 * input  is of dim(K, M)
 * weight is of dim(K, N)
 * ouput  is of dim(N, M)
 *
 * fully overlap input DMA and weight DMA
 * ************/
#define SIMDSIZE 4
void FJR_blas_dgemm_trans(ConvData* param)
{
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int K = param->Ni;
  int N = param->No;
  int M = param->B;
  int T = param->T;

//M, K, Ci, Ri
  double* local_input  = (double*) ldm_malloc(sizeof(double)*K*M/8/8*2);
  int local_input_size = K*M/8/8;
//N, K, K, K
  double* local_weight = (double*) ldm_malloc(sizeof(double)*K*N/8/8);
  int local_weight_size = K*N/64;
//M, N, Co, Ro
  double* local_output = (double*) ldm_malloc(sizeof(double)*N*M/8/8*2);
  int local_output_size = N*M/8/8;

  volatile int  input_replyget = 0, weight_replyget = 0,  replyput = 0;
  dma_desc dma_get_input, dma_get_weight, dma_get_output, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_weight, DMA_GET);
  dma_set_mode(&dma_get_weight, PE_MODE);
  dma_set_reply(&dma_get_weight, &weight_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_iutput(M/8, K/8)
  dma_set_size(&dma_get_input, M*K/8/8*sizeof(double));
  dma_set_bsize(&dma_get_input, M/8*sizeof(double));
  dma_set_stepsize(&dma_get_input, (M*T-M/8)*sizeof(double));

  //DMA for local_weight(K/8, N/8)
  dma_set_size(&dma_get_weight, N*K/8/8*sizeof(double));
  dma_set_bsize(&dma_get_weight, N/8*sizeof(double));
  dma_set_stepsize(&dma_get_weight, N/8*7*sizeof(double));

  //DMA for local_output(M/8, N/8)
  dma_set_size(&dma_put_output, M*N/8/8*sizeof(double));
  dma_set_bsize(&dma_put_output, M/8*sizeof(double));
  dma_set_stepsize(&dma_put_output, (M*T-M/8)*sizeof(double));

  //fetch weight into LDM and use it all the time
  dma(dma_get_weight, (long)((double*)param->weight+(cid*N/8 + rid*K/8*N)), (long)(local_weight));
  dma_wait(&weight_replyget, 1); weight_replyget = 0;

  int cur_input_idx = 0;
  int cur_output_idx = 0;
  int cur_input_offset = local_input_size;
  int cur_output_offset = local_output_size;

  double* input_start = (double*)param->input + cid*K*M*T/8 + rid*M/8;
  double* output_start = (double*)param->output + cid*N*M*T/8 + rid*M/8;

  dma(dma_get_input, (long)(input_start), (long)(local_input + cur_input_idx*cur_input_offset));
  dma_wait(&input_replyget, 1); input_replyget = 0;
  cur_input_idx = 1 - cur_input_idx;

  int cT,i;
  for(cT = 0; cT < T; ++cT) {
    if(cT+1 < T)
      dma(dma_get_input, (long)((double*)input_start + (cT+1)*M), (long)(local_input + cur_input_idx*cur_input_offset));
    for(i = 0; i < local_output_size; ++i)
      (local_output + cur_output_idx*cur_output_offset)[i] = 0;

    //gemmdouble(
    //ldm_dgemm_trans(
    dgemmtransasm(
    //ldm_dgemm_trans(
        (double*)(local_input + (1-cur_input_idx)*cur_input_offset),
				(double*)(local_weight),
				(double*)(local_output + cur_output_idx*cur_output_offset),
				M/8/4,
				M/8/4,
				N/8,
				K/8,
				rid,
				cid);

    if(cT+1 < T) {
      dma_wait(&input_replyget, 1); input_replyget = 0;
      cur_input_idx = 1-cur_input_idx;
    }
    if(cT > 0)
      dma_wait(&replyput, 1); replyput = 0;
    dma(dma_put_output, (long)((double*)output_start + cT*M), 
      (long)(local_output + cur_output_idx*cur_output_offset));
    cur_output_idx = 1 - cur_output_idx;
  }
  dma_wait(&replyput, 1); replyput = 0;
  ldm_free(local_input, sizeof(double)*local_input_size*2);
  ldm_free(local_weight, sizeof(double)*local_weight_size);
  ldm_free(local_output, sizeof(double)*local_output_size*2);

}//main func


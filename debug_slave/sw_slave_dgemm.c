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
 * input  is of dim(B, Ni)
 * weight is of dim(Ni, No)
 * ouput  is of dim(B, No)
 *
 * fully overlap input DMA and weight DMA
 *
 * Oct 16th
 * change weight to 1 pixel read leading to batch-size-version
 *
 * ************/
#define SIMDSIZE 4
void FJR_blas_dgemm(ConvData* param)
{
  int i; 
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int Ni = param->Ni;
  int No = param->No;
  int B = param->B;
  int T = param->T;

//B, Ni, Ci, Ri
  doublev4* local_input  = (doublev4*) ldm_malloc(sizeof(double)*Ni*B/8/8*2);
  int local_input_size = Ni*B/8/8;
//No, Ni, K, K
  double* local_weight = (double*) ldm_malloc(sizeof(double)*Ni*No/8/8);
  int local_weight_size = Ni*No/64;
//B, No, Co, Ro
  doublev4* local_output = (doublev4*) ldm_malloc(sizeof(double)*No*B/8/8*2);
  int local_output_size = No*B/8/8;

//  double local_weight[K*K*Ni/64*No];
//initilize DMA variables
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

  //DMA for local_iutput(B/8, Ni/8)
  dma_set_size(&dma_get_input, B*Ni/8/8/SIMDSIZE*sizeof(doublev4));
  dma_set_bsize(&dma_get_input, Ni/SIMDSIZE/8*sizeof(doublev4));
  dma_set_stepsize(&dma_get_input, Ni/SIMDSIZE/8*7*sizeof(doublev4));

  //DMA for local_weight(Ni/8, No/8)
  dma_set_size(&dma_get_weight, No*Ni/8/8*sizeof(double));
  dma_set_bsize(&dma_get_weight, No/8*sizeof(double));
  dma_set_stepsize(&dma_get_weight, No/8*7*sizeof(double));

  //DMA for local_output(B/8, No/8)
  dma_set_size(&dma_put_output, B*No/8/8/SIMDSIZE*sizeof(doublev4));
  dma_set_bsize(&dma_put_output, No/SIMDSIZE/8*sizeof(doublev4));
  dma_set_stepsize(&dma_put_output, No/SIMDSIZE/8*7*sizeof(doublev4));

  //fetch weight into LDM and use it all the time
  dma(dma_get_weight, (long)((double*)param->weight+(cid*No/8*Ni + rid*No/8)), (long)(local_weight));
  dma_wait(&weight_replyget, 1); weight_replyget = 0;

  int cur_input_idx = 0;
  int cur_output_idx = 0;
  int cur_input_offset = local_input_size/4;
  int cur_output_offset = local_output_size/4;

  double* input_start = (double*)param->input + cid*Ni*B/8 + rid*Ni/8;
  double* output_start = (double*)param->output + cid*No*B/8 + rid*No/8;

  dma(dma_get_input, (long)(input_start), (long)(local_input + cur_input_idx*cur_input_offset));
  dma_wait(&input_replyget, 1); input_replyget = 0;
  cur_input_idx = 1 - cur_input_idx;

  int cT;
  for(cT = 0; cT < T; ++cT) {
    if(cT+1 < T)
      dma(dma_get_input, (long)((double*)input_start + (cT+1)*Ni*B), (long)(local_input + cur_input_idx*cur_input_offset));
    for(i = 0; i < local_output_size/4; ++i)
      (local_output + (cur_output_idx)*cur_output_offset)[i] = 0;

    dgemmasm(
        (double*)(local_weight),
				(double*)(local_input + (1 - cur_input_idx)*cur_input_offset),
				(double*)(local_output + cur_output_idx*cur_output_offset),
				No/8/4,
				No/8/4,
				B/8,
				Ni/8,
				rid,
				cid);

    if(cT+1 < T) {
      dma_wait(&input_replyget, 1); input_replyget = 0;
      cur_input_idx = 1-cur_input_idx;
    }
    if(cT > 0) {
      dma_wait(&replyput, 1); replyput = 0;
    }
    dma(dma_put_output, (long)((double*)output_start + cT*No*B), (long)(local_output + cur_output_idx*cur_output_offset));
    cur_output_idx = 1 - cur_output_idx;
  }
  dma_wait(&replyput, 1); replyput = 0;
  ldm_free(local_input, sizeof(double)*local_input_size*2);
  ldm_free(local_weight, sizeof(double)*local_weight_size);
  ldm_free(local_output, sizeof(double)*local_output_size*2);

}//main func


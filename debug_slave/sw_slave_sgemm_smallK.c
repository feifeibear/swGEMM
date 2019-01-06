#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "include/common_slave.h"
#include <dma.h>

/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2016.Sep.25th
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
void FJR_blas_sgemm_smallB(ConvData* param)
{
  return;
  int i;
  int id = athread_get_id(-1);
  int Ni = param->Ni;
  int No = param->No;
  int B = param->B;
  int T = param->T;

  int local_input_size = Ni*B;
  int local_weight_size = Ni*No;
  int local_output_size = No*B;

  int ldm_use = local_input_size*2 + local_weight_size + local_output_size;
  ldm_use = ldm_use * sizeof(float);
  if(ldm_use > 56*1024) {
    if(id == 0)
      printf("FJR_blas_sgemm_smallB LDM overused %d KB\n", ldm_use/1024);
    return;
  }

//B, Ni, Ci, Ri
  floatv4* local_input  = (floatv4*) ldm_malloc(sizeof(float)*Ni*B*2);
//No, Ni, K, K
  float* local_weight = (float*) ldm_malloc(sizeof(float)*Ni*No);
//B, No, Co, Ro
  floatv4* local_output = (floatv4*) ldm_malloc(sizeof(float)*No*B);



//  float local_weight[K*K*Ni/64*No];
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

  //DMA for local_input(B/8, Ni/8)
  dma_set_size(&dma_get_input, Ni*B*sizeof(float));
//  dma_set_bsize(&dma_get_input, Ni*B*sizeof(float));
//  dma_set_stepsize(&dma_get_input, Ni*sizeof(floatv4));

  //DMA for local_weight(Ni/8, No/8)
  dma_set_size(&dma_get_weight, No*Ni*sizeof(float));
//  dma_set_bsize(&dma_get_weight, No*sizeof(float));
//  dma_set_stepsize(&dma_get_weight, No*sizeof(float));

  //DMA for local_output(B/8, No/8)
  dma_set_size(&dma_put_output, B*No*sizeof(float));
//  dma_set_bsize(&dma_put_output, No/SIMDSIZE/8*sizeof(floatv4));
//  dma_set_stepsize(&dma_put_output, No/SIMDSIZE/8*7*sizeof(floatv4));

  //fetch weight into LDM and use it all the time
  dma(dma_get_weight, (long)((float*)param->weight), (long)(local_weight));
  dma_wait(&weight_replyget, 1); weight_replyget = 0;

  int cur_input_idx = 0;
  int cur_offset = local_input_size/SIMDSIZE;

  float* input_start = (float*)param->input + id*B*Ni;
  float* output_start = (float*)param->output + id*B*No;

  dma(dma_get_input, (long)(input_start), (long)(local_input + cur_input_idx*cur_offset));
  dma_wait(&input_replyget, 1); input_replyget = 0;
  cur_input_idx = cur_input_idx ? 0:1;

  int cT;
  for(cT = 0; cT < T; ++cT) {
    if(cT+1 < T)
      dma(dma_get_input, (long)(input_start + (cT+1)*Ni*64*B), (long)(local_input + cur_input_idx*cur_offset));
    for(i = 0; i < local_output_size/4; ++i)
      local_output[i] = 0.0;

    gemm_small((float*)(local_weight),
				(float*)(local_input + (1-cur_input_idx)*cur_offset),
				(float*)(local_output),
				No/4,
				No/4,
				B,
				Ni);

    if(cT+1 < T) {
      dma_wait(&input_replyget, 1); input_replyget = 0;
      cur_input_idx = 1-cur_input_idx;
    }

    dma(dma_put_output, (long)(output_start + cT*64*No*B), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;

  }
  ldm_free(local_input, sizeof(float)*local_input_size*2);
  ldm_free(local_weight, sizeof(float)*local_weight_size);
  ldm_free(local_output, sizeof(float)*local_output_size);

}//main func


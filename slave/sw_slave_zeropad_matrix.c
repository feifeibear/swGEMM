#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "include/common_slave.h"

inline void mb()
{
    asm volatile("memb");
    asm volatile("":::"memory");
}

__thread_local dma_desc dma_get, dma_put;

#define MAX_LDM_SIZE (48*1024)
void FJR_zeropad_matrix(ZeropadStruct* param)
{
  int i, j;
  int id = athread_get_id(-1);
  float* A = param->A;
  float* A_zeropad = param->A_zeropad;
  int ld = param->ld;
  int ld_pad = param->ld_pad;
  int hd = param->hd;
  int hd_pad = param->hd_pad;
  float* local_buff = (float*)ldm_malloc(MAX_LDM_SIZE);
  volatile int  replyget = 0, replyput = 0;
//  dma_desc dma_get, dma_put;

  dma_set_op(&dma_get, DMA_GET);
  dma_set_mode(&dma_get, PE_MODE);
  dma_set_reply(&dma_get, &replyget);

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);

  if(hd_pad < 64) {
    if(id == 0) {
      printf("not implemented for sw_zeropad_matrix!");
    }
  } else {
    if(ld_pad*sizeof(float) < MAX_LDM_SIZE) {
      for(i = ld; i < ld_pad; ++i)
        local_buff[i] = 0.;
      dma_set_size(&dma_get, ld*sizeof(float));
      dma_set_size(&dma_put, ld_pad*sizeof(float));
      for(i = id; i < hd_pad; i += 64) {
        if(i < hd) {
          dma(dma_get, (long)(A + i*ld), (long)(local_buff));
          dma_wait(&replyget, 1); replyget = 0;
        } else {
          for(j = 0; j < ld; ++j)
            local_buff[j] = 0.;
        }
        dma(dma_put, (long)(A_zeropad + i*ld_pad), (long)(local_buff));
        dma_wait(&replyput, 1); replyput = 0;
      }
    } else {
      if(id == 0) {
        printf("not implemented for sw_zeropad_matrix!");
      }
    }
  }
  ldm_free(local_buff, MAX_LDM_SIZE);
}//main func


void FJR_depad_matrix(ZeropadStruct* param)
{
  int i, j;
  int id = athread_get_id(-1);
  float* A = param->A;
  float* A_zeropad = param->A_zeropad;
  int ld = param->ld;
  int ld_pad = param->ld_pad;
  int hd = param->hd;
  int hd_pad = param->hd_pad;

  float* local_buff = (float*)ldm_malloc(MAX_LDM_SIZE);
  volatile int  replyget = 0, replyput = 0;
  //dma_desc dma_get, dma_put;

  dma_set_op(&dma_get, DMA_GET);
  dma_set_mode(&dma_get, PE_MODE);
  dma_set_reply(&dma_get, &replyget);

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);

  dma_set_size(&dma_get, ld*sizeof(float));
  dma_set_size(&dma_put, ld*sizeof(float));

  if(hd_pad < 64) {
    if(id == 0) {
      printf("not implemented for sw_zeropad_matrix!");
    }
  } else {
    if(ld*sizeof(float) < MAX_LDM_SIZE) {
      for(i = id; i < hd; i += 64) {
      //  athread_get(PE_MODE, A_zeropad + i*ld_pad, local_buff, ld*sizeof(float),&replyget,0,0,0);
      //  while(replyget!=1);
      //  replyget=0;

        dma(dma_get, (long)(A_zeropad + i*ld_pad), (long)(local_buff));
        dma_wait(&replyget, 1); replyget = 0;
        mb();

        //athread_put(PE_MODE, A + i*ld, local_buff, ld*sizeof(float),&replyput,0,0);
        //while(replyput!=1);
        //replyput=0;
        dma(dma_put, (long)(A + i*ld), (long)(local_buff));
        dma_wait(&replyput, 1); replyput = 0;
      }
    } else {
      if(id == 0) {
        printf("not implemented for sw_zeropad_matrix!");
      }
    }
  }
  ldm_free(local_buff, MAX_LDM_SIZE);
}

#undef MAX_LDM_SIZE

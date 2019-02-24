
/*************************************************************************
	> File Name: copy_border.c
	> Author: 
	> Mail: 
	> Created Time: Wed Jan  2 20:07:22 2019
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <slave.h>
#include <dma.h>
#include <math.h>
#include "include/common_slave.h"

__thread_local dma_desc dma_src_back, dma_dst_back;

void copy_border_back_float32(CopyData* params)
{
    float* src = (float*)(params->src);
    float* dst = (float*)(params->dst);
    const int M = params->M;
    const int N = params->N;
    const int Ms = params->Ms;
    const int Ns = params->Ns;
    const int Me = params->Me;
    const int Ne = params->Ne;
    const int trans = params->trans;
    const int id = athread_get_id(-1);
    const int cid = id%8, rid = id/8;
    /*if(id == 0)
        printf("M %d N %d Ms %d Ns %d Me %d Ne %d\n", M,N,Ms,Ns,Me,Ne);
    return ;*/
    if(Ms%32 != 0 || Ns%32 != 0)
    {
        printf("copy border error: Ms %d, Ns %d\n", Ms, Ns);
        exit(0);
    }
    volatile int reply_src = 0, reply_dst = 0;
    //dma_desc dma_src, dma_dst;
    dma_set_op(&dma_dst_back, DMA_PUT);
    dma_set_mode(&dma_dst_back, PE_MODE);
    dma_set_reply(&dma_dst_back, &reply_dst);
    dma_set_op(&dma_src_back, DMA_GET);
    dma_set_mode(&dma_src_back, PE_MODE);
    dma_set_reply(&dma_src_back, &reply_src);

    if(trans == 0) // column major
    {
        if(id < 32) // copy the right panel
        {
            if(Ns >= Ne)
                return;
            int i = 0, j = 0, k = 0;
            float* buf = (float*)ldm_malloc(sizeof(float) * M);
            float* start;
            dma_set_size(&dma_src_back, sizeof(float)*M);
            dma_set_size(&dma_dst_back, sizeof(float)*M);

            for(j = id; j < N-Ns; j += 32)
            {
                if(Ns+j < N)
                {
                    //copy Ns+j column
                    //printf("Ns+j %d read\n", Ns+j);
                    start = &src[(Ns+j)*Me];
                    dma(dma_src_back, (long)start, (long)buf);
                    dma_wait(&reply_src, 1);
                    reply_src = 0;
                    //store Ns+j column
                    //if(id == 0)
                    //printf("id %d begin write\n", id);
                    //printf("Ns+j %d finish\n", Ns+j);
                    start = &dst[(Ns+j)*M];
                    dma(dma_dst_back, (long)start, (long)buf);
                    dma_wait(&reply_dst, 1);
                    reply_dst = 0;
                    //if(id == 0)
                        //printf("end write\n");
                }
            }

            //printf("id %d free\n", id);
            ldm_free(buf, sizeof(float)*M);
            //printf("id %d finish\n", id);
        }
        else //copy the bottom panel
        {
            if(Ms >= Me)
                return;
            int cols = Ns/32;
            float* src_origin = &src[(id-32)*cols*Me];
            float* dst_origin = &dst[(id-32)*cols*M];
            int i, j, k;
            int max_cols = (56*1024) / ((M-Ms)*sizeof(float));
            int numb = (cols + max_cols - 1) / max_cols;
            int BS = max_cols;
            int remBS = cols - (numb-1)*BS;
            float *buf, *start;
            if(numb > 1)
                buf = (float*)ldm_malloc(sizeof(float) * BS * (M-Ms));
            else
                buf = (float*)ldm_malloc(sizeof(float) * remBS * (M-Ms));

            for(i = 0;i < numb; i ++)
            {
                int currBS = (i < numb-1) ? BS : remBS;
                //load
                dma_set_size(&dma_src_back, currBS * (M-Ms) * sizeof(float));
                dma_set_bsize(&dma_src_back, (M-Ms) * sizeof(float));
                dma_set_stepsize(&dma_src_back, (Me-M+Ms) * sizeof(float));
                start = &src_origin[ i*BS*Me + Ms ];
                dma(dma_src_back, (long)start, (long)buf);
                dma_wait(&reply_src, 1);
                reply_src = 0;

                //store
                dma_set_size(&dma_dst_back, currBS * (M-Ms) * sizeof(float));
                dma_set_bsize(&dma_dst_back, (M-Ms) * sizeof(float));
                dma_set_stepsize(&dma_dst_back, Ms * sizeof(float));
                start = &dst_origin[ i*BS*M + Ms ];
                dma(dma_dst_back, (long)start, (long)buf);
                dma_wait(&reply_dst, 1);
                reply_dst = 0;
            }

            if(numb > 1)
                ldm_free(buf, sizeof(float) * BS * (M-Ms));
            else
                ldm_free(buf, sizeof(float) * remBS * (M-Ms));
        }
    }
    else
    {
        printf("ERROR : have not support the trans copy back\n");
        exit(0);
    }
}



void copy_border_back_double64(CopyData* params)
{
    double* src = (double*)(params->src);
    double* dst = (double*)(params->dst);
    const int M = params->M;
    const int N = params->N;
    const int Ms = params->Ms;
    const int Ns = params->Ns;
    const int Me = params->Me;
    const int Ne = params->Ne;
    const int trans = params->trans;
    const int id = athread_get_id(-1);
    const int cid = id%8, rid = id/8;
    /*if(id == 0)
        printf("M %d N %d Ms %d Ns %d Me %d Ne %d\n", M,N,Ms,Ns,Me,Ne);
    return ;*/
    if(Ms%32 != 0 || Ns%32 != 0)
    {
        printf("copy border error: Ms %d, Ns %d\n", Ms, Ns);
        exit(0);
    }
    volatile int reply_src = 0, reply_dst = 0;
    //dma_desc dma_src, dma_dst;
    dma_set_op(&dma_dst_back, DMA_PUT);
    dma_set_mode(&dma_dst_back, PE_MODE);
    dma_set_reply(&dma_dst_back, &reply_dst);
    dma_set_op(&dma_src_back, DMA_GET);
    dma_set_mode(&dma_src_back, PE_MODE);
    dma_set_reply(&dma_src_back, &reply_src);

    if(trans == 0) // column major
    {
        if(id < 32) // copy the right panel
        {
            if(Ns >= Ne)
                return;
            int i = 0, j = 0, k = 0;
            double* buf = (double*)ldm_malloc(sizeof(double) * M);
            double* start;
            dma_set_size(&dma_src_back, sizeof(double)*M);
            dma_set_size(&dma_dst_back, sizeof(double)*M);

            for(j = id; j < N-Ns; j += 32)
            {
                if(Ns+j < N)
                {
                    //copy Ns+j column
                    //printf("Ns+j %d read\n", Ns+j);
                    start = &src[(Ns+j)*Me];
                    dma(dma_src_back, (long)start, (long)buf);
                    dma_wait(&reply_src, 1);
                    reply_src = 0;
                    //store Ns+j column
                    //if(id == 0)
                    //printf("id %d begin write\n", id);
                    //printf("Ns+j %d finish\n", Ns+j);
                    start = &dst[(Ns+j)*M];
                    dma(dma_dst_back, (long)start, (long)buf);
                    dma_wait(&reply_dst, 1);
                    reply_dst = 0;
                    //if(id == 0)
                        //printf("end write\n");
                }
            }

            //printf("id %d free\n", id);
            ldm_free(buf, sizeof(double)*M);
            //printf("id %d finish\n", id);
        }
        else //copy the bottom panel
        {
            if(Ms >= Me)
                return;
            int cols = Ns/32;
            double* src_origin = &src[(id-32)*cols*Me];
            double* dst_origin = &dst[(id-32)*cols*M];
            int i, j, k;
            int max_cols = (56*1024) / ((M-Ms)*sizeof(double));
            int numb = (cols + max_cols - 1) / max_cols;
            int BS = max_cols;
            int remBS = cols - (numb-1)*BS;
            double *buf, *start;
            if(numb > 1)
                buf = (double*)ldm_malloc(sizeof(double) * BS * (M-Ms));
            else
                buf = (double*)ldm_malloc(sizeof(double) * remBS * (M-Ms));

            for(i = 0;i < numb; i ++)
            {
                int currBS = (i < numb-1) ? BS : remBS;
                //load
                dma_set_size(&dma_src_back, currBS * (M-Ms) * sizeof(double));
                dma_set_bsize(&dma_src_back, (M-Ms) * sizeof(double));
                dma_set_stepsize(&dma_src_back, (Me-M+Ms) * sizeof(double));
                start = &src_origin[ i*BS*Me + Ms ];
                dma(dma_src_back, (long)start, (long)buf);
                dma_wait(&reply_src, 1);
                reply_src = 0;

                //store
                dma_set_size(&dma_dst_back, currBS * (M-Ms) * sizeof(double));
                dma_set_bsize(&dma_dst_back, (M-Ms) * sizeof(double));
                dma_set_stepsize(&dma_dst_back, Ms * sizeof(double));
                start = &dst_origin[ i*BS*M + Ms ];
                dma(dma_dst_back, (long)start, (long)buf);
                dma_wait(&reply_dst, 1);
                reply_dst = 0;
            }

            if(numb > 1)
                ldm_free(buf, sizeof(double) * BS * (M-Ms));
            else
                ldm_free(buf, sizeof(double) * remBS * (M-Ms));
        }
    }
    else
    {
        printf("ERROR : have not support the trans copy back\n");
        exit(0);
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <slave.h>
#include <dma.h>
#include <math.h>
#include "include/common_slave.h"
#include "swblas.h"

#include <assert.h>

void sgemm_dma_trans(ConvData* param)
{
    const int id = athread_get_id(-1);
    const int cid = id%8, rid = id/8;
    const int M = param->M;
    const int N = param->N;
    const int K = param->K;
    const int bM = param->blkM;
    const int bN = param->blkN;
    const int bK = param->blkK;
    const int Me = ((((M+bM)-1)/bM)*bM);
    const int Ne = ((((N+bN)-1)/bN)*bN);
    const int Ke = ((((K+bK)-1)/bK)*bK);
    const int numM = (((M+bM)-1)/bM);
    const int remM = (M-(bM*(numM-1)));
    const int numN = (((N+bN)-1)/bN);
    const int remN = (N-(bN*(numN-1)));
    const int numK = (((K+bK)-1)/bK);
    const int remK = (K-(bK*(numK-1)));
    int cM = 0;
    int cN = 0;
    int cK = 0;
    int i = 0;
    int realbM = bM;
    int realbN = bN;
    int realbK = bK;
    int nextbM = bM;
    int nextbN = bN;
    int nextbK = bK;
    int currM = 0;
    int currN = 0;
    int currK = 0;
    int double_buffer_flag = 0;

    double* local_A = (double*)((doublev4*)ldm_malloc(sizeof(double)*bM*bK/8/8*2));
    const int local_A_size = bM*bK/8/8;
    double* local_B = (double*)((doublev4*)ldm_malloc(sizeof(double)*bK*bN/8/8*2));
    const int local_B_size = bK*bN/8/8;
    double* local_C = (double*)((doublev4*)ldm_malloc(sizeof(double)*bM*bN/8/8));
    const int local_C_size = bM*bN/8/8;
    float* fptr, *fptr2;
    double* dptr, *dptr2;
    floatv4 vflt;
    doublev4 vdbl;

    volatile int replygetA = 0, replygetB=0, replyputC=0;
    dma_desc dmagetA, dmagetB, dmaputC;
    dma_set_op(&dmagetA, DMA_GET);
    dma_set_mode(&dmagetA, PE_MODE);
    dma_set_reply(&dmagetA, &replygetA);

    dma_set_op(&dmagetB, DMA_GET);
    dma_set_mode(&dmagetB, PE_MODE);
    dma_set_reply(&dmagetB, &replygetB);

    dma_set_op(&dmaputC, DMA_PUT);
    dma_set_mode(&dmaputC, PE_MODE);
    dma_set_reply(&dmaputC, &replyputC);

    float* startA = (float*)param->input + rid*bM/8 + cid*bK*M/8;
    float* startB = (float*)param->weight + rid*bK*N/8 + cid*bN/8;
    float* startC = (float*)param->output + rid*bM/8 + cid*bN*M/8;
    float* startAp = (float*)param->inputp + rid*bM/8 + cid*bK*Me/8;
    float* startBp = (float*)param->weightp + rid*bK*Ne/8 + cid*bN/8;
    float* startCp = (float*)param->outputp + rid*bM/8 + cid*bN*Me/8;
    float *nextA, *nextB, *realC;
    dma_set_size( &dmagetA, (((bM*bK)/64)*sizeof(float)) ); 
    dma_set_bsize( &dmagetA, ((bM/8)*sizeof(float)) ); 
    dma_set_stepsize( &dmagetA, ((M-(bM/8))*sizeof(float)) ); 
    dma_set_size( &dmagetB, (((bK*bN)/64)*sizeof(float)) ); 
    dma_set_bsize( &dmagetB, ((bN/8)*sizeof(float)) ); 
    dma_set_stepsize( &dmagetB, ((N-(bN/8))*sizeof(float)) ); 
    dma_set_size( &dmaputC, (((bM*bN)/64)*sizeof(float)) ); 
    dma_set_bsize( &dmaputC, ((bM/8)*sizeof(float)) ); 
    dma_set_stepsize( &dmaputC, ((M-(bM/8))*sizeof(float)) ); 
    dma(dmagetA, (long)(startA), (long)((local_A+((1-double_buffer_flag)*local_A_size)))); 
    dma(dmagetB, (long)(startB), (long)((local_B+((1-double_buffer_flag)*local_B_size)))); 
    dma_wait( &replygetA, 1 );
    replygetA = 0;
    dma_wait( &replygetB, 1 );
    replygetB = 0;
    for ( cN=0; cN<numN; cN+=1 ) // begin loop_CN
    {
        if ((cN<(numN-1)) )
        {
            realbN = bN;
        }
        else
        {
            realbN = remN;
        }
        for ( cM=0; cM<numM; cM+=1 ) // begin loop_cM
        {
            if ((cM<(numM-1)) )
            {
                realbM = bM;
            }
            else
            {
                realbM = remM;
            }
            for ( i=0; i<local_C_size; i+=1 ) // begin init_C
            {
                local_C[i] = 0;
            } // end init_C
            for ( cK=0; cK<numK; cK+=1 ) // begin loop_cK
            {
                if ((cK<(numK-1)) )
                {
                    realbK = bK;
                }
                else
                {
                    realbK = remK;
                }
                if (((((((cN*numK)*numM)+(cM*numK))+cK)+1)<((numN*numM)*numK)) )
                {
                    nextbK = realbK;
                    nextbM = realbM;
                    nextbN = realbN;
                    if ((cK==(numK-2)) )
                    {
                        nextbK = remK;
                    }
                    if ((cK==(numK-1)) )
                    {
                        nextbK = bK;
                        if ((cM==(numM-2)) )
                        {
                            nextbM = remM;
                        }
                        if ((cM==(numM-1)) )
                        {
                            nextbM = bM;
                            if ((cN==(numN-2)) )
                            {
                                nextbN = remN;
                            }
                        }
                    }
                    if ((nextbK != bK||nextbM != bM) )
                    {
                        nextA = startAp;
                        currM = Me;
                        dma_set_stepsize( 
                            &dmagetA,
                            ((Me-(bM/8))*sizeof(float)) 
                         );
                    }
                    else
                    {
                        nextA = startA;
                        currM = M;
                        dma_set_stepsize( 
                            &dmagetA,
                            ((M-(bM/8))*sizeof(float)) 
                         );
                    }
                    if ((nextbK != bK||nextbN != bN) )
                    {
                        nextB = startBp;
                        currN = Ne;
                        dma_set_stepsize( 
                            &dmagetB,
                            ((Ne-(bN/8))*sizeof(float)) 
                         );
                    }
                    else
                    {
                        nextB = startB;
                        currN = N;
                        dma_set_stepsize( 
                            &dmagetB,
                            ((N-(bN/8))*sizeof(float)) 
                         );
                    }
                    if ((cK==(numK-1)) )
                    {
                        if ((cM==(numM-1)) )
                        {
                            dma(dmagetA, (long)(nextA), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                            dma(dmagetB, (long)((nextB+(((0*bK)*currN)+((cN+1)*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                        }
                        else
                        {
                            dma(dmagetA, (long)((nextA+(((0*bK)*currM)+((cM+1)*bM)))), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                            dma(dmagetB, (long)((nextB+(((0*bK)*currN)+(cN*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                        }
                    }
                    else
                    {
                        dma(dmagetA, (long)((nextA+((((cK+1)*bK)*currM)+(cM*bM)))), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                        dma(dmagetB, (long)((nextB+((((cK+1)*bK)*currN)+(cN*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                    }
                }
                fptr = (float*)((local_A+((1-double_buffer_flag)*local_A_size)));
                dptr = (double*)((local_A+((1-double_buffer_flag)*local_A_size)));
                for ( i=(local_A_size-4); i>=0; i-=4 ) // begin f2d_A
                {
                    simd_load( 
                        vflt,
                        &fptr[i] 
                     );
                    vdbl = (doublev4)(vflt);
                    simd_store( 
                        vdbl,
                        &dptr[i] 
                     );
                } // end f2d_A
                fptr = (float*)((local_B+((1-double_buffer_flag)*local_B_size)));
                dptr = (double*)((local_B+((1-double_buffer_flag)*local_B_size)));
                for ( i=(local_B_size-4); i>=0; i-=4 ) // begin f2d_B
                {
                    simd_load( 
                        vflt,
                        &fptr[i] 
                     );
                    vdbl = (doublev4)(vflt);
                    simd_store( 
                        vdbl,
                        &dptr[i] 
                     );
                } // end f2d_B
                dgemmtransasm( 
                    (double*)((local_A+((1-double_buffer_flag)*local_A_size))),
                    (double*)((local_B+((1-double_buffer_flag)*local_B_size))),
                    (double*)(local_C),
                    ((bM/8)/4),
                    ((bM/8)/4),
                    (bN/8),
                    (bK/8),
                    rid,
                    cid 
                 );
                if (((((((cN*numK)*numM)+(cM*numK))+cK)+1)<((numN*numM)*numK)) )
                {
                    dma_wait( &replygetA, 1 );
                    replygetA = 0;
                    dma_wait( &replygetB, 1 );
                    replygetB = 0;
                    double_buffer_flag = (1-double_buffer_flag);
                }
            } // end loop_cK
            fptr = (float*)(local_C);
            dptr = (double*)(local_C);
            for ( i=0; i<local_C_size; i+=4 ) // begin d2f_C
            {
                simd_load( 
                    vdbl,
                    &dptr[i] 
                 );
                vflt = (floatv4)(vdbl);
                simd_store( 
                    vflt,
                    &fptr[i] 
                 );
            } // end d2f_C
            if ((realbM != bM||realbN != bN) )
            {
                realC = startCp;
                currM = Me;
                dma_set_stepsize( 
                    &dmaputC,
                    ((Me-(bM/8))*sizeof(float)) 
                 );
            }
            else
            {
                realC = startC;
                currM = M;
                dma_set_stepsize( 
                    &dmaputC,
                    ((M-(bM/8))*sizeof(float)) 
                 );
            }
            dma(dmaputC, (long)((realC+(((cN*bN)*currM)+(cM*bM)))), (long)(local_C)); 
            dma_wait( &replyputC, 1 );
            replyputC = 0;
        } // end loop_cM
    } // end loop_CN

    ldm_free(local_A, sizeof(double)*local_A_size*2);
    ldm_free(local_B, sizeof(double)*local_B_size*2);
    ldm_free(local_C, sizeof(double)*local_C_size);

}




void dgemm_dma_trans(ConvData* param)
{
    const int id = athread_get_id(-1);
    const int cid = id%8, rid = id/8;
    const int M = param->M;
    const int N = param->N;
    const int K = param->K;
    const int bM = param->blkM;
    const int bN = param->blkN;
    const int bK = param->blkK;
    const int Me = ((((M+bM)-1)/bM)*bM);
    const int Ne = ((((N+bN)-1)/bN)*bN);
    const int Ke = ((((K+bK)-1)/bK)*bK);
    const int numM = (((M+bM)-1)/bM);
    const int remM = (M-(bM*(numM-1)));
    const int numN = (((N+bN)-1)/bN);
    const int remN = (N-(bN*(numN-1)));
    const int numK = (((K+bK)-1)/bK);
    const int remK = (K-(bK*(numK-1)));
    const int ldi = param->ldi;
    const int ldw = param->ldw;
    const int ldo = param->ldo;
    int cM = 0;
    int cN = 0;
    int cK = 0;
    int i = 0;
    int realbM = bM;
    int realbN = bN;
    int realbK = bK;
    int nextbM = bM;
    int nextbN = bN;
    int nextbK = bK;
    int currM = 0;
    int currN = 0;
    int currK = 0;
    int double_buffer_flag = 0;

    double* local_A = (double*)((doublev4*)ldm_malloc(sizeof(double)*bM*bK/8/8*2));
    const int local_A_size = bM*bK/8/8;
    double* local_B = (double*)((doublev4*)ldm_malloc(sizeof(double)*bK*bN/8/8*2));
    const int local_B_size = bK*bN/8/8;
    double* local_C = (double*)((doublev4*)ldm_malloc(sizeof(double)*bM*bN/8/8));
    const int local_C_size = bM*bN/8/8;
    float* fptr, *fptr2;
    double* dptr, *dptr2;
    floatv4 vflt;
    doublev4 vdbl;

    volatile int replygetA = 0, replygetB=0, replyputC=0;
    dma_desc dmagetA, dmagetB, dmaputC;
    dma_set_op(&dmagetA, DMA_GET);
    dma_set_mode(&dmagetA, PE_MODE);
    dma_set_reply(&dmagetA, &replygetA);

    dma_set_op(&dmagetB, DMA_GET);
    dma_set_mode(&dmagetB, PE_MODE);
    dma_set_reply(&dmagetB, &replygetB);

    dma_set_op(&dmaputC, DMA_PUT);
    dma_set_mode(&dmaputC, PE_MODE);
    dma_set_reply(&dmaputC, &replyputC);

    double* startA = (double*)param->input + rid*bM/8 + cid*bK*ldi/8;
    double* startB = (double*)param->weight + rid*bK*ldw/8 + cid*bN/8;
    double* startC = (double*)param->output + rid*bM/8 + cid*bN*ldo/8;
    double* startAp = (double*)param->inputp + rid*bM/8 + cid*bK*Me/8;
    double* startBp = (double*)param->weightp + rid*bK*Ne/8 + cid*bN/8;
    double* startCp = (double*)param->outputp + rid*bM/8 + cid*bN*Me/8;
    double *nextA, *nextB, *realC;
    dma_set_size( &dmagetA, (((bM*bK)/64)*sizeof(double)) ); 
    dma_set_bsize( &dmagetA, ((bM/8)*sizeof(double)) ); 
    dma_set_stepsize( &dmagetA, ((ldi-(bM/8))*sizeof(double)) ); 
    dma_set_size( &dmagetB, (((bK*bN)/64)*sizeof(double)) ); 
    dma_set_bsize( &dmagetB, ((bN/8)*sizeof(double)) ); 
    dma_set_stepsize( &dmagetB, ((ldw-(bN/8))*sizeof(double)) ); 
    dma_set_size( &dmaputC, (((bM*bN)/64)*sizeof(double)) ); 
    dma_set_bsize( &dmaputC, ((bM/8)*sizeof(double)) ); 
    dma_set_stepsize( &dmaputC, ((ldo-(bM/8))*sizeof(double)) ); 
    if(M < bM || K < bK){
        dma_set_stepsize( &dmagetA, ((Me-(bM/8))*sizeof(double)) ); 
        dma(dmagetA, (long)(startAp), (long)((local_A+((1-double_buffer_flag)*local_A_size)))); 
    }
    else{
        dma(dmagetA, (long)(startA), (long)((local_A+((1-double_buffer_flag)*local_A_size)))); 
    }
    if(K < bK || N < bN){
        dma_set_stepsize( &dmagetB, ((Ne-(bN/8))*sizeof(double)) ); 
        dma(dmagetB, (long)(startBp), (long)((local_B+((1-double_buffer_flag)*local_B_size)))); 
    }
    else{
        dma(dmagetB, (long)(startB), (long)((local_B+((1-double_buffer_flag)*local_B_size)))); 
    }
    dma_wait( &replygetA, 1 );
    replygetA = 0;
    dma_wait( &replygetB, 1 );
    replygetB = 0;
    for ( cN=0; cN<numN; cN+=1 ) // begin loop_CN
    {
        if ((cN<(numN-1)) )
        {
            realbN = bN;
        }
        else
        {
            realbN = remN;
        }
        for ( cM=0; cM<numM; cM+=1 ) // begin loop_cM
        {
            if ((cM<(numM-1)) )
            {
                realbM = bM;
            }
            else
            {
                realbM = remM;
            }
            for ( i=0; i<local_C_size; i+=1 ) // begin init_C
            {
                local_C[i] = 0;
            } // end init_C
            for ( cK=0; cK<numK; cK+=1 ) // begin loop_cK
            {
                if ((cK<(numK-1)) )
                {
                    realbK = bK;
                }
                else
                {
                    realbK = remK;
                }
                if (((((((cN*numK)*numM)+(cM*numK))+cK)+1)<((numN*numM)*numK)) )
                {
                    nextbK = realbK;
                    nextbM = realbM;
                    nextbN = realbN;
                    if ((cK==(numK-2)) )
                    {
                        nextbK = remK;
                    }
                    if ((cK==(numK-1)) )
                    {
                        //nextbK = bK;
                        //fix a BUG :: when K < bK
                        if (numK != 1) nextbK = bK;
                        else nextbK = remK;
                        if ((cM==(numM-2)) )
                        {
                            nextbM = remM;
                        }
                        if ((cM==(numM-1)) )
                        {
                            //nextbM = bM;
                            //fix a BUG :: when M < bM
                            if (numM != 1) nextbM = bM;
                            else nextbM = remM;
                            if ((cN==(numN-2)) )
                            {
                                nextbN = remN;
                            }
                        }
                    }
                    if ((nextbK != bK||nextbM != bM) )
                    {
                        nextA = startAp;
                        currM = Me;
                        dma_set_stepsize( 
                            &dmagetA,
                            ((Me-(bM/8))*sizeof(double)) 
                         );
                    }
                    else
                    {
                        nextA = startA;
                        currM = ldi;
                        dma_set_stepsize( 
                            &dmagetA,
                            ((ldi-(bM/8))*sizeof(double)) 
                         );
                    }
                    if ((nextbK != bK||nextbN != bN) )
                    {
                        nextB = startBp;
                        currN = Ne;
                        dma_set_stepsize( 
                            &dmagetB,
                            ((Ne-(bN/8))*sizeof(double)) 
                         );
                    }
                    else
                    {
                        nextB = startB;
                        currN = ldw;
                        dma_set_stepsize( 
                            &dmagetB,
                            ((ldw-(bN/8))*sizeof(double)) 
                         );
                    }
                    if ((cK==(numK-1)) )
                    {
                        if ((cM==(numM-1)) )
                        {
                            dma(dmagetA, (long)(nextA), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                            dma(dmagetB, (long)((nextB+(((0*bK)*currN)+((cN+1)*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                        }
                        else
                        {
                            dma(dmagetA, (long)((nextA+(((0*bK)*currM)+((cM+1)*bM)))), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                            dma(dmagetB, (long)((nextB+(((0*bK)*currN)+(cN*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                        }
                    }
                    else
                    {
                        dma(dmagetA, (long)((nextA+((((cK+1)*bK)*currM)+(cM*bM)))), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                        dma(dmagetB, (long)((nextB+((((cK+1)*bK)*currN)+(cN*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                    }
                }

                dgemmtransasm( 
                    (double*)((local_A+((1-double_buffer_flag)*local_A_size))),
                    (double*)((local_B+((1-double_buffer_flag)*local_B_size))),
                    (double*)(local_C),
                    ((bM/8)/4),
                    ((bM/8)/4),
                    (bN/8),
                    (bK/8),
                    rid,
                    cid 
                 );
                if (((((((cN*numK)*numM)+(cM*numK))+cK)+1)<((numN*numM)*numK)) )
                {
                    dma_wait( &replygetA, 1 );
                    replygetA = 0;
                    dma_wait( &replygetB, 1 );
                    replygetB = 0;
                    double_buffer_flag = (1-double_buffer_flag);
                }
            } // end loop_cK

            if ((realbM != bM||realbN != bN) )
            {
                realC = startCp;
                currM = Me;
                dma_set_stepsize( 
                    &dmaputC,
                    ((Me-(bM/8))*sizeof(double)) 
                 );
            }
            else
            {
                realC = startC;
                currM = ldo;
                dma_set_stepsize( 
                    &dmaputC,
                    ((ldo-(bM/8))*sizeof(double)) 
                 );
            }
            dma(dmaputC, (long)((realC+(((cN*bN)*currM)+(cM*bM)))), (long)(local_C)); 
            dma_wait( &replyputC, 1 );
            replyputC = 0;
        } // end loop_cM
    } // end loop_CN

    ldm_free(local_A, sizeof(double)*local_A_size*2);
    ldm_free(local_B, sizeof(double)*local_B_size*2);
    ldm_free(local_C, sizeof(double)*local_C_size);

}



void dgemm_dma_trans_alpham1_beta1(ConvData* param)
{
    const int id = athread_get_id(-1);
    const int cid = id%8, rid = id/8;
    const int M = param->M;
    const int N = param->N;
    const int K = param->K;
    const int bM = param->blkM;
    const int bN = param->blkN;
    const int bK = param->blkK;
    const int Me = ((((M+bM)-1)/bM)*bM);
    const int Ne = ((((N+bN)-1)/bN)*bN);
    const int Ke = ((((K+bK)-1)/bK)*bK);
    const int numM = (((M+bM)-1)/bM);
    const int remM = (M-(bM*(numM-1)));
    const int numN = (((N+bN)-1)/bN);
    const int remN = (N-(bN*(numN-1)));
    const int numK = (((K+bK)-1)/bK);
    const int remK = (K-(bK*(numK-1)));
    const int ldi = param->ldi;
    const int ldw = param->ldw;
    const int ldo = param->ldo;
    int cM = 0;
    int cN = 0;
    int cK = 0;
    int i = 0;
    int realbM = bM;
    int realbN = bN;
    int realbK = bK;
    int nextbM = bM;
    int nextbN = bN;
    int nextbK = bK;
    int currM = 0;
    int currN = 0;
    int currK = 0;
    int double_buffer_flag = 0;
    int double_buffer_flag_C = 0;

    double* local_A = (double*)((doublev4*)ldm_malloc(sizeof(double)*bM*bK/8/8*2));
    const int local_A_size = bM*bK/8/8;
    double* local_B = (double*)((doublev4*)ldm_malloc(sizeof(double)*bK*bN/8/8*2));
    const int local_B_size = bK*bN/8/8;
    double* local_C = (double*)((doublev4*)ldm_malloc(sizeof(double)*bM*bN/8/8*2));
    const int local_C_size = bM*bN/8/8;
    float* fptr, *fptr2;
    double* dptr, *dptr2;
    floatv4 vflt;
    doublev4 vdbl;

    volatile int replygetA = 0, replygetB=0, replygetC=0, replyputC=0;
    dma_desc dmagetA, dmagetB, dmagetC, dmaputC;
    dma_set_op(&dmagetA, DMA_GET);
    dma_set_mode(&dmagetA, PE_MODE);
    dma_set_reply(&dmagetA, &replygetA);

    dma_set_op(&dmagetB, DMA_GET);
    dma_set_mode(&dmagetB, PE_MODE);
    dma_set_reply(&dmagetB, &replygetB);

    dma_set_op(&dmagetC, DMA_GET);
    dma_set_mode(&dmagetC, PE_MODE);
    dma_set_reply(&dmagetC, &replygetC);

    dma_set_op(&dmaputC, DMA_PUT);
    dma_set_mode(&dmaputC, PE_MODE);
    dma_set_reply(&dmaputC, &replyputC);

    double* startA = (double*)param->input + rid*bM/8 + cid*bK*ldi/8;
    double* startB = (double*)param->weight + rid*bK*ldw/8 + cid*bN/8;
    double* startC = (double*)param->output + rid*bM/8 + cid*bN*ldo/8;
    double* startAp = (double*)param->inputp + rid*bM/8 + cid*bK*Me/8;
    double* startBp = (double*)param->weightp + rid*bK*Ne/8 + cid*bN/8;
    double* startCp = (double*)param->outputp + rid*bM/8 + cid*bN*Me/8;
    double *nextA, *nextB, *realC;
    dma_set_size( &dmagetA, (((bM*bK)/64)*sizeof(double)) ); 
    dma_set_bsize( &dmagetA, ((bM/8)*sizeof(double)) ); 
    dma_set_stepsize( &dmagetA, ((ldi-(bM/8))*sizeof(double)) ); 
    dma_set_size( &dmagetB, (((bK*bN)/64)*sizeof(double)) ); 
    dma_set_bsize( &dmagetB, ((bN/8)*sizeof(double)) ); 
    dma_set_stepsize( &dmagetB, ((ldw-(bN/8))*sizeof(double)) ); 

    dma_set_size( &dmagetC, (((bM*bN)/64)*sizeof(double)) ); 
    dma_set_bsize( &dmagetC, ((bM/8)*sizeof(double)) ); 
    dma_set_stepsize( &dmagetC, ((ldo-(bM/8))*sizeof(double)) ); 
    dma_set_size( &dmaputC, (((bM*bN)/64)*sizeof(double)) ); 
    dma_set_bsize( &dmaputC, ((bM/8)*sizeof(double)) ); 
    dma_set_stepsize( &dmaputC, ((ldo-(bM/8))*sizeof(double)) ); 
    if(M < bM || K < bK){
        dma_set_stepsize( &dmagetA, ((Me-(bM/8))*sizeof(double)) ); 
        dma(dmagetA, (long)(startAp), (long)((local_A+((1-double_buffer_flag)*local_A_size)))); 
    }
    else{
        dma(dmagetA, (long)(startA), (long)((local_A+((1-double_buffer_flag)*local_A_size)))); 
    }
    if(K < bK || N < bN){
        dma_set_stepsize( &dmagetB, ((Ne-(bN/8))*sizeof(double)) ); 
        dma(dmagetB, (long)(startBp), (long)((local_B+((1-double_buffer_flag)*local_B_size)))); 
    }
    else{
        dma(dmagetB, (long)(startB), (long)((local_B+((1-double_buffer_flag)*local_B_size)))); 
    }
    
    if(M < bM || N < bN){
        dma_set_stepsize( &dmagetC, ((Me-(bM/8))*sizeof(double)) ); 
        dma(dmagetC, (long)((startCp)), (long)((local_C+((1-double_buffer_flag_C)*local_C_size))));  
    }
    else{
        dma(dmagetC, (long)((startC)), (long)((local_C+((1-double_buffer_flag_C)*local_C_size))));  
    }

    dma_wait( &replygetA, 1 );
    replygetA = 0;
    dma_wait( &replygetB, 1 );
    replygetB = 0;
    dma_wait( &replygetC, 1 );
    replygetC = 0;

    for ( cN=0; cN<numN; cN+=1 ) // begin loop_CN
    {
        if ((cN<(numN-1)) )
        {
            realbN = bN;
        }
        else
        {
            realbN = remN;
        }
        for ( cM=0; cM<numM; cM+=1 ) // begin loop_cM
        {
            if ((cM<(numM-1)) )
            {
                realbM = bM;
            }
            else
            {
                realbM = remM;
            }

            if (cN*numM+cM+1 < numN*numM){
                nextbN = realbN;
                nextbM = realbM;
                if (cM == numM-2){
                    nextbM = remM;
                }
                if (cM == numM-1){
                    if (numM != 1) nextbM = bM;
                    else nextbM = remM;
                    if (cN == numN-2){
                        nextbN = remN;
                    }
                }

                if ((nextbM != bM||nextbN != bN) )
                {
                    realC = startCp;
                    currM = Me;
                    dma_set_stepsize( 
                        &dmagetC,
                        ((Me-(bM/8))*sizeof(double)) 
                     );
                }
                else
                {
                    realC = startC;
                    currM = ldo;
                    dma_set_stepsize( 
                        &dmagetC,
                        ((ldo-(bM/8))*sizeof(double)) 
                     );
                }

                if (cM == numM-1){
                        dma(dmagetC, (long)((realC+((((cN+1)*bN)*currM)+(0*bM)))), (long)((local_C+(double_buffer_flag_C*local_C_size)))); 
                }
                else{
                        dma(dmagetC, (long)((realC+(((cN*bN)*currM)+((cM+1)*bM)))), (long)((local_C+(double_buffer_flag_C*local_C_size)))); 
                }
                
            }

/*
            if ((realbM != bM||realbN != bN) )
            {
                realC = startCp;
                currM = Me;
                dma_set_stepsize( 
                    &dmagetC,
                    ((Me-(bM/8))*sizeof(double)) 
                 );
            }
            else
            {
                realC = startC;
                currM = ldo;
                dma_set_stepsize( 
                    &dmagetC,
                    ((ldo-(bM/8))*sizeof(double)) 
                 );
            }
            dma(dmagetC, (long)((realC+(((cN*bN)*currM)+(cM*bM)))), (long)(local_C));
            dma_wait(&replygetC, 1);
            replygetC = 0;
*/            
            //for ( i=0; i<local_C_size; i+=1 ) // begin init_C
            //{
            //    local_C[i] = 0;
            //} // end init_C
            for ( cK=0; cK<numK; cK+=1 ) // begin loop_cK
            {
                if ((cK<(numK-1)) )
                {
                    realbK = bK;
                }
                else
                {
                    realbK = remK;
                }
                if (((((((cN*numK)*numM)+(cM*numK))+cK)+1)<((numN*numM)*numK)) )
                {
                    nextbK = realbK;
                    nextbM = realbM;
                    nextbN = realbN;
                    if ((cK==(numK-2)) )
                    {
                        nextbK = remK;
                    }
                    if ((cK==(numK-1)) )
                    {
                        //fix a BUG :: when K < bK
                        if (numK != 1) nextbK = bK;
                        else nextbK = remK;
                        if ((cM==(numM-2)) )
                        {
                            nextbM = remM;
                        }
                        if ((cM==(numM-1)) )
                        {
                            //fix a BUG :: when M < bM
                            if (numM != 1) nextbM = bM;
                            else nextbM = remM;
                            if ((cN==(numN-2)) )
                            {
                                nextbN = remN;
                            }
                        }
                    }
                    if ((nextbK != bK||nextbM != bM) )
                    {
                        nextA = startAp;
                        currM = Me;
                        dma_set_stepsize( 
                            &dmagetA,
                            ((Me-(bM/8))*sizeof(double)) 
                         );
                    }
                    else
                    {
                        nextA = startA;
                        currM = ldi;
                        dma_set_stepsize( 
                            &dmagetA,
                            ((ldi-(bM/8))*sizeof(double)) 
                         );
                    }
                    if ((nextbK != bK||nextbN != bN) )
                    {
                        nextB = startBp;
                        currN = Ne;
                        dma_set_stepsize( 
                            &dmagetB,
                            ((Ne-(bN/8))*sizeof(double)) 
                         );
                    }
                    else
                    {
                        nextB = startB;
                        currN = ldw;
                        dma_set_stepsize( 
                            &dmagetB,
                            ((ldw-(bN/8))*sizeof(double)) 
                         );
                    }
                    if ((cK==(numK-1)) )
                    {
                        if ((cM==(numM-1)) )
                        {
                            dma(dmagetA, (long)(nextA), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                            dma(dmagetB, (long)((nextB+(((0*bK)*currN)+((cN+1)*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                        }
                        else
                        {
                            dma(dmagetA, (long)((nextA+(((0*bK)*currM)+((cM+1)*bM)))), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                            dma(dmagetB, (long)((nextB+(((0*bK)*currN)+(cN*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                        }
                    }
                    else
                    {
                        dma(dmagetA, (long)((nextA+((((cK+1)*bK)*currM)+(cM*bM)))), (long)((local_A+(double_buffer_flag*local_A_size)))); 
                        dma(dmagetB, (long)((nextB+((((cK+1)*bK)*currN)+(cN*bN)))), (long)((local_B+(double_buffer_flag*local_B_size)))); 
                    }
                }

                dgemmtransasmm( 
                    (double*)((local_A+((1-double_buffer_flag)*local_A_size))),
                    (double*)((local_B+((1-double_buffer_flag)*local_B_size))),
                    //(double*)(local_C),
                    (double*)((local_C+((1-double_buffer_flag_C)*local_C_size))),
                    ((bM/8)/4),
                    ((bM/8)/4),
                    (bN/8),
                    (bK/8),
                    rid,
                    cid 
                 );
                if (((((((cN*numK)*numM)+(cM*numK))+cK)+1)<((numN*numM)*numK)) )
                {
                    dma_wait( &replygetA, 1 );
                    replygetA = 0;
                    dma_wait( &replygetB, 1 );
                    replygetB = 0;
                    double_buffer_flag = (1-double_buffer_flag);
                }
            } // end loop_cK

            if ((realbM != bM||realbN != bN) )
            {
                realC = startCp;
                currM = Me;
                dma_set_stepsize( 
                    &dmaputC,
                    ((Me-(bM/8))*sizeof(double)) 
                 );
            }
            else
            {
                realC = startC;
                currM = ldo;
                dma_set_stepsize( 
                    &dmaputC,
                    ((ldo-(bM/8))*sizeof(double)) 
                 );
            }
            //dma(dmaputC, (long)((realC+(((cN*bN)*currM)+(cM*bM)))), (long)(local_C)); 
            dma(dmaputC, (long)((realC+(((cN*bN)*currM)+(cM*bM)))), (long)(local_C+((1-double_buffer_flag_C)*local_C_size))); 
            dma_wait( &replyputC, 1 );
            replyputC = 0;

            if (cN*numM+cM+1 < numN*numM){
                dma_wait(&replygetC, 1);
                replygetC = 0;
                double_buffer_flag_C = (1-double_buffer_flag_C);
            }

        } // end loop_cM
    } // end loop_CN

    ldm_free(local_A, sizeof(double)*local_A_size*2);
    ldm_free(local_B, sizeof(double)*local_B_size*2);
    ldm_free(local_C, sizeof(double)*local_C_size*2);

}

#include <stdlib.h>
void *_aligned_malloc(size_t size, size_t align_size)
{
  void *tempPtr = malloc(size + align_size);
  char offset = align_size - ((long)tempPtr % align_size);
  char *alignedPtr = (char*)tempPtr + offset;
  alignedPtr[-1] = offset;
  return (void*)alignedPtr;
}
void _aligned_free(void *ptr)
{
  char offset = ((char*)ptr)[-1];
  free((char*)ptr - offset);
}

void sw_blas_init() {
  athread_init();
}
void sw_blas_stop() {
  athread_halt();
}

void zeropad_matrix(const float* A, int ld, int ld_pad, int hd, int hd_pad, float* A_zeropad) {
  int i, j;
  memset(A_zeropad, 0, ld_pad*hd_pad*sizeof(float));
  for(i = 0; i < hd; ++i)
    for(j = 0; j < ld; ++j) {
      A_zeropad[i*ld_pad + j] = A[i*ld + j];
  }
/*
  int cnt = 0;
  for(i = 0; i < ld_pad*hd_pad; ++i)
    if(A_zeropad2[i] != A_zeropad[i])
      printf("%d %d %lf vs %lf\n", i/ld_pad, i%ld_pad, A_zeropad2[i], A_zeropad[i]);

  free(A_zeropad2);
      */
  return;
}


/*******
 * a light weight zero padding function
 * copy boundary into auxilary buff
 * ld : low dim size 
 * ld_pad : low dim size after pad 
 * l_pad : low dim pad isze
 * *****/
void lw_zeropad_matrix(const float* A, int ld, int ld_pad, int l_pad, int hd, int hd_pad, int h_pad, float* A_zeropad) {
  int i = 0, j = 0;
  if(ld == ld_pad && hd == hd_pad)
    return;
  for(i = hd_pad - h_pad; i < hd; ++i)
    for(j = ld_pad - l_pad; j < ld; ++j) {
      A_zeropad[i*ld_pad + j] = A[i*ld + j];
  }
}

void depad_matrix(float* A, int ld, int ld_pad, int hd, int hd_pad, const float* A_zeropad) {
  int i, j;
  for(i = 0; i < hd; ++i)
    for(j = 0; j < ld; ++j) {
      A[i*ld + j] = A_zeropad[i*ld_pad + j];
    }
  return;
}

void sw_zeropad_matrix(const float* A, int ld, int ld_pad, int hd, int hd_pad, float* A_zeropad);
void sw_depad_matrix(float* A, int ld, int ld_pad, int hd, int hd_pad, const float* A_zeropad);

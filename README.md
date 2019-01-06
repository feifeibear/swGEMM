# swGEMM: a customized GEMM library for swDNN

## Usage:
### 1. build test/debug case use Makefile
make - generate a test case using test.c
make ar - generate swblas.a

### 2. build release library: use cmake
mkdir build
cd build && cmake ..

### 3. use swGEMM in other program - 
link ./build/libswBLASlib.a  and include cblas.h swblas.h

### 4. debug or unitest
A test case in ./test/
sh run.sh $M $K $N

### MACRO:
-DUSE_RTC count time inside CPE
-DUSE_COMP without it, you will get DMA time
-DCHECK_RES check answer with xMath

## API
void sw_sgemm_trans(float* input, float* weight, float* output, int M, int N, int K, int blkM, int blkN, int blkK);
input(K, M) * weight(K, N) -> output (N, K)
input, weight , output are in 2D matrix (high dim, low dim)
blkM/N/K are block size on the corresponding dimension.
Requirments : M and blkM should be 128x, K and blkK should be 8x, N and blkN should be 32x;

## Profile
sh ./auto_test.sh
python ./show_raw_data.py

## BUGs Report
1. use -O1 rather than -O2 for sw_slave_XXX files, otherwise you will get stuck
2. function name in ./asm should not be too long. For example, dgemmasmnoinit will not pass compilation
3. If you need to use SIMD inside CPE, you should allocate LDM space with points in type of floatv4*/doublev4
4. When we use ./build/libswBLASlib.a in other code, accessing MBW map will cause unpredicatable bug! Maybe allocate a large array
in stack space is not supported very well.

## Warning
rpcc time is different with timer for eslapse bwteen athread spawn and join.
if you use rpcc to get time, you will get wrong time in MPE.
Maybe athread time is large in small case.

## Author
Jiarui Fang [THU and NSCCWX] <\br>
fang_jiarui@163.com


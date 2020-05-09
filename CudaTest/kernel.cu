#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include "helper_functions.h" // helper utility functions 

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <opencv2/opencv.hpp>

using namespace cv;

cudaError_t lbpMultiscaleCuda(uint8_t* const c, const uint8_t* const a);

__global__ void test() {

}

template<typename T>
__device__ inline T my_abs_dif(T a, T b) {
    return a > b ? (a - b) : (b - a);
}

__global__ void myKernel(int8_t* d, uint8_t* const c, const uint8_t* const a, uint32_t w, uint32_t h, uint32_t r)
{

    //int8_t d[8][2] = { {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1} };

    uint8_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t prev = -1;
    uint8_t U = 0;
    uint8_t D= 0;
    uint8_t M = 0;
    uint16_t temp = 0;

            uint16_t    pc;
    const   uint8_t     pa   = *(a + (i * w + j));


    pc = 0;
    c[i * w + j] = 0;
   // return;

    if (i > r && j > r && i + r < h - 1 && j + r < w - 1) {
        for(uint8_t ii = 0; ii < 16; ii+=2){
            temp = my_abs_dif(pa, a[(i + d[ii]) * w + (j + d[ii + 1])]);
            
            if (M < temp) {
                D = 7-(ii>>1);
                M = temp;
            }
        }


        for (uint8_t ii = 0; ii < 16; ii+=2) {
            pc <<= 1;
            pc |= (a[(i + d[ii]) * w + (j + d[ii + 1])] > pa);

            if (prev > -1)
                U += (pc ^ prev) & 1;
            prev = pc;
        }

        
        if (U + (pc ^ (pc >> 7)) > 2) 
            c[i * w + j] = 9;
            
        else {
            temp = (pc >> D) | (pc << (8 - D));

            c[i * w + j] = temp;
        }
    }

}


void kernelTest(int8_t* d, uint8_t* const c, const uint8_t* const a, uint32_t w, uint32_t h, uint32_t r) {

    
    int32_t prev = -1;
    uint8_t U = 0;
    uint8_t D = 0;
    uint8_t M = 0;
    uint16_t temp = 0;

    
    // return;
    for(uint8_t i = 0; i<h; i++)
        for (uint8_t j = 0; j < w; j++) {
            uint16_t    pc;
            const   uint8_t     pa = *(a + (i * w + j));

            pc = 0;
            c[i * w + j] = 0;

            if (i > r && j > r && i + r < h - 1 && j + r < w - 1) {
                for (uint8_t ii = 0; ii < 16; ii += 2) {
                    temp = pa - a[(i + d[ii]) * w + (j + d[ii + 1])] > 0 ? pa - a[(i + d[ii]) * w + (j + d[ii + 1])] : -(pa - a[(i + d[ii]) * w + (j + d[ii + 1])]);

                    if (M < temp) {
                        D = 7 - (ii >> 1);
                        M = temp;
                    }
                }


                for (uint8_t ii = 0; ii < 16; ii += 2) {
                    pc <<= 1;
                    pc |= (a[(i + d[ii]) * w + (j + d[ii + 1])] > pa);

                    if (prev > -1)
                        U += (pc ^ prev) & 1;
                    prev = pc;
                }


                if (U + (pc ^ (pc >> 7)) > 2)
                    c[i * w + j] = 9;

                else {
                    temp = (pc >> D) | (pc << (8 - D));

                    c[i * w + j] = temp;
                }
            }
        }
}


cudaError_t lbpMultiscaleCuda(uint8_t* const c, const uint8_t* const a)
{
    printf("1\n");

    uint8_t     w = 64,
        h = 64;
    uint16_t size = w * h;

    dim3 numThreads(16, 32);
    dim3 numBlocks(4, 2);

    uint8_t* dev_a = 0;
    uint8_t* dev_c = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc(&dev_c, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dev_a, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    int8_t d[8][2] = { {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1} };
    int8_t* dev_d = 0;
    int8_t* d1 = 0;


    cudaStatus = cudaMalloc(&dev_d, 16);
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_d, d, 16, cudaMemcpyHostToDevice);
    //cudaStatus = cudaMemcpy(d, dev_d, 16, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    d1 = (int8_t*)malloc(16);
    cudaMemcpy(d1, d, 16, cudaMemcpyHostToHost);

    float gpu_time = 0.f;
    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface* timer = NULL;


    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);


    checkCudaErrors(cudaDeviceSynchronize());

    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    //test <<<1, 256, 0, 0 >>> ();
    for (int i = 0; i < 10000; i++) 
        myKernel <<< numBlocks, numThreads >>> (dev_d, dev_c, dev_a, w, h, 0);
        //kernelTest(d1, c, a, w, h, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    //checkCudaErrors(cudaDeviceSynchronize());
    cudaStatus = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    // Launch a kernel on the GPU with one thread for each element.


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


Error:
    cudaFree(dev_c);
    cudaFree(dev_a);

    return cudaStatus;
}


int main(int argc, char* argv[]) {
    Mat img = imread("H:\\Downloads\\NormalizedFace\\ClientNormalized\\0004\\0004_01_00_01_7.bmp", IMREAD_GRAYSCALE);
    Mat dst(img.size(), CV_8U);

    uint8_t* const pimg = img.ptr();
    uint8_t* const pdst = dst.ptr();

    printf("1\n");
    lbpMultiscaleCuda(pdst, pimg);
    


    //printf("%d, %d", c[0], c[4]);
    return 0;
}
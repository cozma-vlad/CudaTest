#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

#include <cooperative_groups.h>
#include "helper_cuda.h"
#include "helper_functions.h" // helper utility functions 

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstdint>
#include <cstdlib>


using namespace cv;
namespace cg = cooperative_groups;

cudaError_t lbpMultiscaleCuda(uint8_t* const c, const uint8_t* const a);

template<typename T>
__device__ inline T my_abs_dif(T a, T b) {
    return a > b ? (a - b) : (b - a);
}

__constant__ int8_t d[16]; 
__global__ void lbpKernel(uint8_t* const c, const uint8_t* const a, uint32_t w, uint32_t h, uint32_t r)
{

    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;


    int32_t prev = -1;
    uint8_t U = 0;
    uint8_t D = 0;
    uint8_t M = 0;
    uint16_t temp = 0;

            uint16_t    pc   = 0;
    const   uint8_t     pa   = a[i * w + j];

    c[i * w + j] = pa;
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

void lbpKernelCpu(uint8_t* const c, const uint8_t* const a, uint32_t w, uint32_t h, uint32_t r) {

    int32_t prev = -1;
    uint8_t U = 0;
    uint8_t D = 0;
    uint8_t M = 0;
    uint16_t temp = 0;

    uint16_t    pc = 0;

    for(int i=0; i<h/2; i++)
        for (int j = 0; j < w/2; j++) {
            const   uint8_t     pa = *(a + (i * w + j));

            c[i * w + j] = 0;
            // return;

            if (i > r && j > r && i + r < h - 1 && j + r < w - 1) {

                for (uint8_t ii = 0; ii < 16; ii += 2) {
                    temp = abs(pa - a[(i + d[ii]) * w + (j + d[ii + 1])]) > M ? abs(pa - a[(i + d[ii]) * w + (j + d[ii + 1])]) : M;

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


__global__ void histogramKernel(uint32_t H[256], uint8_t* a, uint32_t w) {
    uint32_t l_hist = 0;

    uint32_t j = threadIdx.x;
    uint32_t i = threadIdx.y;
    uint16_t pos = i * (w>>2) + j;


    uint32_t temp = ((uint32_t*)a)[pos + blockIdx.y * blockDim.y * blockDim.x];


#pragma unroll
    for (int i = 0; i < 4; i++) {
        l_hist <<= 8;
        l_hist |= (temp & 0xFF);
        temp >>= 8;
    }
  
#pragma unroll
    for (int i = 0; i < 4; i++) {
        atomicAdd(H + (l_hist & 0xFF), 1);
        l_hist >>= 8;
    }
}

void cpu_hist(uint32_t H[256], const uint8_t* const a, int w, int h) {

    for(int i=0; i<h; i++)
        for (int j = 0; j < w; j++) 
            H[a[i * w + j]]++;
}

cudaError_t lbpMultiscaleCuda(uint8_t* const c, const uint8_t* const a)
{

    int8_t d_temp[16] = { 0, -1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0, -1, -1 };
    cudaMemcpyToSymbol(d, d_temp, 16);
    cudaMemcpyFromSymbol(d_temp, d, 16);

    uint8_t     w = 64,
        h = 64;
    uint16_t size = w * h;

    dim3 numThreads(16, 32);
    dim3 numBlocks1(4, 2);
    dim3 numBlocks2(1, 2);

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
    
    uint32_t hist[256], *d_hist, hist2[256];
    cudaMalloc(&d_hist, 256<<2);

    memset(hist , 0, sizeof(hist ));
    memset(hist2, 0, sizeof(hist2));
    cudaMemset(d_hist, 0, sizeof(hist));

    lbpKernel <<< numBlocks1, numThreads >>> (dev_c, dev_a, w, h, 0);
    cudaStatus = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stdout, "cudaMalloc failed!");
        goto Error;
    }

    histogramKernel << <numBlocks2, numThreads >> > (d_hist, dev_c, w);
    checkCudaErrors(cudaDeviceSynchronize());
    
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    

    
    cudaMemcpy(hist, d_hist, 256 << 2, cudaMemcpyDeviceToHost);
    
    cudaFree(d_hist);


    //checkCudaErrors(cudaDeviceSynchronize());
    

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


cudaError_t lbpMultiscaleCudaCpu(uint8_t* const c, const uint8_t* const a)
{

    int8_t d_temp[16] = { 0, -1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0, -1, -1 };
    cudaMemcpyToSymbol(d, d_temp, 16);
    cudaMemcpyFromSymbol(d_temp, d, 16);

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
    for (int i = 0; i < 10000; i++) {
        lbpKernelCpu(c, a, w, h, 0);
        //cudaDeviceSynchronize();
    }
    //kernelTest(d1, c, a, w, h, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    //checkCudaErrors(cudaDeviceSynchronize());


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
    // any errors encountered during the launch

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);

    return cudaStatus;
}



int main(int argc, char* argv[]) {
    Mat img = imread("C:\\users\\vlads\\desktop\\download.jfif", IMREAD_GRAYSCALE);
    resize(img, img, Size(64, 64), 0, 0);

    Mat dst(img.size(), CV_8U);

    uint8_t* const pimg = img.ptr();
    uint8_t* const pdst = dst.ptr();

    /*for (uint16_t i = 0; i < 16; i++)
    {
        lbpMultiscaleCuda(pdst, pimg);
    }*/


    lbpMultiscaleCuda(pdst, pimg);
    
    return 0;
}

int main1() {
    int8_t d[16] = { 0, -1, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, -1, 0, -1, -1 };
    int m[3][3] = {
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
                    };

    int i = 1, j = 1;
    uint8_t M=0, D=0, temp, U=0, w=64, pc=0;
     int8_t prev = -1;


    for (uint8_t ii = 0; ii < 16; ii += 2) {
        temp = abs(m[1][1] - m[1 + d[ii]][1 + d[ii + 1]]) > M ? abs(m[1][1] - m[1 + d[ii]][1 + d[ii + 1]]) : M;

        if (M < temp) {
            D = 7 - (ii >> 1);
            M = temp;
        }
    }

    for (uint8_t ii = 0; ii < 16; ii += 2) {
        pc <<= 1;
        pc |= (m[(i + d[ii])][(j + d[ii + 1])] > m[1][1]);

        if (prev > -1)
            U += (pc ^ prev) & 1;
        prev = pc;
    }


    temp = (pc >> D) | (pc << (8 - D));

    return 0;
}
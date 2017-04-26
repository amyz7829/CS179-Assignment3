/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/*
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v,
    cufftComplex *out_data,
    int padded_length) {

    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response.

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them.

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    while(idx < padded_length){
      float a = raw_data[idx].x;
      float b = raw_data[idx].y;

      float c = impulse_v[idx].x;
      float d = impulse_v[idx].y;

      cufftComplex ans;
      ans.x = (a * c - b * d) / (float) padded_length;
      ans.y = (a * d + b * c) / (float) padded_length;

      out_data[idx] = ans;

      idx += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others.

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */

      /*
          My first optimization was to ensure that in the first step, every thread had a
          job, preventing warp divergence, and also using all resources. Every thread
          looks at the nth block and the (n + 1)th block, and compares the mth values in
          these two blocks. It then stores the larger of these two values to an array
          in shared memory. This array is the size of the block, as we are halving from
          two blocks.

          We then must unfortunately warp diverge, and find the maximum value in each
          block by using a for loop. After this is done, we run atomic max, finding the
          maximum value across all blocks.
       */
       int tid = threadIdx.x;
      int idx = blockIdx.x * (2 * blockDim.x) + tid;
      extern __shared__ float data[];

       while(idx + blockDim.x < padded_length){
         data[tid] = fmax(out_data[idx].x, out_data[idx + blockDim.x].x);

         __syncthreads();

         if(tid == 0){
           float localMax = data[tid];
           for(unsigned int threadIndex = 1; threadIndex < blockDim.x; threadIndex++){
             localMax = fmax(localMax, data[threadIndex]);
           }
           atomicMax(max_abs_val, localMax);
         }
         idx += gridDim.x * blockDim.x;
       }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val.

    This kernel should be quite short.
    */
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    while(idx < (uint) padded_length){
      out_data[idx].x = out_data[idx].x / *max_abs_val;
      idx += gridDim.x * blockDim.x;
    }

}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {

    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data,
    padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    /*
      This is called with only half of the threads per block because we want each thread
      to initially start by handling two values at once (by looking at two blocks, and
      comparing the nth element of each)
    */
    cudaMaximumKernel<<<blocks / 2, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(out_data, max_abs_val, padded_length);

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}

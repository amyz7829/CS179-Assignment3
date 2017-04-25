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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while(idx < padded_length){
      for(int i = 0; i < padded_length; i++){
        out_data[idx].x += ((raw_data[i].x * impulse_v[(idx - i) % padded_length].x) -
                            raw_data[i].y * impulse_v[(idx - i) % padded_length].y);
        out_data[idx].y += ((raw_data[i].x * impulse_v[(idx - i) % padded_length].y) -
                            raw_data[i].y * impulse_v[(idx - i) % padded_length].x);
      }
      out_data[idx].x = out_data[idx].x / padded_length;
      out_data[idx].y = out_data[idx].y / padded_length;

      idx += blockIdx.x * blockDim.x;
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

    /* Step 1: Find the max value in out data through reduction. Use sequential
       padding for each block, and make each warp read two blocks? (suggested
       on Nvidia site). This starts by storing out_data length / 2 values in
       shared data, and then slowly cutting by half each time until the max
       value is found. Keep in mind reading in a coalesced way, and also
       avoiding bank conflicts! Sequential should prevent bank conflicts.

       Step 2: After this is done, divide every value in the original out data
       array by the found value. Not much better way than just doing it per
       thread?
       */
       int tid = threadIdx.x;
       int idx = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
       __shared__ float data[32];

       float localMax = fmaxf(out_data[idx].x, out_data[idx + blockDim.x].x);
       data[tid] = localMax;
       __syncthreads();

       if(tid == 0){
         for(unsigned int threadIndex = 1; threadIndex < blockDim.x; threadIndex++){
           localMax = fmax(localMax, data[threadIndex]);
         }
         atomicMax(max_abs_val, localMax);
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while(idx < padded_length){
      out_data[idx].x = out_data[idx].x / *max_abs_val;
      idx += blockIdx.x * blockDim.x;
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


    cudaMaximumKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}

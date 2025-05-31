#include <cuda_runtime.h>
#include <npp.h>

// CUDA kernel to compute magnitude from Sobel gradients
__global__ void sobel_magnitude_kernel(const short* grad_x, const short* grad_y, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    float mag = sqrtf(float(grad_x[idx]) * grad_x[idx] + float(grad_y[idx]) * grad_y[idx]);
    mag = fminf(fmaxf(mag, 0.0f), 255.0f);
    output[idx] = (unsigned char)mag;
}

#include <nppi.h>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" void rgb_to_grayscale_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);

extern "C" void sobel_edge_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    NppiSize oSizeROI = { width, height };
    int srcStep = width * channels;
    // int dstStep = width * channels;
    NppStatus status;

    // If input is not grayscale, convert to grayscale first
    if (channels != 1) {
        printf("--NPP Sobel only supports 1 channel (grayscale) images in this implementation. Chaning to grayscale\n");        unsigned char* d_gray = nullptr;
        cudaError_t err = cudaMalloc(&d_gray, width * height * sizeof(unsigned char));
        if (err != cudaSuccess) { printf("cudaMalloc failed for d_gray\n"); return; }
        rgb_to_grayscale_npp(d_input, d_gray, width, height, channels);
        // Now run Sobel on grayscale
        sobel_edge_npp(d_gray, d_output, width, height, 1);
        cudaFree(d_gray);
        return;
    }
    short *d_grad_x = nullptr, *d_grad_y = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_grad_x, width * height * sizeof(short));
    if (err != cudaSuccess) { printf("cudaMalloc failed for d_grad_x\n"); return; }
    err = cudaMalloc(&d_grad_y, width * height * sizeof(short));
    if (err != cudaSuccess) { printf("cudaMalloc failed for d_grad_y\n"); cudaFree(d_grad_x); return; }

    status = nppiFilterSobelHoriz_8u16s_C1R(d_input, srcStep, d_grad_x, width * sizeof(short), oSizeROI, NPP_MASK_SIZE_3_X_3);
    if (status != NPP_SUCCESS) printf("NPP Sobel horiz failed: %d\n", status);
    status = nppiFilterSobelVert_8u16s_C1R(d_input, srcStep, d_grad_y, width * sizeof(short), oSizeROI, NPP_MASK_SIZE_3_X_3);
    if (status != NPP_SUCCESS) printf("NPP Sobel vert failed: %d\n", status);
    // Compute magnitude using CUDA kernel
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    sobel_magnitude_kernel<<<grid, block>>>(d_grad_x, d_grad_y, d_output, width, height);
    cudaDeviceSynchronize();
    cudaFree(d_grad_x);
    cudaFree(d_grad_y);
}

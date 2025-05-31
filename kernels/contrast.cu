#include <npp.h>
#include <nppi.h>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" void contrast_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float contrast_factor) {
    cudaFree(0);
    NppiSize oSizeROI = { width, height };
    int srcStep = width * channels;
    int dstStep = width * channels;
    NppStatus status;

    if (channels == 1) {
        // Mathematically correct float contrast scaling for grayscale: 8u -> 32f, scale, 32f -> 8u
        size_t num_pixels = width * height;
        size_t float_bytes = num_pixels * sizeof(float);
        float* d_float = nullptr;
        float* d_float_out = nullptr;
        cudaMalloc(&d_float, float_bytes);
        cudaMalloc(&d_float_out, float_bytes);
        // Convert 8u to 32f
        NppStatus st1 = nppiConvert_8u32f_C1R(d_input, srcStep, d_float, width * sizeof(float), oSizeROI);
        if (st1 != NPP_SUCCESS) {
            printf("NPP convert 8u->32f failed: %d\n", st1);
            cudaFree(d_float);
            cudaFree(d_float_out);
            return;
        }
        // Apply float scaling
        NppStatus st2 = nppiMulC_32f_C1R(d_float, width * sizeof(float), contrast_factor, d_float_out, width * sizeof(float), oSizeROI);
        if (st2 != NPP_SUCCESS) {
            printf("NPP MulC_32f_C1R failed: %d\n", st2);
            cudaFree(d_float);
            cudaFree(d_float_out);
            return;
        }
        // Convert back to 8u
        NppStatus st3 = nppiConvert_32f8u_C1R(d_float_out, width * sizeof(float), d_output, dstStep, oSizeROI, NPP_RND_NEAR);
        if (st3 != NPP_SUCCESS) {
            printf("NPP convert 32f->8u failed: %d\n", st3);
        }
        cudaFree(d_float);
        cudaFree(d_float_out);
        status = st3;

    } else if (channels == 3) {
        // Mathematically correct float contrast scaling for RGB: 8u -> 32f, scale, 32f -> 8u
        size_t num_pixels = width * height;
        size_t float_bytes = num_pixels * channels * sizeof(float);
        float* d_float = nullptr;
        float* d_float_out = nullptr;
        cudaMalloc(&d_float, float_bytes);
        cudaMalloc(&d_float_out, float_bytes);
        // Convert 8u to 32f
        NppStatus st1 = nppiConvert_8u32f_C3R(d_input, srcStep, d_float, width * 3 * sizeof(float), oSizeROI);
        if (st1 != NPP_SUCCESS) {
            printf("NPP convert 8u->32f failed: %d\n", st1);
            cudaFree(d_float);
            cudaFree(d_float_out);
            return;
        }
        // Apply float scaling
        float factors[3] = { contrast_factor, contrast_factor, contrast_factor };
        NppStatus st2 = nppiMulC_32f_C3R(d_float, width * 3 * sizeof(float), factors, d_float_out, width * 3 * sizeof(float), oSizeROI);
        if (st2 != NPP_SUCCESS) {
            printf("NPP MulC_32f_C3R failed: %d\n", st2);
            cudaFree(d_float);
            cudaFree(d_float_out);
            return;
        }
        // Convert back to 8u
        NppStatus st3 = nppiConvert_32f8u_C3R(d_float_out, width * 3 * sizeof(float), d_output, dstStep, oSizeROI, NPP_RND_NEAR);
        if (st3 != NPP_SUCCESS) {
            printf("NPP convert 32f->8u failed: %d\n", st3);
        }
        cudaFree(d_float);
        cudaFree(d_float_out);
        status = st3;
    }
    if (status != NPP_SUCCESS) {
        printf("NPP contrast failed with status %d\n", status);
    }
}

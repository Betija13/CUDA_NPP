#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cstdio>

extern "C" void gaussian_blur_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, int kernelSizeInt) {
    NppiSize oSizeROI = { width, height };
    int srcStep = width * channels;
    int dstStep = width * channels;
    NppiMaskSize kernelSize;
    if (kernelSizeInt == 33){
        kernelSize = NPP_MASK_SIZE_3_X_3;
    } else if (kernelSizeInt == 55){
        kernelSize = NPP_MASK_SIZE_5_X_5;
    } else if (kernelSizeInt == 77){
        kernelSize = NPP_MASK_SIZE_7_X_7;
    } else if (kernelSizeInt == 99){
        kernelSize = NPP_MASK_SIZE_9_X_9;
    } else if (kernelSizeInt == 1111){
        kernelSize = NPP_MASK_SIZE_11_X_11;
    } else if (kernelSizeInt == 1313){
        kernelSize = NPP_MASK_SIZE_13_X_13;
    } else if (kernelSizeInt == 1515){
        kernelSize = NPP_MASK_SIZE_15_X_15;
    }
    NppStatus status;

    if (channels == 1) {
        status = nppiFilterGauss_8u_C1R(d_input, srcStep, d_output, dstStep, oSizeROI, kernelSize);
    } else if (channels == 3) {
        status = nppiFilterGauss_8u_C3R(d_input, srcStep, d_output, dstStep, oSizeROI, kernelSize);
    } else {
        printf("NPP Gaussian blur only supports 1 or 3 channel images.\n");
        return;
    }
    if (status != NPP_SUCCESS) {
        printf("NPP Gaussian blur failed with status %d\n", status);
    }
}

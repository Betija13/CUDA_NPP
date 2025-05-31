#include <npp.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void rgb_to_grayscale_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    if (channels != 3) {
        printf("NPP grayscale conversion only supports 3-channel RGB images.\n");
        return;
    }
    int srcStep = width * channels * sizeof(unsigned char);
    int dstStep = width * sizeof(unsigned char);
    NppiSize oSizeROI = { width, height };

    NppStatus status = nppiRGBToGray_8u_C3C1R(
        d_input, srcStep,
        d_output, dstStep,
        oSizeROI
    );
    if (status != NPP_SUCCESS) {
        printf("NPP error: %d\n", status);
    }
}

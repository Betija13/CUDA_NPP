#include <npp.h>
#include <nppi.h>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" void brightness_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, int brightness_delta) {
    cudaFree(0); // Ensure CUDA context
    NppiSize oSizeROI = { width, height };
    int srcStep = width * channels;
    int dstStep = width * channels;
    NppStatus status;

    if (channels == 1) {
        unsigned char delta = (unsigned char)brightness_delta;
        status = nppiAddC_8u_C1RSfs(
            d_input, srcStep,
            delta,
            d_output, dstStep,
            oSizeROI, 0 // scale factor 0 = no scaling
        );
    } else if (channels == 3) {
        unsigned char deltas[3] = { (unsigned char)brightness_delta, (unsigned char)brightness_delta, (unsigned char)brightness_delta };
        status = nppiAddC_8u_C3RSfs(
            d_input, srcStep,
            deltas,
            d_output, dstStep,
            oSizeROI, 0
        );
    } else {
        printf("NPP brightness only supports 1 or 3 channel images.\n");
        return;
    }
    if (status != NPP_SUCCESS) {
        printf("NPP brightness failed with status %d\n", status);
    }
}

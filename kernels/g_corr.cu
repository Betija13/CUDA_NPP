#include <npp.h>
#include <nppi.h>
#include <cstdio>

extern "C" void g_corr(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    NppiSize oSizeROI = { width, height };
    if (channels != 3) {
        printf("g_corr only supports 3-channel RGB images.\n");
        return;
    } else {
        int srcStep = width * 3 * sizeof(unsigned char);
        int dstStep = srcStep;
        NppStatus status = nppiGammaFwd_8u_C3R(
            d_input, srcStep,
            d_output, dstStep,
            oSizeROI
        );
        if (status != NPP_SUCCESS) {
            printf("nppiGammaFwd_8u_C3R failed with status %d\n", status);
        } else {
            printf("--Gamma correction (RGB) completed successfully.\n");
        }
    }
}


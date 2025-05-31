#include <npp.h>
#include <nppi.h>
#include <cstdio>
#include <math.h>

extern "C" void lut(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float gamma) {

    NppiSize oSizeROI = { width, height };
    if (channels == 3) {

        int srcStep = width * 3 * sizeof(unsigned char);
        int dstStep = srcStep;
        // Create LUT values and levels for each channel (Npp32s)
        Npp32s* h_pValues[3];
        Npp32s* h_pLevels[3];
        Npp32s* d_pValues[3];
        Npp32s* d_pLevels[3];
        int nLevels[3] = {256, 256, 256};
        for (int c = 0; c < 3; ++c) {
            h_pValues[c] = (Npp32s*)malloc(256 * sizeof(Npp32s));
            h_pLevels[c] = (Npp32s*)malloc(256 * sizeof(Npp32s));
            for (int i = 0; i < 256; ++i) {
                float norm = ((float)i) / 255.0f;
                float gamma_safe = gamma > 0.0f ? gamma : 1.0f;
                h_pLevels[c][i] = i;
                h_pValues[c][i] = (Npp32s)(powf(norm, gamma_safe) * 255.0f + 0.5f);
            }
            cudaMalloc(&d_pValues[c], 256 * sizeof(Npp32s));
            cudaMalloc(&d_pLevels[c], 256 * sizeof(Npp32s));
            cudaMemcpy(d_pValues[c], h_pValues[c], 256 * sizeof(Npp32s), cudaMemcpyHostToDevice);
            cudaMemcpy(d_pLevels[c], h_pLevels[c], 256 * sizeof(Npp32s), cudaMemcpyHostToDevice);
            free(h_pValues[c]);
            free(h_pLevels[c]);
        }
        NppStatus status = nppiLUT_8u_C3R(
            d_input, srcStep,
            d_output, dstStep,
            oSizeROI,
            (const Npp32s**)d_pValues, (const Npp32s**)d_pLevels, nLevels
        );
        for (int c = 0; c < 3; ++c) { cudaFree(d_pValues[c]); cudaFree(d_pLevels[c]); }
        
        if (status != NPP_SUCCESS) {
            printf("nppiLUT_8u_C3R failed with status %d\n", status);
        } else {
            printf("--LUT gamma correction (RGB, gamma=%.2f) completed successfully.\n", gamma);
        }

    } else if (channels == 1) {

        int srcStep = width * sizeof(unsigned char);
        int dstStep = srcStep;
        Npp32s* pValues = (Npp32s*)malloc(256 * sizeof(Npp32s));
        Npp32s* pLevels = (Npp32s*)malloc(256 * sizeof(Npp32s));
        int nLevels = 256;
        
        for (int i = 0; i < 256; ++i) {
            float norm = ((float)i) / 255.0f;
            float gamma_safe = gamma > 0.0f ? gamma : 1.0f;
            pLevels[i] = i;
            pValues[i] = (Npp32s)(powf(norm, gamma_safe) * 255.0f + 0.5f);
            
        }
        
        NppStatus status = nppiLUT_8u_C1R(
            d_input, srcStep,
            d_output, dstStep,
            oSizeROI,
            pValues, pLevels, nLevels
        );
        
        if (status != NPP_SUCCESS) {
            printf("nppiLUT_8u_C1R failed with status %d\n", status);
        } else {
            printf("--LUT gamma correction (grayscale, gamma=%.2f) completed successfully.\n", gamma);
        }
        free(pValues);
        free(pLevels);
    } else {
        printf("lut only supports 1-channel grayscale or 3-channel RGB images.\n");
    }

}

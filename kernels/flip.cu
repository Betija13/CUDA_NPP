#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cstdio>

extern "C" void flip_horizontal_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    NppiSize oSizeROI = { width, height };
    int srcStep = width * channels;
    int dstStep = width * channels;

    NppStatus status;
    if (channels == 1) {
        status = nppiMirror_8u_C1R(d_input, srcStep, d_output, dstStep, oSizeROI, NPP_HORIZONTAL_AXIS);
    } else if (channels == 3) {
        status = nppiMirror_8u_C3R(d_input, srcStep, d_output, dstStep, oSizeROI, NPP_HORIZONTAL_AXIS);
    } else {
        printf("NPP flip only supports NPPI_AXIS_HORIZONTAL or 3 channel images.\n");
        return;
    }

    if (status != NPP_SUCCESS) {
        printf("NPP flip failed with status %d\n", status);
    }
}

extern "C" void flip_vertical_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    NppiSize oSizeROI = { width, height };
    int srcStep = width * channels;
    int dstStep = width * channels;

    NppStatus status;
    if (channels == 1) {
        status = nppiMirror_8u_C1R(d_input, srcStep, d_output, dstStep, oSizeROI, NPP_VERTICAL_AXIS);
    } else if (channels == 3) {
        status = nppiMirror_8u_C3R(d_input, srcStep, d_output, dstStep, oSizeROI, NPP_VERTICAL_AXIS);
    } else {
        printf("NPP flip only supports NPPI_AXIS_HORIZONTAL or 3 channel images.\n");
        return;
    }

    if (status != NPP_SUCCESS) {
        printf("NPP flip failed with status %d\n", status);
    }
}

extern "C" void flip_both_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    NppiSize oSizeROI = { width, height };
    int srcStep = width * channels;
    int dstStep = width * channels;

    NppStatus status;
    if (channels == 1) {
        status = nppiMirror_8u_C1R(d_input, srcStep, d_output, dstStep, oSizeROI, NPP_BOTH_AXIS);
    } else if (channels == 3) {
        status = nppiMirror_8u_C3R(d_input, srcStep, d_output, dstStep, oSizeROI, NPP_BOTH_AXIS);
    } else {
        printf("NPP flip only supports NPPI_AXIS_HORIZONTAL or 3 channel images.\n");
        return;
    }

    if (status != NPP_SUCCESS) {
        printf("NPP flip failed with status %d\n", status);
    }
}

#include <npp.h>
#include <nppi.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" void rotation(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float angle_degrees) {
    
    cudaFree(0);
    
    // Convert angle from degrees to radians
    float angle_radians = angle_degrees * (M_PI / 180.0f);
    
    // Define the size of the region of interest (ROI)
    NppiSize oSizeROI = { width, height };
    
    // Define step sizes for input and output images
    int srcStep = width * channels;
    int dstStep = srcStep;
    
    // Status variable for NPP operations
    NppStatus status;
    
    // Calculate the center of the image
    double xCenter = width / 2.0;
    double yCenter = height / 2.0;
    
    // Create the rotation transform
    double coeffs[2][3];
    // Rotation matrix
    coeffs[0][0] = cos(angle_radians);
    coeffs[0][1] = sin(angle_radians);
    coeffs[0][2] = xCenter - xCenter * cos(angle_radians) - yCenter * sin(angle_radians);
    coeffs[1][0] = -sin(angle_radians);
    coeffs[1][1] = cos(angle_radians);
    coeffs[1][2] = yCenter + xCenter * sin(angle_radians) - yCenter * cos(angle_radians);
    
    printf("--Applying image rotation with angle: %.2f degrees\n", angle_degrees);
    
    if (channels == 1) {
        // Grayscale rotation
        status = nppiWarpAffine_8u_C1R(
            d_input, oSizeROI, srcStep, 
            NppiRect{0, 0, width, height},  // Source ROI
            d_output, dstStep, 
            NppiRect{0, 0, width, height},  // Destination ROI
            coeffs, 
            NPPI_INTER_LINEAR  // Linear interpolation
        );
        
        if (status != NPP_SUCCESS) {
            printf("nppiWarpAffine_8u_C1R failed with status %d\n", status);
            return;
        }
    }
    else if (channels == 3) {
        // RGB rotation
        status = nppiWarpAffine_8u_C3R(
            d_input, oSizeROI, srcStep, 
            NppiRect{0, 0, width, height},  // Source ROI
            d_output, dstStep, 
            NppiRect{0, 0, width, height},  // Destination ROI
            coeffs, 
            NPPI_INTER_LINEAR  // Linear interpolation
        );
        
        if (status != NPP_SUCCESS) {
            printf("nppiWarpAffine_8u_C3R failed with status %d\n", status);
            return;
        }
    }
    else {
        printf("Error: Rotation only supports 1 or 3 channel images.\n");
        return;
    }
    
    printf("--Image rotation completed successfully.\n");
}

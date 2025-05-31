#include <npp.h>
#include <nppi.h>
#include <cstdio>
#include <cuda_runtime.h>

extern "C" void threshold(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, unsigned char thresh) {
    // Ensure CUDA context is initialized
    cudaFree(0);
    
    // Define the size of the region of interest (ROI)
    NppiSize oSizeROI = { width, height };
    
    // Define step sizes for input and output images
    int srcStep = width * channels;
    int dstStep = srcStep;
    
    // Status variable for NPP operations
    NppStatus status;
    
    if (channels == 1) {
        // Grayscale thresholding
        printf("--Applying grayscale thresholding with threshold value: %d\n", thresh);
        
        // Create a temporary buffer for the threshold operation
        unsigned char* d_temp = nullptr;
        cudaError_t err = cudaMalloc(&d_temp, width * height * sizeof(unsigned char));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for temporary buffer: %s\n", cudaGetErrorString(err));
            return;
        }

        status = nppiThreshold_GTVal_8u_C1R(
            d_input, srcStep,
            d_output, dstStep,
            oSizeROI,
            thresh,     // Threshold value
            0           // Value for pixels below threshold
        );
        
        if (status != NPP_SUCCESS) {
            printf("nppiThreshold_GTVal_8u_C1R failed with status %d\n", status);
            cudaFree(d_temp);
            return;
        }
        
        // Free temporary buffer
        cudaFree(d_temp);
    }
    else if (channels == 3) {
        // RGB thresholding
        printf("--Applying RGB thresholding with threshold value: %d\n", thresh);
        
        // Create a temporary buffer for the threshold operation
        unsigned char* d_temp = nullptr;
        cudaError_t err = cudaMalloc(&d_temp, width * height * 3 * sizeof(unsigned char));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for temporary buffer: %s\n", cudaGetErrorString(err));
            return;
        }
        
        
        // Create threshold array for RGB
        Npp8u threshArr[3] = { thresh, thresh, thresh };
        
        // Set all pixels in temp buffer to 255 (white) where input pixels >= threshold
        status = nppiThreshold_GTVal_8u_C3R(
            d_input, srcStep,
            d_temp, srcStep, 
            oSizeROI,
            threshArr,       // Threshold value for each channel
            threshArr        // Output value (255) for pixels above threshold
        );
        
        if (status != NPP_SUCCESS) {
            printf("nppiThreshold_GTVal_8u_C3R failed with status %d\n", status);
            cudaFree(d_temp);
            return;
        }
        
        // Set all pixels in temp buffer to 0 (black) where input pixels < threshold
        Npp8u zeroArr[3] = { 0, 0, 0 };
        status = nppiThreshold_LTVal_8u_C3R(
            d_input, srcStep,
            d_temp, srcStep,
            oSizeROI,
            threshArr,       // Threshold value for each channel
            zeroArr          // Output value (0) for pixels below threshold
        );
        
        if (status != NPP_SUCCESS) {
            printf("nppiThreshold_LTVal_8u_C3R failed with status %d\n", status);
            cudaFree(d_temp);
            return;
        }
        
        // Copy the result to output
        status = nppiCopy_8u_C3R(
            d_temp, srcStep,
            d_output, dstStep,
            oSizeROI
        );
        
        if (status != NPP_SUCCESS) {
            printf("nppiAnd_8u_C3IR failed with status %d\n", status);
            cudaFree(d_temp);
            return;
        }
        
        // Free temporary buffer
        cudaFree(d_temp);
    }
    else {
        printf("Error: Thresholding only supports 1 or 3 channel images.\n");
        return;
    }
    
    printf("--Thresholding completed successfully.\n");
}
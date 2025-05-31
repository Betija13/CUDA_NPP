#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cstdio>

extern "C" void statistics(const unsigned char* d_input, int width, int height, int channels) {
    // Basic image info
    NppiSize oSizeROI = { width, height };
    printf("-- Width: %d, Height: %d, Channels: %d\n", width, height, channels);
    int srcStep = width * channels;
    
    if (channels == 1) {
        // ----- SUM CALCULATION -----
        // Allocate device memory for sum result
        Npp64f* d_sum;
        cudaMalloc((void**)&d_sum, sizeof(Npp64f));
        cudaMemset(d_sum, 0, sizeof(Npp64f));
        
        // Get buffer size for sum scratch memory
        size_t sumBufferSize = 0;
        NppStatus status = nppiSumGetBufferHostSize_8u_C1R(oSizeROI, &sumBufferSize);
        if (status != NPP_SUCCESS) {
            printf("nppiSumGetBufferHostSize_8u_C1R failed with status %d\n", status);
            cudaFree(d_sum);
            return;
        }
        
        // Allocate sum scratch buffer
        Npp8u* pSumBuffer = NULL;
        cudaError_t err = cudaMalloc((void**)&pSumBuffer, sumBufferSize);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for sum buffer: %s\n", cudaGetErrorString(err));
            cudaFree(d_sum);
            return;
        }
        
        // Calculate sum
        status = nppiSum_8u_C1R(d_input, srcStep, oSizeROI, pSumBuffer, d_sum);
        
        // Copy sum result back to host
        Npp64f h_sum = 0.0;
        cudaMemcpy(&h_sum, d_sum, sizeof(Npp64f), cudaMemcpyDeviceToHost);
        
        // Free sum memory
        cudaFree(pSumBuffer);
        cudaFree(d_sum);
        
        if (status == NPP_SUCCESS) {
            printf("NPP Sum: %.0f\n", h_sum);
        } else {
            printf("NPP sum calculation failed with status %d\n", status);
        }
        
        // ----- MIN/MAX CALCULATION -----
        // Allocate device memory for min/max results
        Npp8u* d_min;
        Npp8u* d_max;
        cudaMalloc((void**)&d_min, sizeof(Npp8u));
        cudaMalloc((void**)&d_max, sizeof(Npp8u));
        
        // Get buffer size for min/max scratch memory
        size_t minMaxBufferSize = 0;
        status = nppiMinMaxGetBufferHostSize_8u_C1R(oSizeROI, &minMaxBufferSize);
        if (status != NPP_SUCCESS) {
            printf("nppiMinMaxGetBufferHostSize_8u_C1R failed with status %d\n", status);
            cudaFree(d_min);
            cudaFree(d_max);
            return;
        }
        
        // Allocate min/max scratch buffer
        Npp8u* pMinMaxBuffer = NULL;
        err = cudaMalloc((void**)&pMinMaxBuffer, minMaxBufferSize);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for min/max buffer: %s\n", cudaGetErrorString(err));
            cudaFree(d_min);
            cudaFree(d_max);
            return;
        }
        
        // Calculate min/max
        status = nppiMinMax_8u_C1R(d_input, srcStep, oSizeROI, d_min, d_max, pMinMaxBuffer);
        
        // Copy min/max results back to host
        Npp8u h_min = 0;
        Npp8u h_max = 0;
        cudaMemcpy(&h_min, d_min, sizeof(Npp8u), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max, d_max, sizeof(Npp8u), cudaMemcpyDeviceToHost);
        
        // Free min/max memory
        cudaFree(pMinMaxBuffer);
        cudaFree(d_min);
        cudaFree(d_max);
        
        if (status == NPP_SUCCESS) {
            printf("NPP Min: %d, Max: %d\n", h_min, h_max);
        } else {
            printf("NPP min/max calculation failed with status %d\n", status);
        }
        
        // ----- MEAN/STDDEV CALCULATION -----
        // Allocate device memory for mean and stddev results
        Npp64f* d_mean;
        Npp64f* d_stddev;
        cudaMalloc((void**)&d_mean, sizeof(Npp64f));
        cudaMalloc((void**)&d_stddev, sizeof(Npp64f));
        cudaMemset(d_mean, 0, sizeof(Npp64f));
        cudaMemset(d_stddev, 0, sizeof(Npp64f));
        
        // Get buffer size for mean/stddev scratch memory
        size_t meanStdDevBufferSize = 0;
        status = nppiMeanStdDevGetBufferHostSize_8u_C1R(oSizeROI, &meanStdDevBufferSize);
        if (status != NPP_SUCCESS) {
            printf("nppiMeanStdDevGetBufferHostSize_8u_C1R failed with status %d\n", status);
            cudaFree(d_mean);
            cudaFree(d_stddev);
            return;
        }
        
        // Allocate mean/stddev scratch buffer
        Npp8u* pMeanStdDevBuffer = NULL;
        err = cudaMalloc((void**)&pMeanStdDevBuffer, meanStdDevBufferSize);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for mean/stddev buffer: %s\n", cudaGetErrorString(err));
            cudaFree(d_mean);
            cudaFree(d_stddev);
            return;
        }
        
        // Calculate mean/stddev
        status = nppiMean_StdDev_8u_C1R(d_input, srcStep, oSizeROI, pMeanStdDevBuffer, d_mean, d_stddev);
        
        // Copy mean/stddev results back to host
        Npp64f h_mean = 0.0;
        Npp64f h_stddev = 0.0;
        cudaMemcpy(&h_mean, d_mean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_stddev, d_stddev, sizeof(Npp64f), cudaMemcpyDeviceToHost);
        
        // Free mean/stddev memory
        cudaFree(pMeanStdDevBuffer);
        cudaFree(d_mean);
        cudaFree(d_stddev);
        
        if (status == NPP_SUCCESS) {
            printf("NPP Mean: %.2f, StdDev: %.2f\n", h_mean, h_stddev);
        } else {
            printf("NPP mean/stddev calculation failed with status %d\n", status);
        }
    } else if (channels == 3) {
        // ----- SUM CALCULATION -----
        // Allocate device memory for sum results (3 channels)
        Npp64f* d_sum;
        cudaMalloc((void**)&d_sum, 3 * sizeof(Npp64f));
        cudaMemset(d_sum, 0, 3 * sizeof(Npp64f));
        
        // Get buffer size for sum scratch memory
        size_t sumBufferSize = 0;
        NppStatus status = nppiSumGetBufferHostSize_8u_C3R(oSizeROI, &sumBufferSize);
        if (status != NPP_SUCCESS) {
            printf("nppiSumGetBufferHostSize_8u_C3R failed with status %d\n", status);
            cudaFree(d_sum);
            return;
        }
        
        // Allocate sum scratch buffer
        Npp8u* pSumBuffer = NULL;
        cudaError_t err = cudaMalloc((void**)&pSumBuffer, sumBufferSize);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for sum buffer: %s\n", cudaGetErrorString(err));
            cudaFree(d_sum);
            return;
        }
        
        // Calculate sum
        status = nppiSum_8u_C3R(d_input, srcStep, oSizeROI, pSumBuffer, d_sum);
        
        // Copy sum result back to host
        Npp64f h_sum[3] = {0.0, 0.0, 0.0};
        cudaMemcpy(h_sum, d_sum, 3 * sizeof(Npp64f), cudaMemcpyDeviceToHost);
        
        // Free sum memory
        cudaFree(pSumBuffer);
        cudaFree(d_sum);
        
        if (status == NPP_SUCCESS) {
            printf("NPP Sum: R=%.0f G=%.0f B=%.0f\n", h_sum[0], h_sum[1], h_sum[2]);
        } else {
            printf("NPP sum calculation failed with status %d\n", status);
        }
        
        // ----- MIN/MAX CALCULATION -----
        // Allocate device memory for min/max results (3 channels)
        Npp8u* d_min;
        Npp8u* d_max;
        cudaMalloc((void**)&d_min, 3 * sizeof(Npp8u));
        cudaMalloc((void**)&d_max, 3 * sizeof(Npp8u));
        
        // Get buffer size for min/max scratch memory
        size_t minMaxBufferSize = 0;
        status = nppiMinMaxGetBufferHostSize_8u_C3R(oSizeROI, &minMaxBufferSize);
        if (status != NPP_SUCCESS) {
            printf("nppiMinMaxGetBufferHostSize_8u_C3R failed with status %d\n", status);
            cudaFree(d_min);
            cudaFree(d_max);
            return;
        }
        
        // Allocate min/max scratch buffer
        Npp8u* pMinMaxBuffer = NULL;
        err = cudaMalloc((void**)&pMinMaxBuffer, minMaxBufferSize);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for min/max buffer: %s\n", cudaGetErrorString(err));
            cudaFree(d_min);
            cudaFree(d_max);
            return;
        }
        
        // Calculate min/max
        status = nppiMinMax_8u_C3R(d_input, srcStep, oSizeROI, d_min, d_max, pMinMaxBuffer);
        
        // Copy min/max results back to host
        Npp8u h_min[3] = {0, 0, 0};
        Npp8u h_max[3] = {0, 0, 0};
        cudaMemcpy(h_min, d_min, 3 * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_max, d_max, 3 * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        
        // Free min/max memory
        cudaFree(pMinMaxBuffer);
        cudaFree(d_min);
        cudaFree(d_max);
        
        if (status == NPP_SUCCESS) {
            printf("NPP Min: R=%d G=%d B=%d, Max: R=%d G=%d B=%d\n", 
                   h_min[0], h_min[1], h_min[2], 
                   h_max[0], h_max[1], h_max[2]);
        } else {
            printf("NPP min/max calculation failed with status %d\n", status);
        }
        
        // ----- MEAN/STDDEV CALCULATION -----
        // For RGB, need to calculate mean/stddev for each channel separately
        
        // Allocate device memory for mean and stddev results
        Npp64f* d_mean;
        Npp64f* d_stddev;
        cudaMalloc((void**)&d_mean, sizeof(Npp64f));
        cudaMalloc((void**)&d_stddev, sizeof(Npp64f));
        
        // Get buffer size for mean/stddev scratch memory
        size_t meanStdDevBufferSize = 0;
        status = nppiMeanStdDevGetBufferHostSize_8u_C3CR(oSizeROI, &meanStdDevBufferSize);
        if (status != NPP_SUCCESS) {
            printf("nppiMeanStdDevGetBufferHostSize_8u_C3CR failed with status %d\n", status);
            cudaFree(d_mean);
            cudaFree(d_stddev);
            return;
        }
        
        // Allocate mean/stddev scratch buffer
        Npp8u* pMeanStdDevBuffer = NULL;
        err = cudaMalloc((void**)&pMeanStdDevBuffer, meanStdDevBufferSize);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for mean/stddev buffer: %s\n", cudaGetErrorString(err));
            cudaFree(d_mean);
            cudaFree(d_stddev);
            return;
        }
        
        // Results for each channel
        Npp64f h_mean[3] = {0.0, 0.0, 0.0};
        Npp64f h_stddev[3] = {0.0, 0.0, 0.0};
        
        // Calculate mean/stddev for each channel (R=0, G=1, B=2)
        bool success = true;
        for (int channel = 0; channel < 3; channel++) {
            cudaMemset(d_mean, 0, sizeof(Npp64f));
            cudaMemset(d_stddev, 0, sizeof(Npp64f));
            
            status = nppiMean_StdDev_8u_C3CR(d_input, srcStep, oSizeROI, channel + 1, 
                                             pMeanStdDevBuffer, d_mean, d_stddev);
            
            if (status != NPP_SUCCESS) {
                printf("nppiMean_StdDev_8u_C3CR failed for channel %d with status %d\n", 
                       channel, status);
                success = false;
                break;
            }
            
            // Copy results back to host
            cudaMemcpy(&h_mean[channel], d_mean, sizeof(Npp64f), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_stddev[channel], d_stddev, sizeof(Npp64f), cudaMemcpyDeviceToHost);
        }
        
        // Free mean/stddev memory
        cudaFree(pMeanStdDevBuffer);
        cudaFree(d_mean);
        cudaFree(d_stddev);
        
        if (success) {
            printf("NPP Mean: R=%.2f G=%.2f B=%.2f, StdDev: R=%.2f G=%.2f B=%.2f\n", 
                   h_mean[0], h_mean[1], h_mean[2], 
                   h_stddev[0], h_stddev[1], h_stddev[2]);
        }
    } else {
        printf("Only supports 1 or 3 channel images.\n");
    }
}
// main.cu
// Entry point for CUDA NPP Image Processing Library
// Will load images, call kernels, and save output images.

#include <stdio.h>
#include <string.h>
#include "include/image_utils.h"

extern "C" void rgb_to_grayscale_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
extern "C" void flip_horizontal_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
extern "C" void flip_vertical_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
extern "C" void flip_both_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
extern "C" void gaussian_blur_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, int kernelSizeInt);
extern "C" void sobel_edge_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
extern "C" void statistics(const unsigned char* d_input, int width, int height, int channels);
extern "C" void threshold(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, unsigned char thresh);
extern "C" void threshold_r(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, unsigned char thresh);
extern "C" void rotation(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float angle_degrees);
extern "C" void g_corr(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
extern "C" void lut(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float gamma);
extern "C" void brightness_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, int brightness_delta);
extern "C" void contrast_npp(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float contrast_factor);

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <file_name> <task> [number]\n", argv[0]);
        printf("       <task>: grayscale | flip_horizontal | flip_vertical | flip_both | g_blur | sobel | threshold | threshold_r | rotation | brightness | contrast | statistics\n");
        return 1;
    }
    const char* file_name = argv[1];
    const char* task = argv[2];
    printf("--- Task: %s\n", task);
    
    // printf("Success!\n");
    // Shared image loading
    int width, height, channels;
    unsigned char* h_input = load_image(file_name, &width, &height, &channels);
    if (!h_input) {
        printf("Failed to load input image from args.\n");
        h_input = load_image("test_images/input.png", &width, &height, &channels);
        if (!h_input) {
            printf("Failed to load input image from test_images/input.png.\n");
            return 1;
        } else {
            printf("Using test_images/input.png as input image.\n");
        }
    }
    size_t num_pixels = width * height;

    // Operation-specific variables
    size_t d_input_bytes = 0, d_output_bytes = 0;
    bool valid_task = true;
    char output_filename[256] = "";
    int output_channels = channels;

    // CUDA timing events
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate device memory
    unsigned char *d_input = nullptr, *d_output = nullptr;
    unsigned char* h_output = nullptr;
    
    if (strcmp(task, "grayscale") == 0) {
        d_input_bytes = num_pixels * channels * sizeof(unsigned char);
        d_output_bytes = num_pixels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        // Call NPP grayscale conversion
        rgb_to_grayscale_npp(d_input, d_output, width, height, channels);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_grayscale.png");
        output_channels = 1;
    } else if (strcmp(task, "flip_horizontal") == 0 || strcmp(task, "flip_vertical") == 0 || strcmp(task, "flip_both") == 0) {
        d_input_bytes = d_output_bytes = num_pixels * channels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        // Call NPP flip
        if (strcmp(task, "flip_horizontal") == 0) {
            flip_horizontal_npp(d_input, d_output, width, height, channels);
        } else if (strcmp(task, "flip_vertical") == 0) {
            flip_vertical_npp(d_input, d_output, width, height, channels);
        } else if (strcmp(task, "flip_both") == 0) {
            flip_both_npp(d_input, d_output, width, height, channels);
        }
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        if (strcmp(task, "flip_horizontal") == 0) {
            strcpy(output_filename, "test_images/output_flip_h.png");
        } else if (strcmp(task, "flip_vertical") == 0) {
            strcpy(output_filename, "test_images/output_flip_v.png");
        } else if (strcmp(task, "flip_both") == 0) {
            strcpy(output_filename, "test_images/output_flip_b.png");
        }
        output_channels = channels;
    } else if (strcmp(task, "sobel") == 0) {
        d_input_bytes = num_pixels * channels * sizeof(unsigned char);
        d_output_bytes = num_pixels * sizeof(unsigned char); // Always 1 channel output for Sobel
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        sobel_edge_npp(d_input, d_output, width, height, channels);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_sobel.png");
        output_channels = 1;
    } else if (strcmp(task, "threshold") == 0) {
        unsigned char thresholdValue = (argc > 3) ? (unsigned char)atoi(argv[3]) : 128;
        d_input_bytes = d_output_bytes = num_pixels * channels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        threshold(d_input, d_output, width, height, channels, thresholdValue);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_threshold.png");
        output_channels = channels;
    } else if (strcmp(task, "threshold_r") == 0) {
        unsigned char thresholdValue = (argc > 3) ? (unsigned char)atoi(argv[3]) : 128;
        d_input_bytes = d_output_bytes = num_pixels * channels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        threshold_r(d_input, d_output, width, height, channels, thresholdValue);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_threshold_r.png");
        output_channels = channels;
    } else if (strcmp(task, "brightness") == 0) {
        // 0 to 255
        int brightness_delta = (argc > 3) ? atoi(argv[3]) : 100;
        printf("--Brightness delta: %d\n", brightness_delta);
        d_input_bytes = d_output_bytes = num_pixels * channels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        brightness_npp(d_input, d_output, width, height, channels, brightness_delta);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_brightness.png");
        output_channels = channels;
    } else if (strcmp(task, "contrast") == 0) {
        //  0.0 to 3.0 (but can go higher)
        //  > 1.0 increase contrast
        //  < 1.0 decrease contrast
        //  Value of 1.0 leaves contrast unchanged
        float contrast_factor = (argc > 3) ? (float)atof(argv[3]) : 2.5f;
        d_input_bytes = d_output_bytes = num_pixels * channels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        contrast_npp(d_input, d_output, width, height, channels, contrast_factor);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_contrast.png");
        output_channels = channels;
    } else if (strcmp(task, "rotation") == 0) {
        // 0 to 360
        float angle_degrees = (argc > 3) ? (float)atof(argv[3]) : 45.0f;
        d_input_bytes = d_output_bytes = num_pixels * channels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        rotation(d_input, d_output, width, height, channels, angle_degrees);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_rotation.png");
        output_channels = channels;
    } else if (strcmp(task, "g_corr") == 0) {
        d_input_bytes = num_pixels * channels * sizeof(unsigned char);
        d_output_bytes = d_input_bytes;
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        g_corr(d_input, d_output, width, height, channels);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_g_corr.png");
        output_channels = channels;
    } else if (strcmp(task, "lut") == 0) {
        // 0.1 to 5.0
        // Values < 1.0 brighten mid-tones (make image appear brighter)
        // Values > 1.0 darken mid-tones (make image appear darker)
        float gamma = 3.5f;
        
        if (argc > 3 && argv[3] != NULL) {
            
            gamma = (float)atof(argv[3]);
        }
        
        d_input_bytes = num_pixels * channels * sizeof(unsigned char);
        d_output_bytes = d_input_bytes;
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        lut(d_input, d_output, width, height, channels, gamma);
        
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        strcpy(output_filename, "test_images/output_lut.png");
        output_channels = channels;
    } else if (strcmp(task, "g_blur") == 0) {
        int kernelSizeInt = 77;
        if (argc > 3) {
            const char* kernelSizeStr = argv[3];
            printf("--Kernel size: %s\n", kernelSizeStr);
            kernelSizeInt = atoi(kernelSizeStr);
        }
        d_input_bytes = d_output_bytes = num_pixels * channels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMalloc(&d_output, d_output_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        gaussian_blur_npp(d_input, d_output, width, height, channels, kernelSizeInt);
        h_output = (unsigned char*)malloc(d_output_bytes);
        cudaMemcpy(h_output, d_output, d_output_bytes, cudaMemcpyDeviceToHost);
        sprintf(output_filename, "test_images/output_g_blur_%d.png", kernelSizeInt);
        output_channels = channels;
    } else if (strcmp(task, "statistics") == 0) {
        d_input_bytes = num_pixels * channels * sizeof(unsigned char);
        cudaMalloc(&d_input, d_input_bytes);
        cudaMemcpy(d_input, h_input, d_input_bytes, cudaMemcpyHostToDevice);
        statistics(d_input, width, height, channels);
        
    } else {
        valid_task = false;
    }

    // Stop CUDA timing and print elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("===Elapsed time: %.3f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (valid_task) {
        if (strcmp(task, "statistics") == 0) {
            printf("++Valid task: %s\n", task);
        } else {
            save_image(output_filename, h_output, width, height, output_channels);
            printf("++Saved output image as %s\n", output_filename);
        }
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
    } else {
        free(h_input);
        printf("Unknown task: %s\n", task);
        return 1;
    }
    return 0;
    
}


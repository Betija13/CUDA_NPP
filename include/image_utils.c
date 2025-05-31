// image_utils.c
// Minimal image loading/saving using stb_image and stb_image_write
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <stdio.h>

unsigned char* load_image(const char* filename, int* width, int* height, int* channels) {
    unsigned char* data = stbi_load(filename, width, height, channels, 0);
    if (!data) {
        printf("Failed to load image: %s\n", filename);
    }
    return data;
}

void save_image(const char* filename, unsigned char* data, int width, int height, int channels) {
    int success = 0;
    if (channels == 1) {
        success = stbi_write_png(filename, width, height, 1, data, width);
    } else if (channels == 3) {
        success = stbi_write_png(filename, width, height, 3, data, width * 3);
    } else {
        printf("Unsupported channel count for saving: %d\n", channels);
    }
    if (!success) {
        printf("Failed to save image: %s\n", filename);
    }
}

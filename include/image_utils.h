// image_utils.h
// Utility functions for image loading/saving (stub)
#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

unsigned char* load_image(const char* filename, int* width, int* height, int* channels);
void save_image(const char* filename, unsigned char* data, int width, int height, int channels);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_UTILS_H

#pragma once
#include <stdint.h>

//void avx2_diff_int8(int8_t* a, int8_t* b, int8_t* c, size_t size, uint32_t client_id);
//void avx2_diff_uint8(uint8_t* a, uint8_t* b, uint8_t* c, size_t size, uint32_t client_id);
void plane_blend_avx2(uint8_t* a_inout, uint8_t* b_in, uint32_t stride, uint32_t width, uint32_t height, uint8_t ka, uint8_t kb);

bool plane_diff_mask_detect(
    const uint8_t* a, const uint8_t* b, const uint8_t* m,
    uint32_t stride, uint32_t width, uint32_t height,
    double thr, double* ssd);

void plane_diff_blur_i8(int8_t* a, int8_t* b, int8_t* c, int8_t* d, uint32_t stride, uint32_t width, uint32_t height);
void plane_diff_blur_u8(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, uint32_t stride, uint32_t width, uint32_t height);

bool avx2_mask_detect_uint8(uint8_t* a, uint8_t* m, uint32_t width, uint32_t &x1, uint32_t &x2);
bool avx2_sub_mask_detect_uint8(uint8_t* a, uint8_t* b, uint8_t* m, uint8_t* out, uint32_t width, uint32_t &x1, uint32_t &x2);



void SimdGaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    size_t channelCount, uint8_t * dst, size_t dstStride);

void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
    uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

void SimdAddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
    const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
    uint16_t weight, uint8_t * difference, size_t differenceStride);

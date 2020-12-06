#include <immintrin.h>
#include <stdint.h>
#include <unordered_map>
#include <memory.h>
#include <assert.h>
#include "simd_ops.h"
extern "C" {
//#include "simd_interface.h"
}


typedef struct {
    size_t aligned_size;
    void* allocated_pointer;
    void* aligned_pointer1;
    void* aligned_pointer2;
    void* aligned_pointer3;
} AlignedPointer;

static inline int8_t i8abs(int8_t v) {
    return v < 0 ? -v : v;
}

static inline uint8_t u8diff(uint8_t a, uint8_t b) {
    return a > b ? a - b : b - a;
}

static inline bool is_aligned32(uint64_t value) {
    return !(value & 0x1F);
}

static inline uint64_t align32(uint64_t value) {
    return ((value - 0x1F) & 0xFFFFFFFFFFFFFFE0ull) + 0x20;
}

static inline uint32_t align32(uint32_t value) {
    return ((value - 0x1F) & 0xFFFFFFE0) + 0x20;
}


/*
static inline AlignedPointer* aligned_pointer_create(uint32_t aligned_size) {
    AlignedPointer* ap = new AlignedPointer;
    ap->allocated_pointer = malloc((size_t)aligned_size + 0x20);
    ap->aligned_pointer = (void*)(align32((uint64_t)ap->allocated_pointer));
    ap->aligned_size = aligned_size;
    return ap;
}


static inline void aligned_pointer_release(AlignedPointer* ap) {
    if (ap) {
        if (ap->allocated_pointer) {
            free(ap->allocated_pointer);
            ap->allocated_pointer = NULL;
            ap->aligned_pointer = NULL;
            ap->aligned_size = 0;
        }
    }
}
*/

class CAlignedPointer {
public:
    CAlignedPointer() {
        aligned.allocated_pointer = NULL;
        aligned.aligned_pointer1 = NULL;
        aligned.aligned_pointer2 = NULL;
        aligned.aligned_pointer3 = NULL;
        aligned.aligned_size = 0;
    }
    ~CAlignedPointer() {
        free(aligned.allocated_pointer);
        aligned.allocated_pointer = NULL;
        aligned.aligned_pointer1 = NULL;
        aligned.aligned_pointer2 = NULL;
        aligned.aligned_pointer3 = NULL;
        aligned.aligned_size = 0;
    }

    AlignedPointer* get(size_t aligned_size) {
        if (aligned.aligned_size != aligned_size) {
            if (aligned.allocated_pointer) {
                free(aligned.allocated_pointer);
            }
            // Allocate double aligned size with safe gaps
            aligned.allocated_pointer = malloc((aligned_size + 0x20) * 3);
            aligned.aligned_pointer1 = (void*)(align32((uint64_t)aligned.allocated_pointer));
            aligned.aligned_pointer2 = (void*)((uint64_t)aligned.aligned_pointer1 + aligned_size);
            aligned.aligned_pointer3 = (void*)((uint64_t)aligned.aligned_pointer2 + aligned_size);
            aligned.aligned_size = aligned_size;
        }
        return &aligned;
    }
    AlignedPointer aligned;
};


// The objects stored here are being removed automatically?
std::unordered_map <uint32_t, CAlignedPointer> AlignedPointers;


AlignedPointer* aligned_pointer_get(uint32_t client_id, size_t size) {
    size_t aligned_size = align32(size);
    return AlignedPointers[client_id].get(aligned_size);
}


void avx2_diff_int8(int8_t* a, int8_t* b, int8_t* c, size_t size, uint32_t client_id) {
    int8_t* p1 = NULL;
    int8_t* p2 = NULL;
    int8_t* p3 = NULL;
    int8_t* output = NULL;
    AlignedPointer* ap = NULL;
    // Check alignment
    bool p1_aligned = is_aligned32((uint64_t)a);
    bool p2_aligned = is_aligned32((uint64_t)b);
    bool p3_aligned = is_aligned32((uint64_t)c);
    // Get/create aligned buffers if needed
    if (!(p1_aligned && p2_aligned && p3_aligned)) {
        ap = aligned_pointer_get(client_id, size);
    }
    if (p1_aligned) {
        p1 = a;
    } else {
        p1 = (int8_t*)ap->aligned_pointer1;
        memcpy(p1, a, size);
    }
    if (p2_aligned) {
        p2 = b;
    } else {
        p2 = (int8_t*)ap->aligned_pointer2;
        memcpy(p2, b, size);
    }
    if (p3_aligned) {
        p3 = c;
    } else {
        p3 = (int8_t*)ap->aligned_pointer3;
    }

    // Run conversion
    output = p3;
    for (; size >= 0x20; size -= 0x20, p1 += 0x20, p2 += 0x20, p3 += 0x20) {
        __m256i m1 = _mm256_load_si256((const __m256i*)p1);
        __m256i m2 = _mm256_load_si256((const __m256i*)p2);
        __m256i m0 = _mm256_abs_epi8(_mm256_sub_epi8(m1, m2));
        _mm256_store_si256((__m256i*)p3, m0);
    }
    for (; size; --size, p1++, p2++, p3++) {
        *p3 = i8abs(*p2 - *p1);
    }
    // Copy back result if needed
    if (!p3_aligned) {
        memcpy(c, output, size);
    }
}

void inline avx2_diff_uint8(uint8_t* a, uint8_t* b, uint8_t* c, uint32_t size, uint32_t client_id) {
    uint8_t* p1 = a;
    uint8_t* p2 = b;
    uint8_t* p3 = c;
    uint8_t* output;
    AlignedPointer* ap = NULL;
    __m256i m0, m1, m2;
    // Check alignment
    /*bool p1_aligned = is_aligned32((uint64_t)a);
    bool p2_aligned = is_aligned32((uint64_t)b);
    bool p3_aligned = is_aligned32((uint64_t)c);
    // Get/create aligned buffers if needed
    if (!(p1_aligned && p2_aligned)) {
        ap = aligned_pointer_get(client_id, size);
    }
    if (p1_aligned) {
        p1 = a;
    } else {
        p1 = (uint8_t*)ap->aligned_pointer1;
        memcpy(p1, a, size);
    }
    if (p2_aligned) {
        p2 = b;
    } else {
        p2 = (uint8_t*)ap->aligned_pointer2;
        memcpy(p2, b, size);
    }
    if (p3_aligned) {
        p3 = c;
    } else {
        p3 = (uint8_t*)ap->aligned_pointer3;
    }*/

    // Run conversion
    output = p3;
    for (; size >= 0x20; size -= 0x20, p1 += 0x20, p2 += 0x20, p3 += 0x20) {
        m1 = _mm256_load_si256((const __m256i*)p1);
        m2 = _mm256_load_si256((const __m256i*)p2);
        m0 = _mm256_max_epu8(_mm256_subs_epu8(m1, m2), _mm256_subs_epu8(m2, m1));
        _mm256_store_si256((__m256i*)p3, m0);
    }
    /*for (; size; --size, p1++, p2++, p3++) {
        *p3 = u8diff(*p1, *p2);
    }
    // Copy back result if needed
    if (!p3_aligned) {
        memcpy(c, output, size);
    }*/
}


__declspec(align(32))
static uint8_t M256I_LIMIT[32] = {
    0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 
    0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 
    0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 
    0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F
};


uint32_t inline nonzero_bits_left(uint32_t val) {
    register uint32_t m = 1, v = val, x = 0;
    while (!(v & m)) {
        v >>= 1;
        x++;
    }
    return x;
}

uint32_t inline nonzero_bits_right(uint32_t val) {
    register uint32_t m = 0x80000000, v = val, x = 0x1F;
    v = val;
    while (!(v & m)) {
        v <<= 1;
        x--;
    }
    return x;
}


// Masked detect crop
bool avx2_mask_detect_uint8(
    uint8_t* a, uint8_t* m, uint32_t width,
    uint32_t &x1, uint32_t &x2) {
    bool result1 = false, result2 = false;
    __m256i m0, m1, limit, r;
    limit = _mm256_load_si256((const __m256i*)M256I_LIMIT);
    uint32_t cx = 0, size = width & 0x7FFFFFE0, xx, bits;
    for (;;) {
        //m0 = _mm256_load_si256((const __m256i*)a);
        //m1 = _mm256_load_si256((const __m256i*)b);
        //r = _mm256_max_epu8(_mm256_subs_epu8(m0, m1), _mm256_subs_epu8(m1, m0));
        r = _mm256_load_si256((const __m256i*)a);
        //_mm256_store_si256((__m256i*)out, r);
        r = _mm256_min_epu8(limit, r);
        m0 = _mm256_load_si256((const __m256i*)m);
        m1 = _mm256_cmpgt_epi8(r, m0);
        //_mm256_store_si256((__m256i*)out, m1);
        bits = _mm256_movemask_epi8(m1);
        if (bits) {
            result1 = true;
            xx = cx + nonzero_bits_left(bits);
            if (xx < x1) {
                x1 = xx;
            }
            xx = cx + nonzero_bits_right(bits);
            if (xx > x2) {
                x2 = xx;
            }
        }
        size -= 0x20;
        cx += 0x20; a += 0x20; m += 0x20;
        if (size < 0x20 /*|| cx >= x1*/ || result1) {
            break;
        }
    }
    // Backward
    a += size;
    m += size;
    cx += size;
    for (;;) {
        cx -= 0x20; a -= 0x20; m -= 0x20;
        if (size < 0x20 /*|| cx < x2*/ || result2) {
            break;
        }
        size -= 0x20;
        //m0 = _mm256_load_si256((const __m256i*)a);
        //m1 = _mm256_load_si256((const __m256i*)b);
        //r = _mm256_min_epu8(limit, _mm256_max_epu8(_mm256_subs_epu8(m0, m1), _mm256_subs_epu8(m1, m0)));
        r = _mm256_load_si256((const __m256i*)a);
        r = _mm256_min_epu8(limit, r);
        m0 = _mm256_load_si256((const __m256i*)m);
        m1 = _mm256_cmpgt_epi8(r, m0);
        bits = _mm256_movemask_epi8(m1);
        if (bits) {
            result2 = true;
            xx = cx + nonzero_bits_right(bits);
            if (xx > x2) {
                x2 = xx;
            }
        }
    }
    return result1 || result2;
    /*for (; size; --size, p1++, p2++, p3++) {
    *p3 = u8diff(*p1, *p2);
    }
    // Copy back result if needed
    if (!p3_aligned) {
    memcpy(c, output, size);
    }*/
}

// Masked sub & detect crop
bool avx2_sub_mask_detect_uint8(
    uint8_t* a, uint8_t* b, uint8_t* m, uint8_t* out, uint32_t width,
    uint32_t &x1, uint32_t &x2) {
    bool result1 = false, result2 = false;
    __m256i m0, m1, limit, r;
    limit = _mm256_load_si256((const __m256i*)M256I_LIMIT);
    // Check alignment
    /*bool p1_aligned = is_aligned32((uint64_t)a);
    bool p2_aligned = is_aligned32((uint64_t)b);
    bool p3_aligned = is_aligned32((uint64_t)c);
    // Get/create aligned buffers if needed
    if (!(p1_aligned && p2_aligned)) {
    ap = aligned_pointer_get(client_id, size);
    }
    if (p1_aligned) {
    p1 = a;
    } else {
    p1 = (uint8_t*)ap->aligned_pointer1;
    memcpy(p1, a, size);
    }
    if (p2_aligned) {
    p2 = b;
    } else {
    p2 = (uint8_t*)ap->aligned_pointer2;
    memcpy(p2, b, size);
    }
    if (p3_aligned) {
    p3 = c;
    } else {
    p3 = (uint8_t*)ap->aligned_pointer3;
    }*/

    // Run conversion

    // Go forward
    //uint8_t* aa = a;
    //uint8_t* bb = b;
    //uint8_t* mm = m;
    uint32_t cx = 0, size = width & 0x7FFFFFE0, xx, bits;
    for (;;) {
        m0 = _mm256_load_si256((const __m256i*)a);
        m1 = _mm256_load_si256((const __m256i*)b);
        r = _mm256_max_epu8(_mm256_subs_epu8(m0, m1), _mm256_subs_epu8(m1, m0));
        _mm256_store_si256((__m256i*)out, r);
        r = _mm256_min_epu8(limit, r);
        m0 = _mm256_load_si256((const __m256i*)m);
        m1 = _mm256_cmpgt_epi8(r, m0);
        //_mm256_store_si256((__m256i*)out, m1);
        bits = _mm256_movemask_epi8(m1);
        if (bits) {
            result1 = true;
            xx = cx + nonzero_bits_left(bits);
            if (xx < x1) {
                x1 = xx;
            }
            xx = cx + nonzero_bits_right(bits);
            if (xx > x2) {
                x2 = xx;
            }
        }
        size -= 0x20;
        cx += 0x20; a += 0x20; b += 0x20; m += 0x20; out += 0x20;
        if (size < 0x20 || cx >= x1 || result1) {
            break;
        }
    }
    // Backward
    a += size;
    b += size;
    m += size;
    cx += size;
    out += size;
    for (;;) {
        cx -= 0x20; a -= 0x20; b -= 0x20; m -= 0x20; out -= 0x20;
        if (size < 0x20 || cx < x2 || result2) {
            break;
        }
        size -= 0x20;
        m0 = _mm256_load_si256((const __m256i*)a);
        m1 = _mm256_load_si256((const __m256i*)b);
        r = _mm256_min_epu8(limit, _mm256_max_epu8(_mm256_subs_epu8(m0, m1), _mm256_subs_epu8(m1, m0)));
        m0 = _mm256_load_si256((const __m256i*)m);
        m1 = _mm256_cmpgt_epi8(r, m0);
        _mm256_store_si256((__m256i*)out, m1);
        bits = _mm256_movemask_epi8(m1);
        if (bits) {
            result2 = true;
            xx = cx + nonzero_bits_right(bits);
            if (xx > x2) {
                x2 = xx;
            }
        }
    }
    return result1 || result2;
    /*for (; size; --size, p1++, p2++, p3++) {
    *p3 = u8diff(*p1, *p2);
    }
    // Copy back result if needed
    if (!p3_aligned) {
    memcpy(c, output, size);
    }*/
}


__declspec(align(16))
static uint8_t blend_a[16];
__declspec(align(16))
static uint8_t blend_b[16];
__declspec(align(32))
static uint8_t MASK[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
void plane_blend_avx2(uint8_t* a_in, uint8_t* b_in, uint32_t stride, uint32_t width, uint32_t height, uint8_t ka, uint8_t kb) {
    memset(blend_a, ka, 16);
    memset(blend_b, kb, 16);
    __m256i mka, mkb, m2, m3;
    __m256i m0, m1;
    __m256i mask = _mm256_load_si256((const __m256i*)MASK);
    __m128i r;
    mka = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)blend_a));
    mkb = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)blend_b));
    
    for (uint32_t y = 0; y < height; ++y, a_in += stride, b_in += stride) {
        uint8_t* a = a_in;
        uint8_t* b = b_in;
        uint32_t size = width & 0x7FFFFFE0;
        for (; size >= 0x10; size -= 0x10, a += 0x10, b += 0x10) {
            m0 = _mm256_mullo_epi16(mka, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)a)));
            m1 = _mm256_mullo_epi16(mkb, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)b)));
            //m2 = _mm256_mullo_epi16(m0, mka);
            //m3 = _mm256_mullo_epi16(m1, mkb);
            m2 = _mm256_adds_epu16(m0, m1);

            // Correct by fractional
            m1 = _mm256_min_epu8(_mm256_slli_epi16(_mm256_cmpgt_epi8(mask, m2), 1), mask);
            m2 = _mm256_adds_epu8(m1, m2);

            m3 = _mm256_srai_epi16(m2, 8);
            m0 = _mm256_packs_epi16(m3, m3);
            m0 = _mm256_permute4x64_epi64(m0, 0 + (2 << 2) + (1 << 4) + (3 << 6));
            r = _mm256_extractf128_si256(m0, 0);
            _mm_store_si128((__m128i*)a, r);
        }
    }
}

#define THREADS_COUNT 4

__declspec(align(32))
static uint32_t PDMD_RESULT[8];
// Scan blocks 32x32 pix
// Scale factor for Sum of Squares of Differences (1/(32*32))
#define SSD_SCALE 0.0009765625
bool plane_diff_mask_detect(
    const uint8_t* a, const uint8_t* b, const uint8_t* m,
    uint32_t stride, uint32_t width, uint32_t height,
    double thr, double* ssd) {
    __m256i limit = _mm256_load_si256((const __m256i*)M256I_LIMIT);
    uint64_t _sum = 0;
    uint32_t _count = 0;
    uint32_t stride32 = stride << 5;
    for (; height >= 0x20; height -= 0x20, a += stride32, b += stride32, m += stride32) {
        const uint8_t* aa = a;
        const uint8_t* bb = b;
        const uint8_t* mm = m;
        for (uint32_t x = 0; x < width; x += 0x20, aa += 0x20, bb += 0x20, mm += 0x20) {
            const uint8_t* aaa = aa;
            const uint8_t* bbb = bb;
            const uint8_t* mmm = mm;
            uint64_t local_sum = 0;
            for (uint32_t h = 0; h < 0x20; ++h, aaa += stride, bbb += stride, mmm += stride) {
                __m256i m1 = _mm256_load_si256((const __m256i*)aaa);
                __m256i m2 = _mm256_load_si256((const __m256i*)bbb);
                __m256i m3 = _mm256_load_si256((const __m256i*)mmm);
                __m256i r1;
                // Masked absolute differences
                r1 = _mm256_max_epu8(_mm256_subs_epu8(m1, m2), _mm256_subs_epu8(m2, m1));
                r1 = _mm256_min_epu8(limit, r1);
                r1 = _mm256_subs_epu8(r1, m3);
                // Convert 1 vector of 32 8-bit values to 2 vectors of 16 16-bit values
                m1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r1));
                m2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(_mm256_permute2x128_si256(r1, r1, 1)));
                // Calclate squares
                m1 = _mm256_mullo_epi16(m1, m1);    // [0] [1] [2]...
                m2 = _mm256_mullo_epi16(m2, m2);    // [16][17][18]...
                // Start summing: add squares s0: [0] + [16], s1: [1] + [17] ...
                m3 = _mm256_adds_epu16(m1, m2);
                // Sum 16-bit sums s0+s8, s1+s9, ..., s7+s15
                m1 = _mm256_adds_epu16(m3, _mm256_permute2x128_si256(m3, m3, 1));
                // Convert 128-bit vector of 8 16-bit sums to 8 32-bit sums
                m3 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(m1));
                // Sum 32-bit sums ss0: s0+s4, ss1: s1+s5, ss2: s2+s6, ss3: s3+s7
                m1 = _mm256_add_epi32(m3, _mm256_permute2x128_si256(m3, m3, 1));
                // Sum 64-bit sums x0: ss0+ss2, x1: ss1+ss3
                m3 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(m1));
                m1 = _mm256_add_epi64(m3, _mm256_permute2x128_si256(m3, m3, 1));
                // Finally, x0 + x1
                m3 = _mm256_shuffle_epi32(m1, 2 + (1 << 2));
                m1 = _mm256_add_epi64(m1, m3);
                _mm256_store_si256((__m256i*)PDMD_RESULT, m1);
                local_sum += PDMD_RESULT[0];
            }
            _count++;
            _sum += local_sum;
            if (SSD_SCALE * (double)local_sum >= thr) {
                *ssd = SSD_SCALE * ((double)_sum) / ((double)_count);
                return true;
            }
        }
    }
    *ssd = SSD_SCALE * ((double)_sum) / ((double)_count);
    return false;
}


void plane_diff_i8(int8_t* a, int8_t* b, int8_t* c, uint32_t stride, uint32_t width, uint32_t height) {
    // Break to NUM_THREADS threads
    uint32_t step_y = height / THREADS_COUNT;
#pragma omp parallel num_threads(THREADS_COUNT)
    {
#pragma omp for
        for (uint32_t section_index = 0; section_index < THREADS_COUNT; section_index++) {
            uint32_t y_start = step_y * section_index;
            uint32_t y_end = y_start + step_y;
            for (uint32_t y = y_start; y < y_end; ++y) {
                avx2_diff_int8(a + stride * y, b + stride * y, c + stride * y, width, section_index);
            }
        }
    }
}

void plane_diff_u8(uint8_t* a, uint8_t* b, uint8_t* c, uint32_t stride, uint32_t width, uint32_t height) {

    for (uint32_t y = 0; y < height; ++y, a += stride, b += stride, c += stride) {
        avx2_diff_uint8(a, b, c, width, 0);
    }
}

void plane_diff_blur_i8(int8_t* a, int8_t* b, int8_t* c, int8_t* d, uint32_t stride, uint32_t width, uint32_t height) {
    plane_diff_i8(a, b, c, stride, width, height);
    //SimdMedianFilterSquare5x5((const uint8_t*)c, stride, width, height, 1, (uint8_t*)d, stride);
}

void plane_diff_blur_u8(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, uint32_t stride, uint32_t width, uint32_t height) {
    SimdGaussianBlur3x3((const uint8_t*)a, stride, width, height, 1, (uint8_t*)c, stride);
    SimdGaussianBlur3x3((const uint8_t*)b, stride, width, height, 1, (uint8_t*)d, stride);
    plane_diff_u8(a, b, c, stride, width, height);
    //SimdMedianFilterSquare5x5((const uint8_t*)c, stride, width, height, 1, (uint8_t*)a, stride);
}


#define SIMD_ALIGN 64
#define SIMD_NO_MANS_LAND 64
#define SIMD_INLINE inline
#define SIMD_X64_ENABLE


namespace Simd
{

    SIMD_INLINE size_t DivHi(size_t value, size_t divider)
    {
        return (value + divider - 1) / divider;
    }

    SIMD_INLINE size_t AlignHiAny(size_t size, size_t align)
    {
        return (size + align - 1) / align * align;
    }

    SIMD_INLINE size_t AlignLoAny(size_t size, size_t align)
    {
        return size / align * align;
    }

    SIMD_INLINE size_t AlignHi(size_t size, size_t align)
    {
        return (size + align - 1) & ~(align - 1);
    }

    SIMD_INLINE void * AlignHi(const void * ptr, size_t align)
    {
        return (void *)((((size_t)ptr) + align - 1) & ~(align - 1));
    }

    SIMD_INLINE size_t AlignLo(size_t size, size_t align)
    {
        return size & ~(align - 1);
    }

    SIMD_INLINE void * AlignLo(const void * ptr, size_t align)
    {
        return (void *)(((size_t)ptr) & ~(align - 1));
    }

    SIMD_INLINE bool Aligned(size_t size, size_t align)
    {
        return size == AlignLo(size, align);
    }

    SIMD_INLINE bool Aligned(const void * ptr, size_t align)
    {
        return ptr == AlignLo(ptr, align);
    }

    template <class T> SIMD_INLINE void Swap(T & a, T & b)
    {
        T t = a;
        a = b;
        b = t;
    }

    template <class T> SIMD_INLINE T Min(T a, T b)
    {
        return a < b ? a : b;
    }

    template <class T> SIMD_INLINE T Max(T a, T b)
    {
        return a > b ? a : b;
    }

    template <class T> SIMD_INLINE T Abs(T a)
    {
        return a < 0 ? -a : a;
    }

    template <class T> SIMD_INLINE T RestrictRange(T value, T min, T max)
    {
        return Max(min, Min(max, value));
    }

    template <class T> SIMD_INLINE T Square(T a)
    {
        return a*a;
    }

#ifndef SIMD_ROUND
#define SIMD_ROUND
    SIMD_INLINE int Round(double value)
    {
#if defined(SIMD_SSE2_ENABLE) && ((defined(_MSC_VER) && defined(_M_X64)) || (defined(__GNUC__) && defined(__x86_64__)))
        __m128d t = _mm_set_sd(value);
        return _mm_cvtsd_si32(t);
#else
        return (int)(value + (value >= 0 ? 0.5 : -0.5));
#endif
    }
#endif

#if defined(_MSC_VER) && (defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE))

    template <class T> SIMD_INLINE char GetChar(T value, size_t index)
    {
        return ((char*)&value)[index];
    }

#define SIMD_AS_CHAR(a) char(a)

#define SIMD_AS_2CHARS(a) \
	Simd::GetChar(int16_t(a), 0), Simd::GetChar(int16_t(a), 1)

#define SIMD_AS_4CHARS(a) \
	Simd::GetChar(int32_t(a), 0), Simd::GetChar(int32_t(a), 1), \
	Simd::GetChar(int32_t(a), 2), Simd::GetChar(int32_t(a), 3)

#define SIMD_AS_8CHARS(a) \
	Simd::GetChar(int64_t(a), 0), Simd::GetChar(int64_t(a), 1), \
	Simd::GetChar(int64_t(a), 2), Simd::GetChar(int64_t(a), 3), \
	Simd::GetChar(int64_t(a), 4), Simd::GetChar(int64_t(a), 5), \
	Simd::GetChar(int64_t(a), 6), Simd::GetChar(int64_t(a), 7)

#elif defined(__GNUC__) || (defined(_MSC_VER) && defined(SIMD_NEON_ENABLE))

#define SIMD_CHAR_AS_LONGLONG(a) (((long long)a) & 0xFF)

#define SIMD_SHORT_AS_LONGLONG(a) (((long long)a) & 0xFFFF)

#define SIMD_INT_AS_LONGLONG(a) (((long long)a) & 0xFFFFFFFF)

#define SIMD_LL_SET1_EPI8(a) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(a) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 16) | (SIMD_CHAR_AS_LONGLONG(a) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 32) | (SIMD_CHAR_AS_LONGLONG(a) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 48) | (SIMD_CHAR_AS_LONGLONG(a) << 56)

#define SIMD_LL_SET2_EPI8(a, b) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(b) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 16) | (SIMD_CHAR_AS_LONGLONG(b) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 32) | (SIMD_CHAR_AS_LONGLONG(b) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 48) | (SIMD_CHAR_AS_LONGLONG(b) << 56)

#define SIMD_LL_SETR_EPI8(a, b, c, d, e, f, g, h) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(b) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(c) << 16) | (SIMD_CHAR_AS_LONGLONG(d) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(e) << 32) | (SIMD_CHAR_AS_LONGLONG(f) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(g) << 48) | (SIMD_CHAR_AS_LONGLONG(h) << 56)

#define SIMD_LL_SET1_EPI16(a) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(a) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(a) << 48)

#define SIMD_LL_SET2_EPI16(a, b) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(b) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(b) << 48)

#define SIMD_LL_SETR_EPI16(a, b, c, d) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(b) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(c) << 32) | (SIMD_SHORT_AS_LONGLONG(d) << 48)

#define SIMD_LL_SET1_EPI32(a) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(a) << 32)

#define SIMD_LL_SET2_EPI32(a, b) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(b) << 32)

#endif//defined(__GNUC__) || (defined(_MSC_VER) && defined(SIMD_NEON_ENABLE))
}

namespace Sse2
{
#if defined(_MSC_VER)

#define SIMD_MM_SET1_EPI8(a) \
    {SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a)}

#define SIMD_MM_SET2_EPI8(a0, a1) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1)}

#define SIMD_MM_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a2), SIMD_AS_CHAR(a3), \
    SIMD_AS_CHAR(a4), SIMD_AS_CHAR(a5), SIMD_AS_CHAR(a6), SIMD_AS_CHAR(a7), \
    SIMD_AS_CHAR(a8), SIMD_AS_CHAR(a9), SIMD_AS_CHAR(aa), SIMD_AS_CHAR(ab), \
    SIMD_AS_CHAR(ac), SIMD_AS_CHAR(ad), SIMD_AS_CHAR(ae), SIMD_AS_CHAR(af)}

#define SIMD_MM_SET1_EPI16(a) \
    {SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
    SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a)}

#define SIMD_MM_SET2_EPI16(a0, a1) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
    SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1)}

#define SIMD_MM_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a2), SIMD_AS_2CHARS(a3), \
    SIMD_AS_2CHARS(a4), SIMD_AS_2CHARS(a5), SIMD_AS_2CHARS(a6), SIMD_AS_2CHARS(a7)}

#define SIMD_MM_SET1_EPI32(a) \
    {SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM_SET2_EPI32(a0, a1) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#define SIMD_MM_SETR_EPI32(a0, a1, a2, a3) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a2), SIMD_AS_4CHARS(a3)}

#define SIMD_MM_SET1_EPI64(a) \
    {SIMD_AS_8CHARS(a), SIMD_AS_8CHARS(a)}

#define SIMD_MM_SET2_EPI64(a0, a1) \
    {SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1)}

#define SIMD_MM_SETR_EPI64(a0, a1) \
    {SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1)}

#elif defined(__GNUC__)

#define SIMD_MM_SET1_EPI8(a) \
    {SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a)}

#define SIMD_MM_SET2_EPI8(a0, a1) \
    {SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1)}

#define SIMD_MM_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_LL_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7), SIMD_LL_SETR_EPI8(a8, a9, aa, ab, ac, ad, ae, af)}

#define SIMD_MM_SET1_EPI16(a) \
    {SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a)}

#define SIMD_MM_SET2_EPI16(a0, a1) \
    {SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1)}

#define SIMD_MM_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_LL_SETR_EPI16(a0, a1, a2, a3), SIMD_LL_SETR_EPI16(a4, a5, a6, a7)}

#define SIMD_MM_SET1_EPI32(a) \
    {SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a)}

#define SIMD_MM_SET2_EPI32(a0, a1) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1)}

#define SIMD_MM_SETR_EPI32(a0, a1, a2, a3) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a2, a3)}

#define SIMD_MM_SET1_EPI64(a) \
    {a, a}

#define SIMD_MM_SET2_EPI64(a0, a1) \
    {a0, a1}

#define SIMD_MM_SETR_EPI64(a0, a1) \
    {a0, a1}

#endif// defined(_MSC_VER) || defined(__GNUC__)
    //using namespace Sse;
#if defined(_MSC_VER) && _MSC_VER >= 1700  && _MSC_VER < 1900 // Visual Studio 2012/2013 compiler bug
    using Sse::F;
    using Sse::DF;
    using Sse::QF;
#endif

    const size_t A = sizeof(__m128i);
    const size_t DA = 2 * A;
    const size_t QA = 4 * A;
    const size_t OA = 8 * A;
    const size_t HA = A / 2;

    const __m128i K_ZERO = SIMD_MM_SET1_EPI8(0);
    const __m128i K_INV_ZERO = SIMD_MM_SET1_EPI8(0xFF);

    const __m128i K8_01 = SIMD_MM_SET1_EPI8(0x01);
    const __m128i K8_02 = SIMD_MM_SET1_EPI8(0x02);
    const __m128i K8_03 = SIMD_MM_SET1_EPI8(0x03);
    const __m128i K8_04 = SIMD_MM_SET1_EPI8(0x04);
    const __m128i K8_07 = SIMD_MM_SET1_EPI8(0x07);
    const __m128i K8_08 = SIMD_MM_SET1_EPI8(0x08);
    const __m128i K8_10 = SIMD_MM_SET1_EPI8(0x10);
    const __m128i K8_20 = SIMD_MM_SET1_EPI8(0x20);
    const __m128i K8_40 = SIMD_MM_SET1_EPI8(0x40);
    const __m128i K8_80 = SIMD_MM_SET1_EPI8(0x80);

    const __m128i K8_01_FF = SIMD_MM_SET2_EPI8(0x01, 0xFF);

    const __m128i K16_0001 = SIMD_MM_SET1_EPI16(0x0001);
    const __m128i K16_0002 = SIMD_MM_SET1_EPI16(0x0002);
    const __m128i K16_0003 = SIMD_MM_SET1_EPI16(0x0003);
    const __m128i K16_0004 = SIMD_MM_SET1_EPI16(0x0004);
    const __m128i K16_0005 = SIMD_MM_SET1_EPI16(0x0005);
    const __m128i K16_0006 = SIMD_MM_SET1_EPI16(0x0006);
    const __m128i K16_0008 = SIMD_MM_SET1_EPI16(0x0008);
    const __m128i K16_0020 = SIMD_MM_SET1_EPI16(0x0020);
    const __m128i K16_0080 = SIMD_MM_SET1_EPI16(0x0080);
    const __m128i K16_00FF = SIMD_MM_SET1_EPI16(0x00FF);
    const __m128i K16_FF00 = SIMD_MM_SET1_EPI16(0xFF00);

    const __m128i K32_00000001 = SIMD_MM_SET1_EPI32(0x00000001);
    const __m128i K32_00000002 = SIMD_MM_SET1_EPI32(0x00000002);
    const __m128i K32_00000004 = SIMD_MM_SET1_EPI32(0x00000004);
    const __m128i K32_00000008 = SIMD_MM_SET1_EPI32(0x00000008);
    const __m128i K32_000000FF = SIMD_MM_SET1_EPI32(0x000000FF);
    const __m128i K32_0000FFFF = SIMD_MM_SET1_EPI32(0x0000FFFF);
    const __m128i K32_00010000 = SIMD_MM_SET1_EPI32(0x00010000);
    const __m128i K32_01000000 = SIMD_MM_SET1_EPI32(0x01000000);
    const __m128i K32_00FFFFFF = SIMD_MM_SET1_EPI32(0x00FFFFFF);
    const __m128i K32_FFFFFF00 = SIMD_MM_SET1_EPI32(0xFFFFFF00);

    const __m128i K64_00000000FFFFFFFF = SIMD_MM_SET2_EPI32(0xFFFFFFFF, 0);

    /*const __m128i K16_Y_ADJUST = SIMD_MM_SET1_EPI16(Base::Y_ADJUST);
    const __m128i K16_UV_ADJUST = SIMD_MM_SET1_EPI16(Base::UV_ADJUST);

    const __m128i K16_YRGB_RT = SIMD_MM_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
    const __m128i K16_VR_0 = SIMD_MM_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
    const __m128i K16_UG_VG = SIMD_MM_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
    const __m128i K16_UB_0 = SIMD_MM_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);

    const __m128i K16_BY_RY = SIMD_MM_SET2_EPI16(Base::BLUE_TO_Y_WEIGHT, Base::RED_TO_Y_WEIGHT);
    const __m128i K16_GY_RT = SIMD_MM_SET2_EPI16(Base::GREEN_TO_Y_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
    const __m128i K16_BU_RU = SIMD_MM_SET2_EPI16(Base::BLUE_TO_U_WEIGHT, Base::RED_TO_U_WEIGHT);
    const __m128i K16_GU_RT = SIMD_MM_SET2_EPI16(Base::GREEN_TO_U_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
    const __m128i K16_BV_RV = SIMD_MM_SET2_EPI16(Base::BLUE_TO_V_WEIGHT, Base::RED_TO_V_WEIGHT);
    const __m128i K16_GV_RT = SIMD_MM_SET2_EPI16(Base::GREEN_TO_V_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);

    const __m128i K16_DIVISION_BY_9_FACTOR = SIMD_MM_SET1_EPI16(Base::DIVISION_BY_9_FACTOR);*/
}

namespace AVX2
{
    SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(__m256))
    {
        return Simd::Aligned(size, align);
    }

    SIMD_INLINE bool Aligned(const void * ptr, size_t align = sizeof(__m256))
    {
        return Simd::Aligned(ptr, align);
    }
#if 1

#if defined(_MSC_VER) && (defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE))

    template <class T> SIMD_INLINE char GetChar(T value, size_t index)
    {
        return ((char*)& value)[index];
    }

#define SIMD_AS_CHAR(a) char(a)

#define SIMD_AS_2CHARS(a) \
	Simd::GetChar(int16_t(a), 0), Simd::GetChar(int16_t(a), 1)

#define SIMD_AS_4CHARS(a) \
	Simd::GetChar(int32_t(a), 0), Simd::GetChar(int32_t(a), 1), \
	Simd::GetChar(int32_t(a), 2), Simd::GetChar(int32_t(a), 3)

#define SIMD_AS_8CHARS(a) \
	Simd::GetChar(int64_t(a), 0), Simd::GetChar(int64_t(a), 1), \
	Simd::GetChar(int64_t(a), 2), Simd::GetChar(int64_t(a), 3), \
	Simd::GetChar(int64_t(a), 4), Simd::GetChar(int64_t(a), 5), \
	Simd::GetChar(int64_t(a), 6), Simd::GetChar(int64_t(a), 7)

#elif defined(__GNUC__) || (defined(_MSC_VER) && defined(SIMD_NEON_ENABLE))

#define SIMD_CHAR_AS_LONGLONG(a) (((long long)a) & 0xFF)

#define SIMD_SHORT_AS_LONGLONG(a) (((long long)a) & 0xFFFF)

#define SIMD_INT_AS_LONGLONG(a) (((long long)a) & 0xFFFFFFFF)

#define SIMD_LL_SET1_EPI8(a) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(a) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 16) | (SIMD_CHAR_AS_LONGLONG(a) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 32) | (SIMD_CHAR_AS_LONGLONG(a) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 48) | (SIMD_CHAR_AS_LONGLONG(a) << 56)

#define SIMD_LL_SET2_EPI8(a, b) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(b) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 16) | (SIMD_CHAR_AS_LONGLONG(b) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 32) | (SIMD_CHAR_AS_LONGLONG(b) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 48) | (SIMD_CHAR_AS_LONGLONG(b) << 56)

#define SIMD_LL_SETR_EPI8(a, b, c, d, e, f, g, h) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(b) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(c) << 16) | (SIMD_CHAR_AS_LONGLONG(d) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(e) << 32) | (SIMD_CHAR_AS_LONGLONG(f) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(g) << 48) | (SIMD_CHAR_AS_LONGLONG(h) << 56)

#define SIMD_LL_SET1_EPI16(a) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(a) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(a) << 48)

#define SIMD_LL_SET2_EPI16(a, b) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(b) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(a) << 32) | (SIMD_SHORT_AS_LONGLONG(b) << 48)

#define SIMD_LL_SETR_EPI16(a, b, c, d) \
    SIMD_SHORT_AS_LONGLONG(a) | (SIMD_SHORT_AS_LONGLONG(b) << 16) | \
    (SIMD_SHORT_AS_LONGLONG(c) << 32) | (SIMD_SHORT_AS_LONGLONG(d) << 48)

#define SIMD_LL_SET1_EPI32(a) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(a) << 32)

#define SIMD_LL_SET2_EPI32(a, b) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(b) << 32)

#endif//defined(__GNUC__) || (defined(_MSC_VER) && defined(SIMD_NEON_ENABLE))

#if defined(_MSC_VER)

#define SIMD_MM256_SET1_EPI8(a) \
	{SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a)}

#define SIMD_MM256_SET2_EPI8(a0, a1) \
	{SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1)}

#define SIMD_MM256_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a2), SIMD_AS_CHAR(a3), \
    SIMD_AS_CHAR(a4), SIMD_AS_CHAR(a5), SIMD_AS_CHAR(a6), SIMD_AS_CHAR(a7), \
    SIMD_AS_CHAR(a8), SIMD_AS_CHAR(a9), SIMD_AS_CHAR(aa), SIMD_AS_CHAR(ab), \
    SIMD_AS_CHAR(ac), SIMD_AS_CHAR(ad), SIMD_AS_CHAR(ae), SIMD_AS_CHAR(af), \
    SIMD_AS_CHAR(b0), SIMD_AS_CHAR(b1), SIMD_AS_CHAR(b2), SIMD_AS_CHAR(b3), \
    SIMD_AS_CHAR(b4), SIMD_AS_CHAR(b5), SIMD_AS_CHAR(b6), SIMD_AS_CHAR(b7), \
    SIMD_AS_CHAR(b8), SIMD_AS_CHAR(b9), SIMD_AS_CHAR(ba), SIMD_AS_CHAR(bb), \
    SIMD_AS_CHAR(bc), SIMD_AS_CHAR(bd), SIMD_AS_CHAR(be), SIMD_AS_CHAR(bf)}

#define SIMD_MM256_SET1_EPI16(a) \
	{SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a)}

#define SIMD_MM256_SET2_EPI16(a0, a1) \
	{SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1)}

#define SIMD_MM256_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a2), SIMD_AS_2CHARS(a3), \
    SIMD_AS_2CHARS(a4), SIMD_AS_2CHARS(a5), SIMD_AS_2CHARS(a6), SIMD_AS_2CHARS(a7), \
    SIMD_AS_2CHARS(a8), SIMD_AS_2CHARS(a9), SIMD_AS_2CHARS(aa), SIMD_AS_2CHARS(ab), \
    SIMD_AS_2CHARS(ac), SIMD_AS_2CHARS(ad), SIMD_AS_2CHARS(ae), SIMD_AS_2CHARS(af)}

#define SIMD_MM256_SET1_EPI32(a) \
	{SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), \
	SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM256_SET2_EPI32(a0, a1) \
	{SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), \
	SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#define SIMD_MM256_SETR_EPI32(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a2), SIMD_AS_4CHARS(a3), \
    SIMD_AS_4CHARS(a4), SIMD_AS_4CHARS(a5), SIMD_AS_4CHARS(a6), SIMD_AS_4CHARS(a7)}

#define SIMD_MM256_SET1_EPI64(a) \
	{SIMD_AS_8CHARS(a), SIMD_AS_8CHARS(a), SIMD_AS_8CHARS(a), SIMD_AS_8CHARS(a)}

#define SIMD_MM256_SET2_EPI64(a0, a1) \
	{SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1), SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1)}

#define SIMD_MM256_SETR_EPI64(a0, a1, a2, a3) \
    {SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1), SIMD_AS_8CHARS(a2), SIMD_AS_8CHARS(a3)}

#elif defined(__GNUC__)

#define SIMD_MM256_SET1_EPI8(a) \
    {SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a), \
    SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a)}

#define SIMD_MM256_SET2_EPI8(a0, a1) \
    {SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1), \
    SIMD_LL_SET2_EPI8(a0, a1), SIMD_LL_SET2_EPI8(a0, a1)}

#define SIMD_MM256_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf) \
    {SIMD_LL_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7), SIMD_LL_SETR_EPI8(a8, a9, aa, ab, ac, ad, ae, af), \
    SIMD_LL_SETR_EPI8(b0, b1, b2, b3, b4, b5, b6, b7), SIMD_LL_SETR_EPI8(b8, b9, ba, bb, bc, bd, be, bf)}

#define SIMD_MM256_SET1_EPI16(a) \
    {SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a), \
    SIMD_LL_SET1_EPI16(a), SIMD_LL_SET1_EPI16(a)}

#define SIMD_MM256_SET2_EPI16(a0, a1) \
    {SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1), \
    SIMD_LL_SET2_EPI16(a0, a1), SIMD_LL_SET2_EPI16(a0, a1)}

#define SIMD_MM256_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_LL_SETR_EPI16(a0, a1, a2, a3), SIMD_LL_SETR_EPI16(a4, a5, a6, a7), \
    SIMD_LL_SETR_EPI16(a8, a9, aa, ab), SIMD_LL_SETR_EPI16(ac, ad, ae, af)}

#define SIMD_MM256_SET1_EPI32(a) \
    {SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a), \
    SIMD_LL_SET1_EPI32(a), SIMD_LL_SET1_EPI32(a)}

#define SIMD_MM256_SET2_EPI32(a0, a1) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1), \
    SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a0, a1)}

#define SIMD_MM256_SETR_EPI32(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a2, a3), \
    SIMD_LL_SET2_EPI32(a4, a5), SIMD_LL_SET2_EPI32(a6, a7)}

#define SIMD_MM256_SET1_EPI64(a) \
    {a, a, a, a}

#define SIMD_MM256_SET2_EPI64(a0, a1) \
    {a0, a1, a0, a1}

#define SIMD_MM256_SETR_EPI64(a0, a1, a2, a3) \
    {a0, a1, a2, a3}

#endif// defined(_MSC_VER) || defined(__GNUC__)

    SIMD_INLINE void* Allocate(size_t size, size_t align = SIMD_ALIGN)
    {
#ifdef SIMD_NO_MANS_LAND
        size += 2 * SIMD_NO_MANS_LAND;
#endif
        void* ptr = NULL;
#if defined(_MSC_VER) 
        ptr = _aligned_malloc(size, align);
#elif defined(__MINGW32__) || defined(__MINGW64__)
        ptr = __mingw_aligned_malloc(size, align);
#elif defined(__GNUC__)
        align = AlignHi(align, sizeof(void*));
        size = AlignHi(size, align);
        int result = ::posix_memalign(&ptr, align, size);
#ifdef SIMD_ALLOCATE_ERROR_MESSAGE
        if (result != 0)
            std::cout << "The function posix_memalign can't allocate " << size << " bytes with align " << align << " !" << std::endl << std::flush;
#endif
#ifdef SIMD_ALLOCATE_ASSERT
        assert(result == 0);
#endif
#else
        ptr = malloc(size);
#endif

#ifdef SIMD_NO_MANS_LAND
        if (ptr)
            ptr = (char*)ptr + SIMD_NO_MANS_LAND;
#endif
        return ptr;
    }

    SIMD_INLINE void Free(void* ptr)
    {
#ifdef SIMD_NO_MANS_LAND
        if (ptr)
            ptr = (char*)ptr - SIMD_NO_MANS_LAND;
#endif
#if defined(_MSC_VER) 
        _aligned_free(ptr);
#elif defined(__MINGW32__) || defined(__MINGW64__)
        return __mingw_aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    const size_t A = sizeof(__m256i);
    const size_t DA = 2 * A;
    const size_t QA = 4 * A;
    const size_t OA = 8 * A;
    const size_t HA = A / 2;

    const __m256i K_ZERO = SIMD_MM256_SET1_EPI8(0);
    const __m256i K_INV_ZERO = SIMD_MM256_SET1_EPI8(0xFF);

    const __m256i K8_01 = SIMD_MM256_SET1_EPI8(0x01);
    const __m256i K8_02 = SIMD_MM256_SET1_EPI8(0x02);
    const __m256i K8_03 = SIMD_MM256_SET1_EPI8(0x03);
    const __m256i K8_04 = SIMD_MM256_SET1_EPI8(0x04);
    const __m256i K8_07 = SIMD_MM256_SET1_EPI8(0x07);
    const __m256i K8_08 = SIMD_MM256_SET1_EPI8(0x08);
    const __m256i K8_10 = SIMD_MM256_SET1_EPI8(0x10);
    const __m256i K8_20 = SIMD_MM256_SET1_EPI8(0x20);
    const __m256i K8_40 = SIMD_MM256_SET1_EPI8(0x40);
    const __m256i K8_80 = SIMD_MM256_SET1_EPI8(0x80);

    const __m256i K8_01_FF = SIMD_MM256_SET2_EPI8(0x01, 0xFF);

    const __m256i K16_0001 = SIMD_MM256_SET1_EPI16(0x0001);
    const __m256i K16_0002 = SIMD_MM256_SET1_EPI16(0x0002);
    const __m256i K16_0003 = SIMD_MM256_SET1_EPI16(0x0003);
    const __m256i K16_0004 = SIMD_MM256_SET1_EPI16(0x0004);
    const __m256i K16_0005 = SIMD_MM256_SET1_EPI16(0x0005);
    const __m256i K16_0006 = SIMD_MM256_SET1_EPI16(0x0006);
    const __m256i K16_0008 = SIMD_MM256_SET1_EPI16(0x0008);
    const __m256i K16_0010 = SIMD_MM256_SET1_EPI16(0x0010);
    const __m256i K16_0018 = SIMD_MM256_SET1_EPI16(0x0018);
    const __m256i K16_0020 = SIMD_MM256_SET1_EPI16(0x0020);
    const __m256i K16_0080 = SIMD_MM256_SET1_EPI16(0x0080);
    const __m256i K16_00FF = SIMD_MM256_SET1_EPI16(0x00FF);
    const __m256i K16_FF00 = SIMD_MM256_SET1_EPI16(0xFF00);

    /*const __m256i K32_00000001 = SIMD_MM256_SET1_EPI32(0x00000001);
    const __m256i K32_00000002 = SIMD_MM256_SET1_EPI32(0x00000002);
    const __m256i K32_00000004 = SIMD_MM256_SET1_EPI32(0x00000004);
    const __m256i K32_00000008 = SIMD_MM256_SET1_EPI32(0x00000008);
    const __m256i K32_000000FF = SIMD_MM256_SET1_EPI32(0x000000FF);
    const __m256i K32_0000FFFF = SIMD_MM256_SET1_EPI32(0x0000FFFF);
    const __m256i K32_00010000 = SIMD_MM256_SET1_EPI32(0x00010000);
    const __m256i K32_01000000 = SIMD_MM256_SET1_EPI32(0x01000000);
    const __m256i K32_FFFFFF00 = SIMD_MM256_SET1_EPI32(0xFFFFFF00);

    const __m256i K16_Y_ADJUST = SIMD_MM256_SET1_EPI16(Base::Y_ADJUST);
    const __m256i K16_UV_ADJUST = SIMD_MM256_SET1_EPI16(Base::UV_ADJUST);

    const __m256i K16_YRGB_RT = SIMD_MM256_SET2_EPI16(Base::Y_TO_RGB_WEIGHT, Base::YUV_TO_BGR_ROUND_TERM);
    const __m256i K16_VR_0 = SIMD_MM256_SET2_EPI16(Base::V_TO_RED_WEIGHT, 0);
    const __m256i K16_UG_VG = SIMD_MM256_SET2_EPI16(Base::U_TO_GREEN_WEIGHT, Base::V_TO_GREEN_WEIGHT);
    const __m256i K16_UB_0 = SIMD_MM256_SET2_EPI16(Base::U_TO_BLUE_WEIGHT, 0);*/

    /*const __m256i K16_BY_RY = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_Y_WEIGHT, Base::RED_TO_Y_WEIGHT);
    const __m256i K16_GY_RT = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_Y_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
    const __m256i K16_BU_RU = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_U_WEIGHT, Base::RED_TO_U_WEIGHT);
    const __m256i K16_GU_RT = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_U_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);
    const __m256i K16_BV_RV = SIMD_MM256_SET2_EPI16(Base::BLUE_TO_V_WEIGHT, Base::RED_TO_V_WEIGHT);
    const __m256i K16_GV_RT = SIMD_MM256_SET2_EPI16(Base::GREEN_TO_V_WEIGHT, Base::BGR_TO_YUV_ROUND_TERM);

    const __m256i K16_DIVISION_BY_9_FACTOR = SIMD_MM256_SET1_EPI16(Base::DIVISION_BY_9_FACTOR);*/

    const __m256i K8_SHUFFLE_0 = SIMD_MM256_SETR_EPI8(
        0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70,
        0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0);

    const __m256i K8_SHUFFLE_1 = SIMD_MM256_SETR_EPI8(
        0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
        0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70);

    const __m256i K8_SHUFFLE_GRAY_TO_BGR0 = SIMD_MM256_SETR_EPI8(
        0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5,
        0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA);
    const __m256i K8_SHUFFLE_GRAY_TO_BGR1 = SIMD_MM256_SETR_EPI8(
        0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7,
        0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xA, 0xB, 0xB, 0xB, 0xC, 0xC, 0xC, 0xD);
    const __m256i K8_SHUFFLE_GRAY_TO_BGR2 = SIMD_MM256_SETR_EPI8(
        0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA,
        0xA, 0xB, 0xB, 0xB, 0xC, 0xC, 0xC, 0xD, 0xD, 0xD, 0xE, 0xE, 0xE, 0xF, 0xF, 0xF);

    const __m256i K8_SHUFFLE_PERMUTED_BLUE_TO_BGR0 = SIMD_MM256_SETR_EPI8(
        0x0, -1, -1, 0x1, -1, -1, 0x2, -1, -1, 0x3, -1, -1, 0x4, -1, -1, 0x5,
        -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA, -1);
    const __m256i K8_SHUFFLE_PERMUTED_BLUE_TO_BGR1 = SIMD_MM256_SETR_EPI8(
        -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1,
        0x8, -1, -1, 0x9, -1, -1, 0xA, -1, -1, 0xB, -1, -1, 0xC, -1, -1, 0xD);
    const __m256i K8_SHUFFLE_PERMUTED_BLUE_TO_BGR2 = SIMD_MM256_SETR_EPI8(
        -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA, -1,
        -1, 0xB, -1, -1, 0xC, -1, -1, 0xD, -1, -1, 0xE, -1, -1, 0xF, -1, -1);

    const __m256i K8_SHUFFLE_PERMUTED_GREEN_TO_BGR0 = SIMD_MM256_SETR_EPI8(
        -1, 0x0, -1, -1, 0x1, -1, -1, 0x2, -1, -1, 0x3, -1, -1, 0x4, -1, -1,
        0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA);
    const __m256i K8_SHUFFLE_PERMUTED_GREEN_TO_BGR1 = SIMD_MM256_SETR_EPI8(
        -1, -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6, -1, -1, 0x7, -1,
        -1, 0x8, -1, -1, 0x9, -1, -1, 0xA, -1, -1, 0xB, -1, -1, 0xC, -1, -1);
    const __m256i K8_SHUFFLE_PERMUTED_GREEN_TO_BGR2 = SIMD_MM256_SETR_EPI8(
        0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA,
        -1, -1, 0xB, -1, -1, 0xC, -1, -1, 0xD, -1, -1, 0xE, -1, -1, 0xF, -1);

    const __m256i K8_SHUFFLE_PERMUTED_RED_TO_BGR0 = SIMD_MM256_SETR_EPI8(
        -1, -1, 0x0, -1, -1, 0x1, -1, -1, 0x2, -1, -1, 0x3, -1, -1, 0x4, -1,
        -1, 0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1);
    const __m256i K8_SHUFFLE_PERMUTED_RED_TO_BGR1 = SIMD_MM256_SETR_EPI8(
        0x2, -1, -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6, -1, -1, 0x7,
        -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA, -1, -1, 0xB, -1, -1, 0xC, -1);
    const __m256i K8_SHUFFLE_PERMUTED_RED_TO_BGR2 = SIMD_MM256_SETR_EPI8(
        -1, 0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1,
        0xA, -1, -1, 0xB, -1, -1, 0xC, -1, -1, 0xD, -1, -1, 0xE, -1, -1, 0xF);

    const __m256i K8_SHUFFLE_BGR0_TO_BLUE = SIMD_MM256_SETR_EPI8(
        0x0, 0x3, 0x6, 0x9, 0xC, 0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 0x2, 0x5, 0x8, 0xB, 0xE, -1, -1, -1, -1, -1);
    const __m256i K8_SHUFFLE_BGR1_TO_BLUE = SIMD_MM256_SETR_EPI8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x4, 0x7, 0xA, 0xD,
        0x0, 0x3, 0x6, 0x9, 0xC, 0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i K8_SHUFFLE_BGR2_TO_BLUE = SIMD_MM256_SETR_EPI8(
        -1, -1, -1, -1, -1, -1, 0x2, 0x5, 0x8, 0xB, 0xE, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x4, 0x7, 0xA, 0xD);

    const __m256i K8_SHUFFLE_BGR0_TO_GREEN = SIMD_MM256_SETR_EPI8(
        0x1, 0x4, 0x7, 0xA, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF, -1, -1, -1, -1, -1);
    const __m256i K8_SHUFFLE_BGR1_TO_GREEN = SIMD_MM256_SETR_EPI8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x5, 0x8, 0xB, 0xE,
        0x1, 0x4, 0x7, 0xA, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i K8_SHUFFLE_BGR2_TO_GREEN = SIMD_MM256_SETR_EPI8(
        -1, -1, -1, -1, -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x5, 0x8, 0xB, 0xE);

    const __m256i K8_SHUFFLE_BGR0_TO_RED = SIMD_MM256_SETR_EPI8(
        0x2, 0x5, 0x8, 0xB, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, 0x1, 0x4, 0x7, 0xA, 0xD, -1, -1, -1, -1, -1, -1);
    const __m256i K8_SHUFFLE_BGR1_TO_RED = SIMD_MM256_SETR_EPI8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF,
        0x2, 0x5, 0x8, 0xB, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i K8_SHUFFLE_BGR2_TO_RED = SIMD_MM256_SETR_EPI8(
        -1, -1, -1, -1, -1, 0x1, 0x4, 0x7, 0xA, 0xD, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x3, 0x6, 0x9, 0xC, 0xF);

    const __m256i K8_BGRA_TO_BGR_SHUFFLE = SIMD_MM256_SETR_EPI8(
        0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
        0x4, 0x5, 0x6, -1, 0x7, 0x8, 0x9, -1, 0xA, 0xB, 0xC, -1, 0xD, 0xE, 0xF, -1);

    const __m256i K8_BGRA_TO_RGB_SHUFFLE = SIMD_MM256_SETR_EPI8(
        0x2, 0x1, 0x0, -1, 0x5, 0x4, 0x3, -1, 0x8, 0x7, 0x6, -1, 0xB, 0xA, 0x9, -1,
        0x6, 0x5, 0x4, -1, 0x9, 0x8, 0x7, -1, 0xC, 0xB, 0xA, -1, 0xF, 0xE, 0xD, -1);

    const __m256i K32_TWO_UNPACK_PERMUTE = SIMD_MM256_SETR_EPI32(0, 2, 4, 6, 1, 3, 5, 7);


#if 1

    SIMD_INLINE __m256i SetInt8(char a0, char a1)
    {
        return _mm256_unpacklo_epi8(_mm256_set1_epi8(a0), _mm256_set1_epi8(a1));
    }

    SIMD_INLINE __m256i SetInt16(short a0, short a1)
    {
        return _mm256_unpacklo_epi16(_mm256_set1_epi16(a0), _mm256_set1_epi16(a1));
    }

    SIMD_INLINE __m256i SetInt32(int a0, int a1)
    {
        return _mm256_unpacklo_epi32(_mm256_set1_epi32(a0), _mm256_set1_epi32(a1));
    }

    SIMD_INLINE __m256 SetFloat(float a0, float a1)
    {
        return _mm256_unpacklo_ps(_mm256_set1_ps(a0), _mm256_set1_ps(a1));
    }

    template <bool align> SIMD_INLINE __m256i Load(const __m256i * p);

    template <> SIMD_INLINE __m256i Load<false>(const __m256i * p)
    {
        return _mm256_loadu_si256(p);
    }

    template <> SIMD_INLINE __m256i Load<true>(const __m256i * p)
    {
        return _mm256_load_si256(p);
    }

    template <bool align> SIMD_INLINE __m128i LoadHalf(const __m128i * p);

    template <> SIMD_INLINE __m128i LoadHalf<false>(const __m128i * p)
    {
        return _mm_loadu_si128(p);
    }

    template <> SIMD_INLINE __m128i LoadHalf<true>(const __m128i * p)
    {
        return _mm_load_si128(p);
    }

    template <size_t count> SIMD_INLINE __m128i LoadHalfBeforeFirst(__m128i first)
    {
        return _mm_or_si128(_mm_slli_si128(first, count), _mm_and_si128(first, _mm_srli_si128(Sse2::K_INV_ZERO, HA - count)));
    }

    template <size_t count> SIMD_INLINE __m128i LoadHalfAfterLast(__m128i last)
    {
        return _mm_or_si128(_mm_srli_si128(last, count), _mm_and_si128(last, _mm_slli_si128(Sse2::K_INV_ZERO, HA - count)));
    }

    template <bool align> SIMD_INLINE __m256i LoadPermuted(const __m256i * p)
    {
        return _mm256_permute4x64_epi64(Load<align>(p), 0xD8);
    }

    template <bool align> SIMD_INLINE __m256i LoadMaskI8(const __m256i * p, __m256i index)
    {
        return _mm256_cmpeq_epi8(Load<align>(p), index);
    }

    SIMD_INLINE __m256i PermutedUnpackLoU8(__m256i a, __m256i b = K_ZERO)
    {
        return _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(a, b), 0xD8);
    }

    SIMD_INLINE __m256i PermutedUnpackHiU8(__m256i a, __m256i b = K_ZERO)
    {
        return _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(a, b), 0xD8);
    }

    SIMD_INLINE __m256i PermutedUnpackLoU16(__m256i a, __m256i b = K_ZERO)
    {
        return _mm256_permute4x64_epi64(_mm256_unpacklo_epi16(a, b), 0xD8);
    }

    SIMD_INLINE __m256i PermutedUnpackHiU16(__m256i a, __m256i b = K_ZERO)
    {
        return _mm256_permute4x64_epi64(_mm256_unpackhi_epi16(a, b), 0xD8);
    }

    template <bool align, size_t step> SIMD_INLINE __m256i LoadBeforeFirst(const uint8_t * p)
    {
        __m128i lo = LoadHalfBeforeFirst<step>(LoadHalf<align>((__m128i*)p));
        __m128i hi = _mm_loadu_si128((__m128i*)(p + HA - step));
        return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 0x1);
    }

    template <bool align, size_t step> SIMD_INLINE void LoadBeforeFirst(const uint8_t * p, __m256i & first, __m256i & second)
    {
        __m128i firstLo = LoadHalfBeforeFirst<step>(LoadHalf<align>((__m128i*)p));
        __m128i firstHi = _mm_loadu_si128((__m128i*)(p + HA - step));
        first = _mm256_inserti128_si256(_mm256_castsi128_si256(firstLo), firstHi, 0x1);

        __m128i secondLo = LoadHalfBeforeFirst<step>(firstLo);
        __m128i secondHi = _mm_loadu_si128((__m128i*)(p + HA - 2 * step));
        second = _mm256_inserti128_si256(_mm256_castsi128_si256(secondLo), secondHi, 0x1);
    }

    template <bool align, size_t step> SIMD_INLINE __m256i LoadAfterLast(const uint8_t * p)
    {
        __m128i lo = _mm_loadu_si128((__m128i*)(p + step));
        __m128i hi = LoadHalfAfterLast<step>(LoadHalf<align>((__m128i*)(p + HA)));
        return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 0x1);
    }

    template <bool align, size_t step> SIMD_INLINE void LoadAfterLast(const uint8_t * p, __m256i & first, __m256i & second)
    {
        __m128i firstLo = _mm_loadu_si128((__m128i*)(p + step));
        __m128i firstHi = LoadHalfAfterLast<step>(LoadHalf<align>((__m128i*)(p + HA)));
        first = _mm256_inserti128_si256(_mm256_castsi128_si256(firstLo), firstHi, 0x1);

        __m128i secondLo = _mm_loadu_si128((__m128i*)(p + 2 * step));
        __m128i secondHi = LoadHalfAfterLast<step>(firstHi);
        second = _mm256_inserti128_si256(_mm256_castsi128_si256(secondLo), secondHi, 0x1);
    }

    template <bool align, size_t step> SIMD_INLINE void LoadNose3(const uint8_t * p, __m256i a[3])
    {
        a[0] = LoadBeforeFirst<align, step>(p);
        a[1] = Load<align>((__m256i*)p);
        a[2] = _mm256_loadu_si256((__m256i*)(p + step));
    }

    template <bool align, size_t step> SIMD_INLINE void LoadBody3(const uint8_t * p, __m256i a[3])
    {
        a[0] = _mm256_loadu_si256((__m256i*)(p - step));
        a[1] = Load<align>((__m256i*)p);
        a[2] = _mm256_loadu_si256((__m256i*)(p + step));
    }

    template <bool align, size_t step> SIMD_INLINE void LoadTail3(const uint8_t * p, __m256i a[3])
    {
        a[0] = _mm256_loadu_si256((__m256i*)(p - step));
        a[1] = Load<align>((__m256i*)p);
        a[2] = LoadAfterLast<align, step>(p);
    }

    template <bool align, size_t step> SIMD_INLINE void LoadNose5(const uint8_t * p, __m256i a[5])
    {
        LoadBeforeFirst<align, step>(p, a[1], a[0]);
        a[2] = Load<align>((__m256i*)p);
        a[3] = _mm256_loadu_si256((__m256i*)(p + step));
        a[4] = _mm256_loadu_si256((__m256i*)(p + 2 * step));
    }

    template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, __m256i a[5])
    {
        a[0] = _mm256_loadu_si256((__m256i*)(p - 2 * step));
        a[1] = _mm256_loadu_si256((__m256i*)(p - step));
        a[2] = Load<align>((__m256i*)p);
        a[3] = _mm256_loadu_si256((__m256i*)(p + step));
        a[4] = _mm256_loadu_si256((__m256i*)(p + 2 * step));
    }

    template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, __m256i a[5])
    {
        a[0] = _mm256_loadu_si256((__m256i*)(p - 2 * step));
        a[1] = _mm256_loadu_si256((__m256i*)(p - step));
        a[2] = Load<align>((__m256i*)p);
        LoadAfterLast<align, step>(p, a[3], a[4]);
    }

    SIMD_INLINE void LoadNoseDx(const uint8_t * p, __m256i a[3])
    {
        a[0] = LoadBeforeFirst<false, 1>(p);
        a[2] = _mm256_loadu_si256((__m256i*)(p + 1));
    }

    SIMD_INLINE void LoadBodyDx(const uint8_t * p, __m256i a[3])
    {
        a[0] = _mm256_loadu_si256((__m256i*)(p - 1));
        a[2] = _mm256_loadu_si256((__m256i*)(p + 1));
    }

    SIMD_INLINE void LoadTailDx(const uint8_t * p, __m256i a[3])
    {
        a[0] = _mm256_loadu_si256((__m256i*)(p - 1));
        a[2] = LoadAfterLast<false, 1>(p);
    }

    template <bool align> SIMD_INLINE __m256 Load(const float * p);

    template <> SIMD_INLINE __m256 Load<false>(const float * p)
    {
        return _mm256_loadu_ps(p);
    }

    template <> SIMD_INLINE __m256 Load<true>(const float * p)
    {
#ifdef _MSC_VER
        return _mm256_castsi256_ps(_mm256_load_si256((__m256i*)p));
#else
        return _mm256_load_ps(p);
#endif
    }


    template <bool align> SIMD_INLINE void Store(__m256i * p, __m256i a);

    template <> SIMD_INLINE void Store<false>(__m256i * p, __m256i a)
    {
        _mm256_storeu_si256(p, a);
    }

    template <> SIMD_INLINE void Store<true>(__m256i * p, __m256i a)
    {
        _mm256_store_si256(p, a);
    }

    template <bool align> SIMD_INLINE void StoreMasked(__m256i * p, __m256i value, __m256i mask)
    {
        __m256i old = Load<align>(p);
        Store<align>(p, _mm256_blendv_epi8(old, value, mask));
    }

    SIMD_INLINE __m256i PackI16ToI8(__m256i lo, __m256i hi)
    {
        return _mm256_permute4x64_epi64(_mm256_packs_epi16(lo, hi), 0xD8);
    }

    SIMD_INLINE __m256i PackU16ToU8(__m256i lo, __m256i hi)
    {
        return _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xD8);
    }

    SIMD_INLINE __m256i PackI32ToI16(__m256i lo, __m256i hi)
    {
        return _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0xD8);
    }

    SIMD_INLINE __m256i PackU32ToI16(__m256i lo, __m256i hi)
    {
        return _mm256_permute4x64_epi64(_mm256_packus_epi32(lo, hi), 0xD8);
    }

    SIMD_INLINE void Permute2x128(__m256i & lo, __m256i & hi)
    {
        __m256i _lo = lo;
        lo = _mm256_permute2x128_si256(lo, hi, 0x20);
        hi = _mm256_permute2x128_si256(_lo, hi, 0x31);
    }


    template <class T> SIMD_INLINE __m256i SetMask(T first, size_t position, T second)
    {
        const size_t size = A / sizeof(T);
        assert(position <= size);
        T mask[size];
        for (size_t i = 0; i < position; ++i)
            mask[i] = first;
        for (size_t i = position; i < size; ++i)
            mask[i] = second;
        return _mm256_loadu_si256((__m256i*)mask);
    }

#endif

#if 1

    SIMD_INLINE __m256i SaturateI16ToU8(__m256i value)
    {
        return _mm256_min_epi16(K16_00FF, _mm256_max_epi16(value, K_ZERO));
    }

    SIMD_INLINE __m256i MaxI16(__m256i a, __m256i b, __m256i c)
    {
        return _mm256_max_epi16(a, _mm256_max_epi16(b, c));
    }

    SIMD_INLINE __m256i MinI16(__m256i a, __m256i b, __m256i c)
    {
        return _mm256_min_epi16(a, _mm256_min_epi16(b, c));
    }

    SIMD_INLINE void SortU8(__m256i & a, __m256i & b)
    {
        __m256i t = a;
        a = _mm256_min_epu8(t, b);
        b = _mm256_max_epu8(t, b);
    }

    SIMD_INLINE __m256i HorizontalSum32(__m256i a)
    {
        return _mm256_add_epi64(_mm256_unpacklo_epi32(a, K_ZERO), _mm256_unpackhi_epi32(a, K_ZERO));
    }

    SIMD_INLINE __m256i AbsDifferenceU8(__m256i a, __m256i b)
    {
        return _mm256_sub_epi8(_mm256_max_epu8(a, b), _mm256_min_epu8(a, b));
    }

    SIMD_INLINE __m256i AbsDifferenceI16(__m256i a, __m256i b)
    {
        return _mm256_sub_epi16(_mm256_max_epi16(a, b), _mm256_min_epi16(a, b));
    }

    SIMD_INLINE __m256i MulU8(__m256i a, __m256i b)
    {
        __m256i lo = _mm256_mullo_epi16(_mm256_unpacklo_epi8(a, K_ZERO), _mm256_unpacklo_epi8(b, K_ZERO));
        __m256i hi = _mm256_mullo_epi16(_mm256_unpackhi_epi8(a, K_ZERO), _mm256_unpackhi_epi8(b, K_ZERO));
        return _mm256_packus_epi16(lo, hi);
    }

    SIMD_INLINE __m256i DivideI16By255(__m256i value)
    {
        return _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(value, K16_0001), _mm256_srli_epi16(value, 8)), 8);
    }

    SIMD_INLINE __m256i BinomialSum16(const __m256i & a, const __m256i & b, const __m256i & c)
    {
        return _mm256_add_epi16(_mm256_add_epi16(a, c), _mm256_add_epi16(b, b));
    }

    template <bool abs> __m256i ConditionalAbs(__m256i a);

    template <> SIMD_INLINE __m256i ConditionalAbs<true>(__m256i a)
    {
        return _mm256_abs_epi16(a);
    }

    template <> SIMD_INLINE __m256i ConditionalAbs<false>(__m256i a)
    {
        return a;
    }

    template <int part> SIMD_INLINE __m256i UnpackU8(__m256i a, __m256i b = K_ZERO);

    template <> SIMD_INLINE __m256i UnpackU8<0>(__m256i a, __m256i b)
    {
        return _mm256_unpacklo_epi8(a, b);
    }

    template <> SIMD_INLINE __m256i UnpackU8<1>(__m256i a, __m256i b)
    {
        return _mm256_unpackhi_epi8(a, b);
    }

    template <int index> __m256i U8To16(__m256i a);

    template <> SIMD_INLINE __m256i U8To16<0>(__m256i a)
    {
        return _mm256_and_si256(a, K16_00FF);
    }

    template <> SIMD_INLINE __m256i U8To16<1>(__m256i a)
    {
        return _mm256_and_si256(_mm256_srli_si256(a, 1), K16_00FF);
    }

    template<int part> SIMD_INLINE __m256i SubUnpackedU8(__m256i a, __m256i b)
    {
        return _mm256_maddubs_epi16(UnpackU8<part>(a, b), K8_01_FF);
    }

    template <int part> SIMD_INLINE __m256i UnpackU16(__m256i a, __m256i b = K_ZERO);

    template <> SIMD_INLINE __m256i UnpackU16<0>(__m256i a, __m256i b)
    {
        return _mm256_unpacklo_epi16(a, b);
    }

    template <> SIMD_INLINE __m256i UnpackU16<1>(__m256i a, __m256i b)
    {
        return _mm256_unpackhi_epi16(a, b);
    }

    template<int shift> SIMD_INLINE __m256 Alignr(const __m256 & s0, const __m256 & s4)
    {
        return _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(s4), _mm256_castps_si256(s0), shift * 4));
    }

    template<int imm> SIMD_INLINE __m256i Shuffle32i(__m256i lo, __m256i hi)
    {
        return _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(lo), _mm256_castsi256_ps(hi), imm));
    }

    template<int imm> SIMD_INLINE __m256 Permute4x64(__m256 a)
    {
        return _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(a), imm));
    }

    template<int imm> SIMD_INLINE __m256 Shuffle32f(__m256 a)
    {
        return _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(a), imm));
    }

    template <int index> SIMD_INLINE __m256 Broadcast(__m256 a)
    {
        return _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(a), index * 0x55));
    }

    SIMD_INLINE __m256i Average16(const __m256i & a, const __m256i & b, const __m256i & c, const __m256i & d)
    {
        return _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(a, b), _mm256_add_epi16(c, d)), K16_0002), 2);
    }

    SIMD_INLINE __m256i Merge16(const __m256i & even, __m256i odd)
    {
        return _mm256_or_si256(_mm256_slli_si256(odd, 1), even);
    }

    SIMD_INLINE const __m256i Shuffle(const __m256i & value, const __m256i & shuffle)
    {
        return _mm256_or_si256(_mm256_shuffle_epi8(value, _mm256_add_epi8(shuffle, K8_SHUFFLE_0)),
            _mm256_shuffle_epi8(_mm256_permute4x64_epi64(value, 0x4E), _mm256_add_epi8(shuffle, K8_SHUFFLE_1)));
    }

#endif

    struct Buffer
    {
        Buffer(size_t width)
        {
            _p = Allocate(sizeof(uint16_t) * 3 * width);
            src0 = (uint16_t*)_p;
            src1 = src0 + width;
            src2 = src1 + width;
        }

        ~Buffer()
        {
            Free(_p);
        }

        uint16_t* src0;
        uint16_t* src1;
        uint16_t* src2;
    private:
        void* _p;
    };


    SIMD_INLINE __m256i DivideBy16(__m256i value)
    {
        return _mm256_srli_epi16(_mm256_add_epi16(value, K16_0008), 4);
    }

    const __m256i K8_01_02 = SIMD_MM256_SET2_EPI8(0x01, 0x02);

    template<int part> SIMD_INLINE __m256i BinomialSumUnpackedU8(__m256i a[3])
    {
        return _mm256_add_epi16(_mm256_maddubs_epi16(UnpackU8<part>(a[0], a[1]), K8_01_02), UnpackU8<part>(a[2]));
    }

    template<bool align> SIMD_INLINE void BlurCol(__m256i a[3], uint16_t* b)
    {
        Store<align>((__m256i*)b + 0, BinomialSumUnpackedU8<0>(a));
        Store<align>((__m256i*)b + 1, BinomialSumUnpackedU8<1>(a));
    }

    template<bool align> SIMD_INLINE __m256i BlurRow16(const Buffer& buffer, size_t offset)
    {
        return DivideBy16(BinomialSum16(
            Load<align>((__m256i*)(buffer.src0 + offset)),
            Load<align>((__m256i*)(buffer.src1 + offset)),
            Load<align>((__m256i*)(buffer.src2 + offset))));
    }

    template<bool align> SIMD_INLINE __m256i BlurRow(const Buffer& buffer, size_t offset)
    {
        return _mm256_packus_epi16(BlurRow16<align>(buffer, offset), BlurRow16<align>(buffer, offset + HA));
    }

    template <bool align, size_t step> void GaussianBlur3x3(
        const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
    {
        assert(step * (width - 1) >= A);
        if (align)
            assert(Aligned(src) && Aligned(srcStride) && Aligned(step * width) && Aligned(dst) && Aligned(dstStride));

        __m256i a[3];

        size_t size = step * width;
        size_t bodySize = Simd::AlignHi(size, A) - A;

        Buffer buffer(Simd::AlignHi(size, A));

        LoadNose3<align, step>(src + 0, a);
        BlurCol<true>(a, buffer.src0 + 0);
        for (size_t col = A; col < bodySize; col += A)
        {
            LoadBody3<align, step>(src + col, a);
            BlurCol<true>(a, buffer.src0 + col);
        }
        LoadTail3<align, step>(src + size - A, a);
        BlurCol<true>(a, buffer.src0 + bodySize);

        memcpy(buffer.src1, buffer.src0, sizeof(uint16_t) * (bodySize + A));

        for (size_t row = 0; row < height; ++row, dst += dstStride)
        {
            const uint8_t* src2 = src + srcStride * (row + 1);
            if (row >= height - 2)
                src2 = src + srcStride * (height - 1);

            LoadNose3<align, step>(src2 + 0, a);
            BlurCol<true>(a, buffer.src2 + 0);
            for (size_t col = A; col < bodySize; col += A)
            {
                LoadBody3<align, step>(src2 + col, a);
                BlurCol<true>(a, buffer.src2 + col);
            }
            LoadTail3<align, step>(src2 + size - A, a);
            BlurCol<true>(a, buffer.src2 + bodySize);

            for (size_t col = 0; col < bodySize; col += A)
                Store<align>((__m256i*)(dst + col), BlurRow<true>(buffer, col));
            Store<align>((__m256i*)(dst + size - A), BlurRow<true>(buffer, bodySize));

            Simd::Swap(buffer.src0, buffer.src2);
            Simd::Swap(buffer.src0, buffer.src1);
        }
    }

    template <bool align> void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t* dst, size_t dstStride)
    {
        assert(channelCount > 0 && channelCount <= 4);

        switch (channelCount)
        {
        case 1: GaussianBlur3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
        case 2: GaussianBlur3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
        case 3: GaussianBlur3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
        case 4: GaussianBlur3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
        }
    }

    void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t* dst, size_t dstStride)
    {
        if (Aligned(src) &&
            Aligned(srcStride) &&
            Aligned(channelCount * width) &&
            Aligned(dst) &&
            Aligned(dstStride))
            GaussianBlur3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
        else
            GaussianBlur3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
    }

#endif

#if 1

    template <bool align> SIMD_INLINE void BackgroundGrowRangeFast(const uint8_t * value, uint8_t * lo, uint8_t * hi)
    {
        const __m256i _value = Load<align>((__m256i*)value);
        const __m256i _lo = Load<align>((__m256i*)lo);
        const __m256i _hi = Load<align>((__m256i*)hi);

        Store<align>((__m256i*)lo, _mm256_min_epu8(_lo, _value));
        Store<align>((__m256i*)hi, _mm256_max_epu8(_hi, _value));
    }

    template <bool align> void BackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
    {
        assert(width >= A);
        if (align)
        {
            assert(Aligned(value) && Aligned(valueStride));
            assert(Aligned(lo) && Aligned(loStride));
            assert(Aligned(hi) && Aligned(hiStride));
        }

        size_t alignedWidth = Simd::AlignLo(width, A);
        for (size_t row = 0; row < height; ++row)
        {
            for (size_t col = 0; col < alignedWidth; col += A)
                BackgroundGrowRangeFast<align>(value + col, lo + col, hi + col);
            if (alignedWidth != width)
                BackgroundGrowRangeFast<false>(value + width - A, lo + width - A, hi + width - A);
            value += valueStride;
            lo += loStride;
            hi += hiStride;
        }
    }

    void BackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
    {
        if (Aligned(value) &&
            Aligned(valueStride) &&
            Aligned(lo) &&
            Aligned(loStride) &&
            Aligned(hi) &&
            Aligned(hiStride))
            BackgroundGrowRangeFast<true>(value, valueStride, width, height, lo, loStride, hi, hiStride);
        else
            BackgroundGrowRangeFast<false>(value, valueStride, width, height, lo, loStride, hi, hiStride);
    }

#endif
#if 1

    SIMD_INLINE __m256i FeatureDifference(__m256i value, __m256i lo, __m256i hi)
    {
        return _mm256_max_epu8(_mm256_subs_epu8(value, hi), _mm256_subs_epu8(lo, value));
    }

    SIMD_INLINE __m256i ShiftedWeightedSquare16(__m256i difference, __m256i weight)
    {
        return _mm256_mulhi_epu16(_mm256_mullo_epi16(difference, difference), weight);
    }

    SIMD_INLINE __m256i ShiftedWeightedSquare8(__m256i difference, __m256i weight)
    {
        const __m256i lo = ShiftedWeightedSquare16(_mm256_unpacklo_epi8(difference, K_ZERO), weight);
        const __m256i hi = ShiftedWeightedSquare16(_mm256_unpackhi_epi8(difference, K_ZERO), weight);
        return _mm256_packus_epi16(lo, hi);
    }

    template <bool align> SIMD_INLINE void AddFeatureDifference(const uint8_t * value, const uint8_t * lo, const uint8_t * hi,
        uint8_t * difference, size_t offset, __m256i weight, __m256i mask)
    {
        const __m256i _value = Load<align>((__m256i*)(value + offset));
        const __m256i _lo = Load<align>((__m256i*)(lo + offset));
        const __m256i _hi = Load<align>((__m256i*)(hi + offset));
        __m256i _difference = Load<align>((__m256i*)(difference + offset));

        const __m256i featureDifference = FeatureDifference(_value, _lo, _hi);
        const __m256i inc = _mm256_and_si256(mask, ShiftedWeightedSquare8(featureDifference, weight));
        Store<align>((__m256i*)(difference + offset), _mm256_adds_epu8(_difference, inc));
    }

    template <bool align> void AddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
        uint16_t weight, uint8_t * difference, size_t differenceStride)
    {
        assert(width >= A);
        if (align)
        {
            assert(Aligned(value) && Aligned(valueStride));
            assert(Aligned(lo) && Aligned(loStride));
            assert(Aligned(hi) && Aligned(hiStride));
            assert(Aligned(difference) && Aligned(differenceStride));
        }

        size_t alignedWidth = Simd::AlignLo(width, A);
        __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);
        __m256i _weight = _mm256_set1_epi16((short)weight);

        for (size_t row = 0; row < height; ++row)
        {
            for (size_t col = 0; col < alignedWidth; col += A)
                AddFeatureDifference<align>(value, lo, hi, difference, col, _weight, K_INV_ZERO);
            if (alignedWidth != width)
                AddFeatureDifference<false>(value, lo, hi, difference, width - A, _weight, tailMask);
            value += valueStride;
            lo += loStride;
            hi += hiStride;
            difference += differenceStride;
        }
    }

    void AddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
        uint16_t weight, uint8_t * difference, size_t differenceStride)
    {
        if (Aligned(value) &&
            Aligned(valueStride) &&
            Aligned(lo) &&
            Aligned(loStride) &&
            Aligned(hi) &&
            Aligned(hiStride) &&
            Aligned(difference) &&
            Aligned(differenceStride))
            AddFeatureDifference<true>(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
        else
            AddFeatureDifference<false>(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    }

#endif

}


void SimdGaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    size_t channelCount, uint8_t * dst, size_t dstStride)
{
    AVX2::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
    uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
{
    AVX2::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
}


void SimdAddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
    const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
    uint16_t weight, uint8_t * difference, size_t differenceStride)
{
    AVX2::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
}

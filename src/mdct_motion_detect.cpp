#include "mdct_motion_detect.hpp"
#include "simd_ops.h"
/*extern "C" {
#include "simd_interface.h"
}*/

#define DYNAMIC 1
#define MASK_DIFF_INDEX 0
#define SIMILAR_COUNT_START 100
#define SIMILAR_COUNT_END 125

void inline copy_replace(uint8_t* pdst, uint8_t* psrc, bool halfres, uint8_t r0, uint32_t size) {
    uint8_t v;
    if (halfres) {
        for (; size; size--, psrc += 2, pdst++) {
            v = *psrc;
            if (!v || v > r0) {
                *pdst = r0;
            }
        }
    } else {
        for (; size; size--, psrc++, pdst++) {
            v = *psrc;
            if (!v || v > r0) {
                *pdst = r0;
            }
        }
    }
    
}

#define COLLAPSE_SPEED 33

uint32_t inline umin(uint32_t a, uint32_t b) {
    if(a < b)
        return a;
    return b;
}

void inline crop2zone(uint32_t x1, uint32_t x2, uint32_t y1, uint32_t y2, uint32_t w, uint32_t h, ZoneCrop& zone) {
    zone.x1 = x1 < zone.x1 ? x1 : zone.x1 + umin(x1 - zone.x1, COLLAPSE_SPEED);
    zone.x2 = x2 > zone.x2 ? x2 : zone.x2 - umin(zone.x2 - x2, COLLAPSE_SPEED);
    zone.y1 = y1 < zone.y1 ? y1 : zone.y1 + umin(y1 - zone.y1, COLLAPSE_SPEED);
    zone.y2 = y2 > zone.y2 ? y2 : zone.y2 - umin(zone.y2 - y2, COLLAPSE_SPEED);
}

void CMotionZone::mask_plane() {
    memset(m_mask_Y, m_threshold, m_buffer_size);
    memset(m_mask_UV, m_threshold, m_buffer_size >> 1);
    memset(m_mask_diff_Y, MASK_DIFF_INDEX, m_buffer_size);
    memset(m_mask_diff_UV, MASK_DIFF_INDEX, m_buffer_size >> 1);
    if (m_mask_source) {
        uint32_t lsz = m_linesize[0] < m_mask_w ? m_linesize[0] : m_mask_w;
        uint32_t copy_h = m_height < m_mask_h ? m_height : m_mask_h;
        uint8_t* psrc = (uint8_t*)m_mask_source;
        uint8_t* pdst1 = (uint8_t*)m_mask_Y;
        uint8_t* pdst3 = (uint8_t*)m_mask_diff_Y;
        for (uint32_t i = 0; i < copy_h; ++i) {
            copy_replace(pdst1, psrc, false, 127, lsz);
            copy_replace(pdst3, psrc, false, 127, lsz);
            psrc += m_mask_w;
            pdst1 += m_linesize[0];
            pdst3 += m_linesize[0];
        }
        // Half-res
        lsz = lsz >> 1;
        copy_h = copy_h >> 1;
        psrc = (uint8_t*)m_mask_source;
        uint8_t* pdst2 = (uint8_t*)m_mask_UV;
        uint8_t* pdst4 = (uint8_t*)m_mask_diff_UV;
        for (uint32_t i = 0; i < copy_h; ++i) {
            copy_replace(pdst2, psrc, true, 127, lsz);
            copy_replace(pdst4, psrc, true, 127, lsz);
            psrc += m_mask_w;
            psrc += m_mask_w;
            pdst2 += m_linesize[2];
            pdst4 += m_linesize[2];
        }
    } else {
        // Fill lower 16 lines with masking values
        void* p;
        p = (void*)((uint64_t)m_mask_Y + (m_height - 16)*m_linesize[0]);
        memset(p, 127, 16*m_linesize[0]);
        p = (void*)((uint64_t)m_mask_UV + (m_height - 16)*m_linesize[2]);
        memset(p, 127, 16*m_linesize[2]);
        p = (void*)((uint64_t)m_mask_diff_Y + (m_height - 16)*m_linesize[0]);
        memset(p, 127, 16*m_linesize[0]);
        p = (void*)((uint64_t)m_mask_diff_UV + (m_height - 16)*m_linesize[2]);
        memset(p, 127, 16*m_linesize[2]);
    }
    
}
void CMotionZone::mask_update(gs_image_file_t* mask) {
    if (m_mask_source) {
        bfree(m_mask_source);
        m_mask_source = 0;
    }
    if (mask && mask->loaded) {
        m_mask_w = mask->cx;
        m_mask_h = mask->cy;
        m_mask_source = bzalloc(m_mask_w * m_mask_h);
        uint8_t* src = mask->texture_data;
        uint8_t* dst = (uint8_t*)m_mask_source;
        // Copy B from BGRA
        for (uint32_t y = 0; y < m_mask_h; ++y) {
            for (uint32_t x = 0; x < m_mask_w; ++x) {
                *dst = *src;
                dst++;
                src += 4;
            }
        }
        //memcpy(mask_source, mask->texture_data, mask_w * mask_h);
        mask_plane();
    }
}
void CMotionZone::threshold_set(uint8_t thr) {
    if (thr) {
        m_threshold = thr;
        mask_plane();
    }
}
bool CMotionZone::check_init(obs_source_frame* f) {
    if (f && f->data && f->data[0]) {
        if (f->format != m_format) {
            m_format = f->format;
            m_width = f->width;
            m_height = f->height;
            m_zone.x1 = 0;
            m_zone.x2 = m_width;
            m_zone.y1 = 0;
            m_zone.y2 = m_height;
            m_linesize[0] = f->linesize[0];
            m_linesize[1] = f->linesize[1];
            m_linesize[2] = f->linesize[2];
            if (m_buffers[0]) {
                bfree(m_buffers[0]);
                m_buffers[0] = 0;
            }
        }
        if (!m_buffers[0]) {
            m_buffer_size = m_linesize[0] * m_height;
            m_buffers[0] =  bzalloc(16 * m_buffer_size);                                // 0  Plane Y store buffer (current frame)
            m_buffers[1] =  (void*)((uint64_t)m_buffers[0]  + m_buffer_size);           // 1  Plane Y background lo
            m_buffers[2] =  (void*)((uint64_t)m_buffers[1]  + m_buffer_size);           // 2  Plane Y background hi
            m_buffers[3] =  (void*)((uint64_t)m_buffers[2]  + m_buffer_size);           // 3  Plane Y feature diff
            m_buffers[4] =  (void*)((uint64_t)m_buffers[3]  + m_buffer_size);           // 4  Plane Y temp buffer (current frame)
            m_buffers[5] =  (void*)((uint64_t)m_buffers[4]  + m_buffer_size);           // 5  Plane Y background lo temp
            m_buffers[6] =  (void*)((uint64_t)m_buffers[5]  + m_buffer_size);           // 6  Plane Y background hi temp
            m_buffers[7] =  (void*)((uint64_t)m_buffers[6]  + m_buffer_size);           // 7  Plane U/V background lo
            m_buffers[8] =  (void*)((uint64_t)m_buffers[7]  + (m_buffer_size >> 1));    // 8  Plane U/V background hi
            m_buffers[9] =  (void*)((uint64_t)m_buffers[8]  + (m_buffer_size >> 1));    // 9  Plane U/V feature diff
            m_buffers[10] = (void*)((uint64_t)m_buffers[9]  + (m_buffer_size >> 1));    // 10 Plane U/V background hi temp
            m_buffers[11] = (void*)((uint64_t)m_buffers[10] + (m_buffer_size >> 1));    // 11 Plane U/V background hi temp
            m_mask_Y =       (void*)((uint64_t)m_buffers[11]   + (m_buffer_size >> 1)); // Plane Y mask for zone detection
            m_mask_UV =      (void*)((uint64_t)m_mask_Y        + m_buffer_size);        // Plane U/V mask for zone detection
            m_mask_diff_Y =  (void*)((uint64_t)m_mask_UV       + (m_buffer_size >> 1)); // Plane Y mask for motion detection
            m_mask_diff_UV = (void*)((uint64_t)m_mask_diff_Y   + m_buffer_size);        // Plane Y mask for motion detection
            m_poutput =      (void*)((uint64_t)m_mask_diff_UV  + (m_buffer_size >> 1));
            m_buffer_index = 0;
            mask_plane();
        }
    }
    return m_buffers[0] != 0;
}
#if DYNAMIC
void CMotionZone::detect_y() {
    uint8_t* c = (uint8_t*)m_buffers[9];
    uint32_t offset = 0;
    uint32_t x1cb = m_width >> 1, x2cb = 0, y1cb = m_height >> 1, y2cb = 0;
    for (uint32_t y = 0; y < m_height >> 1; ++y, offset += m_linesize[2]) {
        if (avx2_mask_detect_uint8(c + offset, (uint8_t*)m_mask_UV + offset, m_width >> 1, x1cb, x2cb)) {
            if (y < y1cb) {
                y1cb = y;
            }
            y2cb = y;
        }
    }
    // Check result and take smaller area
    if (m_x1ab < m_x2ab && x1cb < x2cb && m_y1ab < m_y2ab && y1cb < y2cb) {
        uint32_t sab = (m_x2ab - m_x1ab) * (m_y2ab - m_y1ab);
        uint32_t scb = (x2cb - x1cb) * (y2cb - y1cb);
        if (sab < scb) {
                // ab -> zone
                crop2zone(m_x1ab << 1, m_x2ab << 1, m_y1ab << 1, m_y2ab << 1, m_width, m_height, m_zone);
        } else {
                // cb -> zone
                crop2zone(x1cb << 1, x2cb << 1, y1cb << 1, y2cb << 1, m_width, m_height, m_zone);
        }
    }
    // Store C-B crop
    m_x1ab = x1cb; m_x2ab = x2cb; m_y1ab = y1cb; m_y2ab = y2cb;
}
void CMotionZone::feed_y(obs_source_frame* f, texture_position_t* texp) {
    if (check_init(f)) {
        double ssd = 0.0;
        // HALF-RESOLUTION PROCESSING!!!
        SimdGaussianBlur3x3(
            (const uint8_t*)f->data[2], m_linesize[2] << 1,
            m_width >> 1, m_height >> 1, 1,
            (uint8_t*)m_buffers[4], m_linesize[2]
        );
        bool motion = plane_diff_mask_detect(
            (uint8_t*)m_buffers[4],
            (uint8_t*)m_buffers[0],
            (uint8_t*)m_mask_diff_UV,
            m_linesize[2],
            m_width >> 1, m_height >> 1, m_ssd_threshold,
            &ssd
        );
        memcpy(m_buffers[0], m_buffers[4], (m_height >> 1) * m_linesize[2]);

        memset(m_buffers[9], 0, (m_height >> 1) * m_linesize[2]);
        SimdAddFeatureDifference(
            (const uint8_t*)m_buffers[0], m_linesize[2],
            m_width, m_height,
            (const uint8_t*)m_buffers[7], m_linesize[2],
            (const uint8_t*)m_buffers[8], m_linesize[2],
            0x8000,
            (uint8_t*)m_buffers[9], m_linesize[2]
        );
        detect_y();
        zonecrop2textureposition(texp);
        // For visual testing:
        //memcpy(f->data[0], poutput, linesize[0] * height);
        draw_zone_y(f);

        if (ssd > 0.000000001) {
            // Update background
            if(!motion){
                m_similar_count++;
                if (m_similar_count == SIMILAR_COUNT_START) {
                    // Initialize background capture U/V plane
                    memcpy(m_buffers[10], m_buffers[0], m_height * m_linesize[2]);
                    memcpy(m_buffers[11], m_buffers[0], m_height * m_linesize[2]);
                } else if (m_similar_count > SIMILAR_COUNT_START) {
                    if (m_similar_count < SIMILAR_COUNT_END) {
                        // Update background ranges U/V
                        SimdBackgroundGrowRangeFast(
                            (const uint8_t*)m_buffers[0], m_linesize[2],
                            m_width >> 1, m_height >> 1,
                            (uint8_t*)m_buffers[10], m_linesize[2],
                            (uint8_t*)m_buffers[11], m_linesize[2]
                        );
                    } else if (m_similar_count == SIMILAR_COUNT_END) {
                        // Use background U/V
                        memcpy(m_buffers[7], m_buffers[10], m_height * m_linesize[2]);
                        memcpy(m_buffers[8], m_buffers[11], m_height * m_linesize[2]);
                    }
                }
            } else {
                m_similar_count = 0;
            }
        }
    }
}
#else
void CMotionZone::detect_y() {
    // Scan
    uint8_t* b = (uint8_t*)buffers[buffer_index];
    uint8_t* c = (uint8_t*)buffers[2];

    uint32_t offset = 0;
    uint32_t x1cb = width, x2cb = 0, y1cb = height, y2cb = 0;
    for (uint32_t y = 0; y < height; ++y, offset += linesize[0]) {
        if (avx2_sub_mask_detect_uint8(
            c + offset, b + offset,
            (uint8_t*)mask_planed + offset,
            (uint8_t*)poutput + offset,
            width, x1cb, x2cb)) {
            if (y < y1cb) {
                y1cb = y;
            }
            y2cb = y;
        }
    }
    if (x1ab < x2ab && x1cb < x2cb && y1ab < y2ab && y1cb < y2cb) {
        uint32_t sab = (x2ab - x1ab) * (y2ab - y1ab);
        uint32_t scb = (x2cb - x1cb) * (y2cb - y1cb);
        if (sab < scb) {
            // ab -> zone
            crop2zone(x1ab, x2ab, y1ab, y2ab, width, height, zone);
        } else {
            // cb -> zone
            crop2zone(x1cb, x2cb, y1cb, y2cb, width, height, zone);
        }
    }
    // Store C-B crop
    x1ab = x1cb; x2ab = x2cb; y1ab = y1cb; y2ab = y2cb;
}
void CMotionZone::feed_y(obs_source_frame* f, texture_position_t* texp) {
    if (check_init(f)) {
        // Blur Y plane and store it
        SimdGaussianBlur3x3(
            (const uint8_t*)f->data[0], linesize[0],
            width, height, 1,
            (uint8_t*)buffers[1], linesize[0]
        );
        SimdGaussianBlur3x3(
            (const uint8_t*)buffers[1], linesize[0],
            width, height, 1,
            (uint8_t*)buffers[0], linesize[0]
        );
        //memcpy(buffers[buffer_index], f->data[0], linesize[0] * height);
        if (capture) {
            // Capture sample
            capture = false;
            memcpy(buffers[2], buffers[0], buffer_size);
        }
        detect_y();
        //draw_zone_y(f);
        plane_blend_avx2((uint8_t*)buffers[2], (uint8_t*)buffers[0], linesize[0], width, height, 240, 16);
        // For visual testing:
        //memcpy(f->data[0], buffers[2], linesize[0] * height);
        draw_zone_y(f);
        counter++;
        if (counter == 200) {
            counter = 0;
            capture = true;
        }
        //buffer_index = (buffer_index + 1) % 2;
        zonecrop2textureposition(texp);
    }
}
#endif
void line_horz(uint32_t y, uint8_t* plane, uint32_t stride, uint32_t width, uint32_t height) {
    plane += y * stride;
    memset(plane, 255, width);
    plane += stride;
    memset(plane, 255, width);
}
void line_vert(uint32_t x, uint8_t* plane, uint32_t stride, uint32_t width, uint32_t height) {
    uint8_t* p = plane + x;
    for (uint32_t y = 0; y < height; ++y) {
        p[0] = 255;
        p[1] = 255;
        p += stride;
    }
}
void CMotionZone::draw_zone_y(obs_source_frame* f) {
    line_horz(m_zone.y1, f->data[0], f->linesize[0], m_width, m_height);
    line_horz(m_zone.y2 - 2, f->data[0], f->linesize[0], m_width, m_height);
    line_vert(m_zone.x1, f->data[0], f->linesize[0], m_width, m_height);
    line_vert(m_zone.x2 - 2, f->data[0], f->linesize[0], m_width, m_height);
}
/*void CMotionZone::process_y(obs_source_frame* f1, obs_source_frame* f2) {
    if (check_init(f1, f2)) {

        //memcpy(buffer_diff, f1->data[0], linesize[0] * height);

        plane_diff_blur_u8(f1->data[0], f2->data[0], (uint8_t*)buffer_diff, (uint8_t*)buffer_blur, linesize[0], width, height);
        // For visual testing:
        //memcpy(f1->data[0], buffer_diff, linesize[0] * height);
    }
}
void CMotionZone::process_u(obs_source_frame* f1, obs_source_frame* f2) {
    if (check_init(f1, f2)) {
        plane_diff_blur_i8(
            (int8_t*)f1->data[1], (int8_t*)f2->data[1],
            (int8_t*)buffer_diff, (int8_t*)buffer_blur,
            width >> 1, height,
            linesize[1]
        );
        // For visual testing:
        memcpy(f1->data[1], buffer_blur, linesize[1] * (height));
    }
}
void CMotionZone::process_v(obs_source_frame* f1, obs_source_frame* f2) {
    if (check_init(f1, f2)) {
        plane_diff_blur_i8(
            (int8_t*)f1->data[2], (int8_t*)f2->data[2],
            (int8_t*)buffer_diff, (int8_t*)buffer_blur,
            width >> 1, height,
            linesize[2]
        );
        // For visual testing:
        memcpy(f1->data[2], buffer_blur, linesize[2] * (height));
    }
}*/


void inline strict_texp(texture_position_t* tp) {
    if (tp->offset.x < 0.0f) {
        tp->offset.x = 0.0f;
    } else {
        float mdiff = tp->offset.x + tp->multiply.x - 1.0f;
        if (mdiff > 0.0f) {
            tp->offset.x -= mdiff;
        }
    }
    if (tp->offset.y < 0.0f) {
        tp->offset.y = 0.0f;
    } else {
        float mdiff = tp->offset.y + tp->multiply.y - 1.0f;
        if (mdiff > 0.0f) {
            tp->offset.y -= mdiff;
        }
    }
}

void inline conform_texp(texture_position_t* tp, float mult_min) {
    if (tp->multiply.x < mult_min) {
        tp->offset.x -= 0.5f * (mult_min - tp->multiply.x);
        tp->multiply.x = mult_min;
    } else if (tp->multiply.x > 1.0f) {
        tp->offset.x = 0.0f;
        tp->multiply.x = 1.0f;
    }
    if (tp->multiply.y < mult_min) {
        tp->offset.y -= 0.5f * (mult_min - tp->multiply.y);
        tp->multiply.y = mult_min;
    } else if (tp->multiply.y > 1.0f) {
        tp->offset.y = 0.0f;
        tp->multiply.y = 1.0f;
    }
    float mdiff = 0.5f * f_abs(tp->multiply.x - tp->multiply.y);
    if (mdiff > 0.0001f) {
        if (tp->multiply.x < tp->multiply.y) {
            tp->offset.x -= mdiff;
            tp->multiply.x = tp->multiply.y;
        } else {
            tp->offset.y -= mdiff;
            tp->multiply.y = tp->multiply.x;
        }
    }
}

void inline conform_bound(float& v, float g) {
    if (v < g) {
        v = g;
    } else if (v > (1.0f - g)) {
        v = 1.0f - g;
    }
}

void inline conform_cp(center_position_t* cp, float mult_min) {
    cp->scale = f_min(1.0f, f_max(mult_min, cp->scale));
    float gap = 0.5f * cp->scale;
    conform_bound(cp->cx, gap);
    conform_bound(cp->cy, gap);
}

/*float process_accel(
    float current_v, float target_v,
    float speed,
    float accel_pos, float accel_neg,
    float speed_pos, float speed_neg,
    float dev_pos, float dev_neg,
    bool &brake
    )
{
    accel_pos = (1.0f - dev_pos) * (1.0f - dev_pos) * accel_pos;
    accel_neg = (1.0f - dev_neg) * (1.0f - dev_neg) * accel_neg;

    float crowling_speed = 0.005f * (target_v - current_v);
    float planned_v;
    float accel = 0.0f;

    if (brake) {
        if (speed > 0.0f) {
            accel = -2.0f*accel_pos;
            if (speed + accel <= 0.0f) {
                brake = false;
            }
        } else if (speed < 0.0f) {
            accel = 2.0f*accel_neg;
            if (speed + accel >= 0.0f) {
                brake = false;
            }
        } else {
            brake = false;
        }
    } else {
        if (target_v > current_v) {
            // Positive movement
            planned_v = current_v + dev_pos;
            if (speed > speed_pos) {
                // Limit speed
                accel = -accel_neg;
            } else if (target_v > planned_v) {
                // Accelerate
                accel = accel_pos;
            } else {
                // Brake (use inverted positive accel)
                if (speed >= 0.0f) {
                    if (speed > crowling_speed) {
                        accel = -f_min(speed - crowling_speed, accel_pos);
                    } else {
                        accel = crowling_speed;// - speed_x;
                    }
                } else {
                    accel = accel_pos;
                    brake = true;
                }
            }
            // Tune accel
            if (accel > 0.0f && speed > 0.0f) {
                accel = accel * (1.0f - 0.9f * speed / speed_pos);
            }
        } else {
            // Negative movement
            planned_v = current_v - dev_neg;
            if (speed < -speed_neg) {
                // Limit speed
                accel = accel_pos;
            } else if (target_v < planned_v) {
                // Accelerate
                accel = -accel_neg;
            } else {
                // Brake (use inverted negative accel)
                if (speed > 0.0f) {
                    accel = -accel_neg;
                    brake = true;
                } else {
                    if (speed < crowling_speed) {  // Note that crowling_speed and speed_x are negative here
                        accel = f_min(crowling_speed - speed, accel_neg);
                    } else {
                        accel = crowling_speed;// - speed_x;
                    }
                }
            }
            // Tune accel
            if (accel < 0.0f && speed < 0.0f) {
                accel = accel * (1.0f + 0.9f * speed / speed_neg);
            }
        }
    }
    return accel;
}*/

float process_accel(
    float current_v, float target_v,
    float speed,
    float accel_pos, float accel_neg,
    float speed_pos, float speed_neg,
    float dev_pos, float dev_neg,
    bool &brake
)
{
    float crowling_speed = 0.01f * (target_v - current_v);
    float planned_v;
    float accel = 0.0f;
    float tune = 2.0f * sqrtf((current_v - target_v) * (current_v - target_v));

    if (brake) {
        if (speed > 0.0f) {
            accel = -2.0f*accel_pos * tune;
            if (speed + accel <= 0.0f) {
                brake = false;
            }
        } else if (speed < 0.0f) {
            accel = 2.0f*accel_neg * tune;
            if (speed + accel >= 0.0f) {
                brake = false;
            }
        } else {
            brake = false;
        }
    } else {
        if (target_v > current_v) {
            // Positive movement
            planned_v = current_v + dev_pos;
            if (speed > speed_pos) {
                // Limit speed
                accel = -accel_neg * tune;
            } else if (target_v > planned_v) {
                // Accelerate
                accel = accel_pos * tune;
            } else {
                // Brake (use inverted positive accel)
                if (speed >= 0.0f) {
                    if (speed > crowling_speed) {
                        accel = -f_min(speed - crowling_speed, accel_pos);
                    } else {
                        accel = crowling_speed;// - speed_x;
                    }
                } else {
                    //accel = accel_pos;
                    brake = true;
                }
            }
            // Tune accel
            /*if (accel > 0.0f && speed > 0.0f) {
            accel = accel * (1.0f - 0.9f * speed / speed_pos);
            }*/
        } else {
            // Negative movement
            planned_v = current_v - dev_neg;
            if (speed < -speed_neg) {
                // Limit speed
                accel = accel_pos * tune;
            } else if (target_v < planned_v) {
                // Accelerate
                accel = -accel_neg * tune;
            } else {
                // Brake (use inverted negative accel)
                if (speed > 0.0f) {
                    //accel = -accel_neg;
                    brake = true;
                } else {
                    if (speed < crowling_speed) {  // Note that crowling_speed and speed_x are negative here
                        accel = f_min(crowling_speed - speed, accel_neg);
                    } else {
                        accel = crowling_speed;// - speed_x;
                    }
                }
            }
            // Tune accel
            /*if (accel < 0.0f && speed < 0.0f) {
            accel = accel * (1.0f + 0.9f * speed / speed_neg);
            }*/
        }
    }
    return accel;
}


void CropTrack::tick(texture_position_t* tp, float seconds) {
    // Current boundaries:
    // texp.offest.x to texp.offest.x + texp.multiply.x
    // texp.offest.y to texp.offest.y + texp.multiply.y

    // Calc target texp
    //texture_position_t texp_conformed;
    
    float dx1, dx2;
    float dy1, dy2;
    // Conform input
    float mult_min = 1.0f / max_scale;
    float accel_tune;

    center_position_t current_cp, target_cp, target_cpg;
    tp2cp(&texp, &current_cp);
    tp2cp(tp, &target_cp);
    target_cpg.cx = target_cp.cx;
    target_cpg.cy = target_cp.cy;
    target_cpg.scale = 1.2f * target_cp.scale;
    conform_cp(&target_cp, mult_min);
    conform_cp(&target_cpg, mult_min);

    dx1 = 0.5f * f_max(0.0f, tp->offset.x - f_max(target_cp.cx - 0.5f*target_cp.scale, target_cpg.cx - 0.5f*target_cpg.scale));
    dy1 = 0.5f * f_max(0.0f, tp->offset.y - f_max(target_cp.cy - 0.5f*target_cp.scale, target_cpg.cy - 0.5f*target_cpg.scale));
    dx2 = 0.5f * f_max(0.0f, f_min(target_cp.cx + 0.5f*target_cp.scale, target_cpg.cx + 0.5f*target_cpg.scale) - (tp->offset.x + tp->multiply.x));
    dy2 = 0.5f * f_max(0.0f, f_min(target_cp.cy + 0.5f*target_cp.scale, target_cpg.cy + 0.5f*target_cpg.scale) - (tp->offset.y + tp->multiply.y));


    // Add zoom gaps
    /*texp_conformed.multiply.x = f_max(mult_min, 1.2f * tp->multiply.x);
    texp.offset.x -= 0.5f * (texp_conformed.multiply.x - tp->multiply.x);
    texp_conformed.multiply.y = f_max(mult_min, 1.2f * tp->multiply.y);
    texp.offset.y -= 0.5f * (texp_conformed.multiply.y - tp->multiply.y);
    conform_texp(&texp_conformed, mult_min);
    strict_texp(&texp_conformed);
    float target_x = texp_conformed.offset.x + 0.5f * texp_conformed.multiply.x;
    float target_y = texp_conformed.offset.y + 0.5f * texp_conformed.multiply.y;
    float gap;
    gap = 0.2f * texp_conformed.multiply.x;
    dx1 = f_max(0.0f, tp->offset.x - texp_conformed.offset.x - gap);
    dx2 = f_max(0.0f, texp_conformed.offset.x + texp_conformed.multiply.x - (tp->offset.x + tp->multiply.x) - gap);
    gap = 0.2f * texp_conformed.multiply.y;
    dy1 = f_max(0.0f, tp->offset.y - texp_conformed.offset.y - gap);
    dy2 = f_max(0.0f, texp_conformed.offset.y + texp_conformed.multiply.y - (tp->offset.y + tp->multiply.y) - gap);*/
    
    //float target_v;
    //float current_x, current_y, current_z;
    float accel_pos;
    float accel_neg;
    float speed_pos;
    float speed_neg;
    float ax, ay, az;

    //current_x = texp.offset.x + 0.5f * texp.multiply.x;
    //target_v = tp->offset.x + 0.5f * tp->multiply.x;
    accel_pos = accel_horiz * SCALE_PARAMS;
    accel_neg = accel_horiz * SCALE_PARAMS;
    speed_pos = speed_horiz * SCALE_PARAMS;
    speed_neg = speed_horiz * SCALE_PARAMS;

    // TODO: ADD BRAKE MODE!!!
    ax = process_accel(
        current_cp.cx, target_cpg.cx,
        speed_x,
        accel_pos, accel_neg,
        speed_pos, speed_neg,
        dx2, dx1,
        brake_x
    );

    speed_x += ax;
    current_cp.cx += speed_x;

    //current_y = texp.offset.y + 0.5f * texp.multiply.y;
    //target_v = tp->offset.y + 0.5f * tp->multiply.y;
    accel_pos = accel_down * SCALE_PARAMS;
    accel_neg = accel_up * SCALE_PARAMS;
    speed_pos = speed_down * SCALE_PARAMS;
    speed_neg = speed_up * SCALE_PARAMS;

    ay = process_accel(
        current_cp.cy, target_cpg.cy,
        speed_y,
        accel_pos, accel_neg,
        speed_pos, speed_neg,
        dy2, dy1,
        brake_y
    );

    speed_y += ay;
    current_cp.cy += speed_y;

    //current_z = texp.multiply.x;
    //target_v = tp->multiply.x;
    accel_pos = accel_zoom_out * SCALE_PARAMS;
    accel_neg = accel_zoom_in * SCALE_PARAMS;
    speed_pos = speed_zoom_out * SCALE_PARAMS;
    speed_neg = speed_zoom_in * SCALE_PARAMS;

    az = process_accel(
        current_cp.scale, target_cpg.scale,
        speed_z,
        accel_pos, accel_neg,
        speed_pos, speed_neg,
        0.2f, 0.1f,
        brake_z
    );

    speed_z += az;
    current_cp.scale += speed_z;

    conform_cp(&current_cp, mult_min);
    cp2tp(&current_cp, &texp);
    /*texp.offset.x = current_x - 0.5f * current_z;
    texp.offset.y = current_y - 0.5f * current_z;
    texp.multiply.x = current_z;
    texp.multiply.y = current_z;
    conform_texp(&texp, mult_min);
    strict_texp(&texp);*/

    //texp.multiply = texp_conformed.multiply;
    //texp.offset = texp_conformed.offset;
    //texp.multiply.y = 0.25f;
    //texp.offset.y = 0.75f;
}

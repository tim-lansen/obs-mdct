#pragma once
extern "C" {
#include <obs-module.h>
#include <graphics/image-file.h>
}

#define SCALE_PARAMS 0.01f

float inline f_abs(float v) {
    if (v >= 0.0f)
        return v;
    return -v;
}

float inline f_min(float a, float b) {
    if (a > b)
        return b;
    return a;
}

float inline f_max(float a, float b) {
    if (a > b)
        return a;
    return b;
}

typedef struct {
    uint32_t x1, x2, y1, y2;
} ZoneCrop;

typedef struct {
    vec2 multiply;
    vec2 offset;
} TexturePosition;

typedef struct {
    float cx, cy, scale;
} CenterPosition;

typedef TexturePosition texture_position_t;
typedef CenterPosition center_position_t;

void inline tp2cp(texture_position_t* tp, center_position_t* cp) {
    cp->cx = tp->offset.x + 0.5f * tp->multiply.x;
    cp->cy = tp->offset.y + 0.5f * tp->multiply.y;
    cp->scale = f_max(tp->multiply.x, tp->multiply.y);
}

void inline cp2tp(center_position_t* cp, texture_position_t* tp) {
    tp->multiply.x = cp->scale;
    tp->multiply.y = cp->scale;
    tp->offset.x = cp->cx - 0.5f*cp->scale;
    tp->offset.y = cp->cy - 0.5f*cp->scale;
}

void inline tp_reset(texture_position_t* texp) {
    texp->multiply.x = 1.0f;
    texp->multiply.y = 1.0f;
    texp->offset.x = 0.0f;
    texp->offset.y = 0.0f;
}

class CMotionZone {
public:
    CMotionZone()
        : m_mask_source(0)
        , m_mask_Y(0)
        , m_mask_UV(0)
        , m_mask_diff_Y(0)
        , m_mask_diff_UV(0)
        , m_similar_count(0)
        , m_ssd_threshold(5.0)
        , m_capture(false)
        , m_counter(0)
        , m_mask_w(0), m_mask_h(0)
        , m_width(0)
        , m_height(0)
        , m_buffer_size(0)
        , m_buffer_index(0)
        , m_format(VIDEO_FORMAT_NONE)
        , m_threshold(50)
        , m_x1ab(1), m_x2ab(0), m_y1ab(1), m_y2ab(0)
    {
        m_linesize[0] = 0;
        m_linesize[1] = 0;
        m_linesize[2] = 0;
        m_linesize[3] = 0;
        m_buffers[0] = 0;
        m_buffers[1] = 0;
        m_buffers[2] = 0;
        m_buffers[3] = 0;
        m_buffers[4] = 0;
        m_buffers[5] = 0;
        m_buffers[6] = 0;
        m_buffers[7] = 0;
        m_buffers[8] = 0;
        m_buffers[9] = 0;
        m_buffers[10] = 0;
        m_buffers[11] = 0;
        m_poutput = 0;
        m_zone.x1 = 0;
        m_zone.x2 = 0;
        m_zone.y1 = 0;
        m_zone.y2 = 0;
    }
    ~CMotionZone(){
        if (m_buffers[0]) {
            bfree(m_buffers[0]);
        }
        if (m_mask_source) {
            bfree(m_mask_source);
        }
        /*if (mask_planed) {
            bfree(mask_planed);
        }*/
    }
    void mask_plane();
    void mask_update(gs_image_file_t* mask);
    void threshold_set(uint8_t thr);
    bool check_init(obs_source_frame* f);
    void feed_y(obs_source_frame* f, texture_position_t* texp);
    void draw_zone_y(obs_source_frame* f);
    void detect_y();
    video_format m_format;
    uint8_t m_threshold;
    uint8_t m_buffer_index;
    size_t m_buffer_size;
    void* m_buffers[12];
    void* m_poutput;
    void* m_mask_source;
    void* m_mask_Y;
    void* m_mask_UV;
    void* m_mask_diff_Y;
    void* m_mask_diff_UV;
    uint32_t m_similar_count;
    double m_ssd_threshold;
    bool m_capture;
    uint32_t m_counter;
    uint32_t m_mask_w, m_mask_h;
    uint32_t m_width, m_height, m_linesize[4];
    uint32_t m_x1ab, m_x2ab, m_y1ab, m_y2ab;
    ZoneCrop m_zone;
    //TexturePosition texp;
    void zonecrop2textureposition(texture_position_t* texp) {
        if (m_width && m_height) {
            float w = (float)m_width;
            float h = (float)m_height;
            texp->multiply.x = (float)(m_zone.x2 - m_zone.x1) / w;
            texp->multiply.y = (float)(m_zone.y2 - m_zone.y1) / h;
            texp->offset.x = (float)m_zone.x1 / w;
            texp->offset.y = (float)m_zone.y1 / h;
        }
    }
};


class CropTrack {
public:
    CropTrack()
        : accel_horiz    (0.1f)
        , speed_horiz    (0.5f)
        , accel_down     (0.3f)
        , speed_down     (1.0f)
        , accel_up       (0.1f)
        , speed_up       (0.4f)
        , accel_zoom_in  (0.1f)
        , speed_zoom_in  (0.2f)
        , accel_zoom_out (0.2f)
        , speed_zoom_out (0.5f)
        , center_dev     (0.1f)
        , max_scale (2.2f)
        , speed_x (0.0f)
        , speed_y (0.0f)
        , speed_z (0.0f)
        , brake_x (false)
        , brake_y (false)
        , brake_z (false)
    {
        tp_reset(&texp);
    }
    void tick(texture_position_t* tp, float seconds);
    void process_scale(bool slow);
    texture_position_t texp;
    // Setup
    float accel_horiz, accel_down, accel_up, accel_zoom_in, accel_zoom_out;
    float speed_horiz, speed_down, speed_up, speed_zoom_in, speed_zoom_out;
    float center_dev;
    float max_scale;
    // Dynamic
    float speed_x, speed_y, speed_z;
    bool brake_x, brake_y, brake_z;
};

extern "C" {
#include <graphics/vec2.h>
#include <graphics/image-file.h>
#include <obs-module.h>
#include <obs-source.h>
#include <util/circlebuf.h>
#include <util/dstr.h>
}
//#include "simd_interface.h"
#include "mdct_motion_detect.hpp"

#define S_MZ_SLOT               "mz_slot"
#define S_MASK_PATH             "mask_path"
#define T_MASK_PATH             "Path"
#define S_DIFF_THRESHOLD        "diff_thr"
#define T_DIFF_THRESHOLD        "Capture background threshold"
#define TEXT_PATH_IMAGES               "BrowsePath.Images"
#define TEXT_PATH_ALL_FILES            "BrowsePath.AllFiles"
//#define IMAGE_FILTER_EXTENSIONS " (*.bmp *.jpg *.jpeg *.tga *.gif *.png)"
#define IMAGE_FILTER_EXTENSIONS " (*.bmp *.png)"
static char *NAME_MD = "MotionDetect";
static char *NAME_CT = "CropTrack";

// Shared TexturePosition slots

static texture_position_t tp_slots[10];

struct motion_zone_filter_data {
    uint32_t mz_slot;
    obs_source_t *context;
    char* mask_path;
    gs_image_file_t mask;
    CMotionZone* mz;
};


extern "C" {

    const char* mdf_get_name(void* unused)
    {
        UNUSED_PARAMETER(unused);
        return NAME_MD;
    }

    obs_properties_t* mdf_properties(void* data)
    {
        struct dstr filter_str = {0};
        dstr_copy(&filter_str, TEXT_PATH_IMAGES);
        dstr_cat(&filter_str, IMAGE_FILTER_EXTENSIONS ";;");
        dstr_cat(&filter_str, TEXT_PATH_ALL_FILES);
        dstr_cat(&filter_str, " (*.*)");

        obs_properties_t* props = obs_properties_create();
        obs_properties_add_int(props, S_MZ_SLOT, S_MZ_SLOT, 0, 9, 1);
        obs_properties_add_path(props, S_MASK_PATH, T_MASK_PATH, OBS_PATH_FILE, filter_str.array, NULL);
        obs_properties_add_float(props, S_DIFF_THRESHOLD, T_DIFF_THRESHOLD, 0.1, 127.9, 0.1);
        UNUSED_PARAMETER(data);
        return props;
    }

    void mdf_defaults(obs_data_t* settings)
    {
        obs_data_set_default_int(settings, S_MZ_SLOT, 0);
        obs_data_set_default_double(settings, S_DIFF_THRESHOLD,     5.0);
    }

    void mdf_update(void* data, obs_data_t* s)
    {
        struct motion_zone_filter_data* f = (motion_zone_filter_data*)data;
        f->mz_slot = (uint32_t)obs_data_get_int(s, S_MZ_SLOT);
        f->mz->m_ssd_threshold = obs_data_get_double(s, S_DIFF_THRESHOLD);
        const char *path = obs_data_get_string(s, S_MASK_PATH);
        if (f->mask_path)
            bfree(f->mask_path);
        f->mask_path = bstrdup(path);
        gs_image_file_free(&f->mask);
        gs_image_file_init(&f->mask, path);
        f->mz->mask_update(&f->mask);
    }

    void* mdf_create(obs_data_t* settings, obs_source_t* context)
    {
        struct motion_zone_filter_data* f = (motion_zone_filter_data*)bzalloc(sizeof(motion_zone_filter_data));
        f->context = context;
        f->mz = new CMotionZone();
        obs_source_update(context, settings);
        return f;
    }

    void mdf_destroy(void* data)
    {
        struct motion_zone_filter_data* f = (motion_zone_filter_data*)data;
        if (f->mz) {
            delete f->mz;
            f->mz = 0;
        }
        if (f->mask_path) {
            bfree(f->mask_path);
            f->mask_path = 0;
        }
        gs_image_file_free(&f->mask);
        bfree(f);
    }

    static void free_video_data(struct motion_zone_filter_data* f, obs_source_t* parent)
    {
        if (f->mz) {
            delete f->mz;
            f->mz = 0;
        }
        if (f->mask_path) {
            bfree(f->mask_path);
            f->mask_path = 0;
        }
        gs_image_file_free(&f->mask);
    }

    struct obs_source_frame* mdf_video(void* data, struct obs_source_frame* frame)
    {
        struct motion_zone_filter_data* filter = (motion_zone_filter_data*)data;
        obs_source_t* parent = obs_filter_get_parent(filter->context);
        filter->mz->feed_y(frame, &tp_slots[filter->mz_slot]);
        return frame;
    }

    void mdf_remove(void* data, obs_source_t* parent)
    {
        struct motion_zone_filter_data* filter = (motion_zone_filter_data*)data;
        free_video_data(filter, parent);
    }

}

#if 1

#define S_CT_SLOT            "ct_slot"
#define S_HORIZ_ACCEL        "horizontal_acceleration"
#define S_HORIZ_SPEED_MAX    "horizontal_max_speed"
#define S_DOWN_ACCEL         "down_acceleration"
#define S_DOWN_SPEED_MAX     "down_max_speed"
#define S_UP_ACCEL           "up_acceleration"
#define S_UP_SPEED_MAX       "up_max_speed"
#define S_ZOOM_IN_ACCEL      "zoom_in_acceleration"
#define S_ZOOM_IN_SPEED_MAX  "zoom_in_speed_max"
#define S_ZOOM_OUT_ACCEL     "zoom_out_acceleration"
#define S_ZOOM_OUT_SPEED_MAX "zoom_out_speed_max"
#define S_CENTER_DEV         "max_center_deviation"

#define S_MAX_SCALE     "max_scale"
//#define S_TAILING_COUNT "block_tail"

static const char* ct_effect_text = "uniform float4x4 ViewProj;\
uniform texture2d image;\
\
uniform float2 mul_val;\
uniform float2 add_val;\
\
sampler_state textureSampler {\
    Filter    = Linear;\
AddressU  = Border;\
AddressV  = Border;\
BorderColor = 00000000;\
};\
\
struct VertData {\
    float4 pos : POSITION;\
    float2 uv  : TEXCOORD0;\
};\
\
VertData VSCrop(VertData v_in)\
{\
    VertData vert_out;\
    vert_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);\
    vert_out.uv  = v_in.uv * mul_val + add_val;\
    return vert_out;\
}\
\
float4 PSCrop(VertData v_in) : TARGET\
{\
    return image.Sample(textureSampler, v_in.uv);\
}\
\
technique Draw\
{\
    pass\
    {\
        vertex_shader = VSCrop(v_in);\
    pixel_shader  = PSCrop(v_in);\
    }\
}\
";

#define SCALE_OPTIONS 0.05

struct crop_track_data {
    obs_source_t *context;

    gs_effect_t *effect;
    // User defined
    uint32_t ct_slot;
    CropTrack* ct;
    /*float accel_horiz, accel_down, accel_up, accel_zoom_in, accel_zoom_out;
    float speed_horiz, speed_down, speed_up, speed_zoom_in, speed_zoom_out;
    float center_dev;*/
    // Maximum scale
    //float max_scale;

    // Dynamic pass to shader
    gs_eparam_t *param_mul;
    gs_eparam_t *param_add;
    // Output dimensions
    uint32_t width, height;

    //struct vec2 mul_val;
    //struct vec2 add_val;
};

extern "C" {

    const char* cte_get_name(void* unused)
    {
        UNUSED_PARAMETER(unused);
        return NAME_CT;
    }

    void* cte_create(obs_data_t* settings, obs_source_t* context)
    {
        struct crop_track_data* filter = (crop_track_data*)bzalloc(sizeof(*filter));
        char* errors = NULL;
        filter->ct = NULL;
        filter->context = context;
        obs_enter_graphics();
        filter->effect = gs_effect_create(ct_effect_text, NULL, &errors);
        obs_leave_graphics();

        if (!filter->effect) {
            bfree(filter);
            return NULL;
        }

        filter->ct = new CropTrack();

        filter->param_mul = gs_effect_get_param_by_name(filter->effect, "mul_val");
        filter->param_add = gs_effect_get_param_by_name(filter->effect, "add_val");

        obs_source_update(context, settings);
        return filter;
    }

    void cte_destroy(void* data)
    {
        struct crop_track_data* filter = (crop_track_data*)data;

        obs_enter_graphics();
        gs_effect_destroy(filter->effect);
        obs_leave_graphics();
        if(filter->ct)
            delete filter->ct;
        bfree(filter);
    }

    void cte_update(void* data, obs_data_t* s)
    {
        struct crop_track_data* f = (crop_track_data*)data;
        f->ct_slot = (uint32_t)obs_data_get_int(s, S_CT_SLOT);
        f->ct->accel_horiz =     (float)obs_data_get_double(s, S_HORIZ_ACCEL);
        f->ct->accel_down =      (float)obs_data_get_double(s, S_DOWN_ACCEL);
        f->ct->accel_up =        (float)obs_data_get_double(s, S_UP_ACCEL);
        f->ct->accel_zoom_in =   (float)obs_data_get_double(s, S_ZOOM_IN_ACCEL);
        f->ct->accel_zoom_out =  (float)obs_data_get_double(s, S_ZOOM_OUT_ACCEL);
        f->ct->speed_horiz =     (float)obs_data_get_double(s, S_HORIZ_SPEED_MAX);
        f->ct->speed_down =      (float)obs_data_get_double(s, S_DOWN_SPEED_MAX);
        f->ct->speed_up =        (float)obs_data_get_double(s, S_UP_SPEED_MAX);
        f->ct->speed_zoom_in =   (float)obs_data_get_double(s, S_ZOOM_IN_SPEED_MAX);
        f->ct->speed_zoom_out =  (float)obs_data_get_double(s, S_ZOOM_OUT_SPEED_MAX);
        f->ct->center_dev =      (float)obs_data_get_double(s, S_CENTER_DEV);
        f->ct->max_scale =       (float)obs_data_get_double(s, S_MAX_SCALE);
        tp_reset(&tp_slots[f->ct_slot]);
        // ???
        f->width = 0;
        f->height = 0;
    }

    obs_properties_t* cte_properties(void* data)
    {
        obs_properties_t* props = obs_properties_create();
        obs_properties_add_int(props, S_CT_SLOT, S_CT_SLOT, 0, 9, 1);
        obs_properties_add_float(props, S_HORIZ_ACCEL, S_HORIZ_ACCEL,               0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_HORIZ_SPEED_MAX, S_HORIZ_SPEED_MAX,       0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_DOWN_ACCEL, S_DOWN_ACCEL,                 0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_DOWN_SPEED_MAX, S_DOWN_SPEED_MAX,         0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_UP_ACCEL, S_UP_ACCEL,                     0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_UP_SPEED_MAX, S_UP_SPEED_MAX,             0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_ZOOM_IN_ACCEL, S_ZOOM_IN_ACCEL,           0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_ZOOM_IN_SPEED_MAX, S_ZOOM_IN_SPEED_MAX,   0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_ZOOM_OUT_ACCEL, S_ZOOM_OUT_ACCEL,         0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_ZOOM_OUT_SPEED_MAX, S_ZOOM_OUT_SPEED_MAX, 0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_CENTER_DEV, S_CENTER_DEV,                 0.01, 9.9, 0.01);
        obs_properties_add_float(props, S_MAX_SCALE, S_MAX_SCALE,                   2.00, 4.0, 0.1);
        UNUSED_PARAMETER(data);
        return props;
    }

    void cte_defaults(obs_data_t* settings)
    {
        obs_data_set_default_int(settings, S_CT_SLOT, 0);
        obs_data_set_default_double(settings, S_HORIZ_ACCEL,        0.1);
        obs_data_set_default_double(settings, S_HORIZ_SPEED_MAX,    0.5);
        obs_data_set_default_double(settings, S_DOWN_ACCEL,         0.3);
        obs_data_set_default_double(settings, S_DOWN_SPEED_MAX,     1.0);
        obs_data_set_default_double(settings, S_UP_ACCEL,           0.1);
        obs_data_set_default_double(settings, S_UP_SPEED_MAX,       0.4);
        obs_data_set_default_double(settings, S_ZOOM_IN_ACCEL,      0.1);
        obs_data_set_default_double(settings, S_ZOOM_IN_SPEED_MAX,  0.2);
        obs_data_set_default_double(settings, S_ZOOM_OUT_ACCEL,     0.2);
        obs_data_set_default_double(settings, S_ZOOM_OUT_SPEED_MAX, 0.5);
        obs_data_set_default_double(settings, S_CENTER_DEV,         0.1);
        obs_data_set_default_double(settings, S_MAX_SCALE, 2.2);
    }

    void cte_tick(void* data, float seconds)
    {
        struct crop_track_data* filter = (crop_track_data*)data;

        //vec2_zero(&filter->mul_val);
        //vec2_zero(&filter->add_val);
        //calc_crop_dimensions(filter);

        obs_source_t *target = obs_filter_get_target(filter->context);
        if (target) {
            filter->width = obs_source_get_base_width(target);
            filter->height = obs_source_get_base_height(target);
        }
        filter->ct->tick(&tp_slots[filter->ct_slot], seconds);

        //UNUSED_PARAMETER(seconds);
    }

    void cte_render(void* data, gs_effect_t* effect)
    {
        struct crop_track_data* filter = (crop_track_data*)data;

        if (!obs_source_process_filter_begin(filter->context, GS_RGBA,
            OBS_NO_DIRECT_RENDERING))
            return;

        gs_effect_set_vec2(filter->param_mul, &filter->ct->texp.multiply);
        gs_effect_set_vec2(filter->param_add, &filter->ct->texp.offset);

        obs_source_process_filter_end(filter->context, filter->effect, filter->width, filter->height);

        UNUSED_PARAMETER(effect);
    }
}

#endif


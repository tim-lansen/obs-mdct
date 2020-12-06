#include <obs-module.h>


const char *mdf_get_name(void *unused);
obs_properties_t *mdf_properties(void *data);
void mdf_defaults(obs_data_t* settings);
void mdf_update(void *data, obs_data_t *s);
void *mdf_create(obs_data_t *settings, obs_source_t *context);
void mdf_destroy(void *data);
struct obs_source_frame* mdf_video(void *data, struct obs_source_frame *frame);
void mdf_remove(void *data, obs_source_t *parent);


struct obs_source_info motion_detect_filter = {
    .id = "motion_zone",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
    .get_name = mdf_get_name,
    .get_properties = mdf_properties,
    .get_defaults = mdf_defaults,
    .update = mdf_update,
    .create = mdf_create,
    .destroy = mdf_destroy,
    
    .filter_video = mdf_video,
    .filter_remove = mdf_remove,
};


const char *cte_get_name(void *unused);
void *cte_create(obs_data_t *settings, obs_source_t *context);
void cte_destroy(void *data);
void cte_update(void *data, obs_data_t *s);
obs_properties_t* cte_properties(void *data);
void cte_defaults(obs_data_t *settings);
void cte_tick(void *data, float seconds);
void cte_render(void *data, gs_effect_t *effect);

struct obs_source_info crop_track_effect = {
    .id = "crop_track",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = cte_get_name,
    .create = cte_create,
    .destroy = cte_destroy,
    .update = cte_update,
    .get_properties = cte_properties,
    .get_defaults = cte_defaults,
    .video_tick = cte_tick,
    .video_render = cte_render,
    //.get_width = crop_auto_width,
    //.get_height = crop_auto_height,
};


OBS_DECLARE_MODULE()

/*static obs_module_t *obs_module_pointer;
MODULE_EXPORT void obs_module_set_pointer(obs_module_t *module);
void obs_module_set_pointer(obs_module_t *module)
{
obs_module_pointer = module;
}

obs_module_t *obs_current_module(void)
{
return obs_module_pointer;
}

MODULE_EXPORT uint32_t obs_module_ver(void);
uint32_t obs_module_ver(void)
{
return LIBOBS_API_VER;
}*/

OBS_MODULE_USE_DEFAULT_LOCALE("obs-motion-zone-crop-track", "en-US")

/*lookup_t *obs_module_lookup = NULL;
const char *obs_module_text(const char *val)
{
const char *out = val;
text_lookup_getstr(obs_module_lookup, val, &out);
return out;
}
bool obs_module_get_string(const char *val, const char **out)
{
return text_lookup_getstr(obs_module_lookup, val, out);
}
void obs_module_set_locale(const char *locale)
{
if (obs_module_lookup)
text_lookup_destroy(obs_module_lookup);
obs_module_lookup = obs_module_load_locale(obs_current_module(), "en-US", locale);
}
void obs_module_free_locale(void)
{
text_lookup_destroy(obs_module_lookup);
obs_module_lookup = NULL;
}*/


bool obs_module_load(void)
{
    obs_register_source(&crop_track_effect);
    obs_register_source(&motion_detect_filter);
    return true;
}

void obs_module_unload(void)
{
}

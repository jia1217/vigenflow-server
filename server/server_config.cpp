#include "server_config.hpp"

namespace server::config {

// const std::string kInstallDir = "/opt/vigenflow";
const std::string kInstallDir = ".";

std::string model_id = "flux.2-klein-4B";
std::string served_model_id;
std::string weights_path = kInstallDir + "/model_weights";
std::string npu_base_dir = kInstallDir + "/npu_files";
std::string exe_base_dir = kInstallDir + "/exe_models";
std::string custom_exe;
std::string workdir = kInstallDir;

std::string lora_dir;
std::string lora_catalog_dir;
std::string lora_model_list;
std::string lora_model_list_q41;
std::string lora_model_list_bf16;
std::string lora_model_list_flux;
std::string lora_model_list_flux_edit;
std::string lora_source;
std::string lora_file;
std::string lora_revision = "main";
std::string lora_cache_dir;
std::string lora_export_dir;
bool lora_force = false;
bool lora_keep_cache = false;
int lora_rank = 0;

std::string gguf_path;
std::string gguf_repo = "unsloth/Z-Image-Turbo-GGUF";
std::string gguf_file = "z-image-turbo-Q4_1.gguf";
std::string gguf_revision = "main";
std::string gguf_hf_token;
bool force_gguf_download = false;

bool auto_download_z_image_bf16 = true;
bool force_z_image_bf16_download = false;
std::string z_image_bf16_repo = "Tongyi-MAI/Z-Image-Turbo";
std::string z_image_bf16_revision = "main";
std::string z_image_bf16_cache_dir;
std::string z_image_bf16_hf_token;

bool auto_download_flux_klein_bf16 = true;
bool force_flux_klein_bf16_download = false;
std::string flux_klein_bf16_repo = "Kelsey1217/FLUX.2-klein-4B-npu";
std::string flux_klein_bf16_revision = "main";
std::string flux_klein_bf16_cache_dir;
std::string flux_klein_bf16_hf_token;

unsigned short port = 11283;
std::string public_base_url = "http://127.0.0.1:" + std::to_string(port);
std::string output_dir = "../images";

bool keep_images = false;

int default_seed = 42;
int default_steps = 4;
int default_height = 1024;
int default_width = 1024;
std::string default_prompt =
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup...";

}  // namespace server::config

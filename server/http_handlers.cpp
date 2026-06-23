#include "http_handlers.hpp"

#include "file_utils.hpp"
#include "request_parser.hpp"
#include "server_config.hpp"
#include "worker.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace server {

namespace {

std::mutex g_infer_mutex;
std::mutex g_openwebui_model_mutex;
std::mutex g_last_image_usage_mutex;
std::string g_last_openwebui_model;
std::uint64_t g_openwebui_model_generation = 0;
json g_last_image_usage = { 
    {"1_Image_H", 0},
    {"2_Image_W", 0},
    {"3_Denoising_steps", 0},
    {"4_Text_encoder", "0.00s"},
    {"5_Denoising", "0.00s/step, 0s"},
    {"6_Vae_decoder", "0.00s"},
    {"7_E2E_runtime", "0.00s"}};
   

struct RememberedOpenWebUIModel {
    std::optional<std::string> model;
    std::uint64_t generation = 0;
};

http::response<http::string_body> make_json_response(
    http::status status, unsigned version, bool keep_alive, const json& body) {
    http::response<http::string_body> res{status, version};
    res.set(http::field::content_type, "application/json");
    res.keep_alive(keep_alive);
    res.body() = body.dump();
    res.prepare_payload();
    return res;
}

http::response<http::vector_body<char>> make_image_response(
    unsigned version, bool keep_alive, std::vector<char>&& data, const std::string& mime) {
    http::response<http::vector_body<char>> res{http::status::ok, version};
    res.set(http::field::content_type, mime);
    res.keep_alive(keep_alive);
    res.body() = std::move(data);
    res.prepare_payload();
    return res;
}

double timing_value_or_zero(const std::optional<double>& value) {
    return value.value_or(0.0);
}

long long rounded_ms(double value) {
    return static_cast<long long>(value + 0.5);
}

std::string format_seconds(double ms) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << (ms / 1000.0) << "s";
    return out.str();
}

std::string format_whole_seconds(double ms) {
    return std::to_string(static_cast<long long>((ms / 1000.0) + 0.5)) + "s";
}

std::string format_denoising_time(double step_ms, double total_ms) {
    return format_seconds(step_ms) + "/step, " + format_whole_seconds(total_ms);
}

void add_optional_timing_ms(json& timing_ms, const char* key, const std::optional<double>& value) {
    if (value) {
        timing_ms[key] = *value;
    }
}

json build_image_usage(const WorkerTiming& timing, const GenParams& params) {
    const double text_encoder_ms = timing_value_or_zero(timing.text_encoder_ms);
    const double denoising_ms = timing_value_or_zero(timing.denoising_ms);
    const double vae_decoder_ms = timing_value_or_zero(timing.vae_decoder_ms);
    const double input_ms = text_encoder_ms + denoising_ms;
    const double output_ms = vae_decoder_ms;
    const double total_ms = timing.e2e_runtime_ms.value_or(
        timing.server_elapsed_ms.value_or(input_ms + output_ms));

    json timing_ms = json::object();
    add_optional_timing_ms(timing_ms, "text_encoder", timing.text_encoder_ms);
    add_optional_timing_ms(timing_ms, "denoising", timing.denoising_ms);
    add_optional_timing_ms(timing_ms, "denoising_step", timing.denoising_step_ms);
    add_optional_timing_ms(timing_ms, "vae_decoder", timing.vae_decoder_ms);
    add_optional_timing_ms(timing_ms, "e2e_runtime", timing.e2e_runtime_ms);
    add_optional_timing_ms(timing_ms, "server_elapsed", timing.server_elapsed_ms);

    json usage = {
        {"prompt_tokens", rounded_ms(input_ms)},
        {"completion_tokens", rounded_ms(output_ms)},
        {"total_tokens", rounded_ms(total_ms)},
        {"input_tokens", rounded_ms(input_ms)},
        {"output_tokens", rounded_ms(output_ms)},
        {"image_H", params.H},
        {"image_W", params.W},
        {"steps", params.steps},
        {"input_tokens_details",
         {{"text_tokens", rounded_ms(text_encoder_ms)},
          {"image_tokens", rounded_ms(denoising_ms)}}}};

    if (!timing_ms.empty()) {
        usage["timing_ms"] = timing_ms;
    }

    return usage;
}

json openwebui_usage_from_image_usage(const json& usage) {
    if (usage.contains("timing_ms") && usage["timing_ms"].is_object()) {
        const json& timing_ms = usage["timing_ms"];
        return {
            {"1_Image_H", usage.value("image_H", 0)},
            {"2_Image_W", usage.value("image_W", 0)},
            {"3_Denoising_steps", usage.value("steps", 0)},
            {"4_Text_encoder", format_seconds(timing_ms.value("text_encoder", 0.0))},
            {"5_Denoising",
             format_denoising_time(
                 timing_ms.value("denoising_step", timing_ms.value("denoising", 0.0)),
                 timing_ms.value("denoising", 0.0))},
            {"6_Vae_decoder", format_seconds(timing_ms.value("vae_decoder", 0.0))},
            {"7_E2E_runtime", format_seconds(timing_ms.value("e2e_runtime", usage.value("total_tokens", 0)))}};
    }

    return {
        {"prompt_tokens", usage.value("prompt_tokens", usage.value("input_tokens", 0))},
        {"completion_tokens", usage.value("completion_tokens", usage.value("output_tokens", 0))},
        {"total_tokens", usage.value("total_tokens", 0)}};
}

void remember_last_image_usage(const json& usage) {
    std::lock_guard<std::mutex> lock(g_last_image_usage_mutex);
    g_last_image_usage = openwebui_usage_from_image_usage(usage);
}

json last_image_usage() {
    std::lock_guard<std::mutex> lock(g_last_image_usage_mutex);
    return g_last_image_usage;
}

json build_image_response_body(const GenParams& params, const WorkerResult& worker_result) {
    const auto data = read_binary_file(worker_result.image_path);
    const json item = {{"b64_json", base64_encode(data)}, {"revised_prompt", params.prompt}};
    return {
        {"created", static_cast<long long>(std::time(nullptr))},
        {"data", json::array({item})},
        {"usage", build_image_usage(worker_result.timing, params)}};
}

constexpr const char* kZImageTurboQ41ModelId = "z-image-turbo-Q4_1-GGUF";
constexpr const char* kZImageTurboQ41LoraModelId = "z-image-turbo-Q4_1-lora";
constexpr const char* kZImageTurboBf16ModelId = "z-image-turbo-BF16";
constexpr const char* kZImageTurboBf16LoraModelId = "z-image-turbo-BF16-lora";

std::string configured_served_model_id() {
    const std::string model_id = !config::served_model_id.empty()
        ? config::served_model_id
        : config::model_id;

    if (model_id == "flux.2-klein-4B (lora)" || model_id == "flux.2-klein-4B(lora)" ||
        model_id == "flux.2-klein-lora" || model_id == "flux.2-klein (lora)" ||
        model_id == "flux.2-klein(lora)") {
        return "flux.2-klein-4B-lora";
    }
    if (model_id == "flux.2-klein-4B-edit (lora)" ||
        model_id == "flux.2-klein-4B-edit(lora)" ||
        model_id == "flux.2-klein-edit-lora" ||
        model_id == "flux.2-klein-edit (lora)" ||
        model_id == "flux.2-klein-edit(lora)") {
        return "flux.2-klein-4B-edit-lora";
    }
    if (model_id == "z-image-turbo" || model_id == "z-image-turbo-q41") {
        return kZImageTurboQ41ModelId;
    }
    if (model_id == "z-image-turbo-q41-lora" || model_id == "z-image-turbo-q41_lora" ||
        model_id == "z-image-turbo-lora" || model_id == "z-image-turbo-lora-q41") {
        return kZImageTurboQ41LoraModelId;
    }
    if (model_id == "z-image-turbo-bf16") {
        return kZImageTurboBf16ModelId;
    }
    if (model_id == "z-image-turbo-bf16 (lora)" || model_id == "z-image-turbo-bf16(lora)" ||
        model_id == "z-image-turbo-bf16-lora") {
        return kZImageTurboBf16LoraModelId;
    }

    return model_id;
}

bool is_flux_klein_model(const std::string& model_id) {
    return model_id == "flux.2-klein-4B" || model_id == "flux.2-klein" ||
           model_id == "flux.2-klein-4B (lora)" || model_id == "flux.2-klein-4B(lora)" ||
           model_id == "flux.2-klein-4B-lora" || model_id == "flux.2-klein (lora)" ||
           model_id == "flux.2-klein(lora)" || model_id == "flux.2-klein-lora" ||
           model_id == "flux.2-klein-4B-edit" || model_id == "flux.2-klein-edit" ||
           model_id == "flux.2-klein-4B-edit (lora)" ||
           model_id == "flux.2-klein-4B-edit(lora)" ||
           model_id == "flux.2-klein-4B-edit-lora" ||
           model_id == "flux.2-klein-edit (lora)" ||
           model_id == "flux.2-klein-edit(lora)" ||
           model_id == "flux.2-klein-edit-lora";
}

bool is_flux_klein_lora_model(const std::string& model_id) {
    return model_id == "flux.2-klein-4B (lora)" || model_id == "flux.2-klein-4B(lora)" ||
           model_id == "flux.2-klein-4B-lora" || model_id == "flux.2-klein (lora)" ||
           model_id == "flux.2-klein(lora)" || model_id == "flux.2-klein-lora" ||
           model_id == "flux.2-klein-4B-edit (lora)" ||
           model_id == "flux.2-klein-4B-edit(lora)" ||
           model_id == "flux.2-klein-4B-edit-lora" ||
           model_id == "flux.2-klein-edit (lora)" ||
           model_id == "flux.2-klein-edit(lora)" ||
           model_id == "flux.2-klein-edit-lora";
}

bool is_z_image_bf16_model(const std::string& model_id) {
    return model_id == "z-image-turbo-bf16" || model_id == kZImageTurboBf16ModelId ||
           model_id == "z-image-turbo-bf16 (lora)" || model_id == "z-image-turbo-bf16(lora)" ||
           model_id == "z-image-turbo-bf16-lora" || model_id == kZImageTurboBf16LoraModelId;
}

bool is_z_image_bf16_lora_model(const std::string& model_id) {
    return model_id == "z-image-turbo-bf16 (lora)" || model_id == "z-image-turbo-bf16(lora)" ||
           model_id == "z-image-turbo-bf16-lora" || model_id == kZImageTurboBf16LoraModelId;
}

bool is_z_image_q41_model(const std::string& model_id) {
    return model_id == "z-image-turbo" || model_id == "z-image-turbo-q41" ||
           model_id == kZImageTurboQ41ModelId ||
           model_id == "z-image-turbo-q41-lora" || model_id == "z-image-turbo-q41_lora" ||
           model_id == "z-image-turbo-lora" || model_id == "z-image-turbo-lora-q41" ||
           model_id == kZImageTurboQ41LoraModelId;
}

bool serving_z_image_bf16(const std::string& served_model_id) {
    return is_z_image_bf16_model(served_model_id) || is_z_image_bf16_model(config::model_id);
}

bool serving_z_image_q41(const std::string& served_model_id) {
    return is_z_image_q41_model(served_model_id) || is_z_image_q41_model(config::model_id);
}

void add_model_entry(json& data, std::set<std::string>& seen, const std::string& model_id) {
    if (model_id.empty() || !seen.insert(model_id).second) {
        return;
    }

    data.push_back(
        {{"id", model_id},
         {"object", "model"},
         {"created", 0},
         {"owned_by", "local"}});
}


bool is_local_model_alias(const std::string& model_id) {
    return model_id.empty() || model_id == "local-model-1" || model_id == "edit_model";
}

std::optional<std::string> canonical_lora_model_id(const std::string& model_id) {
    const auto entry = resolve_lora_model_entry(model_id);
    if (!entry) {
        return std::nullopt;
    }
    return entry->model_id;
}

std::optional<std::string> canonical_openwebui_model_id(const std::string& model_id) {
    if (const auto lora_model = canonical_lora_model_id(model_id)) {
        return lora_model;
    }
    if (is_flux_klein_model(model_id) ||
        is_z_image_q41_model(model_id) ||
        is_z_image_bf16_model(model_id)) {
        return model_id;
    }
    return std::nullopt;
}

bool is_generic_flux_klein_lora_model(const std::string& model_id) {
    if (canonical_lora_model_id(model_id)) {
        return false;
    }
    if (is_flux_klein_lora_model(model_id)) {
        return true;
    }
    if (is_local_model_alias(model_id)) {
        if (is_flux_klein_lora_model(config::model_id) &&
            !canonical_lora_model_id(config::model_id)) {
            return true;
        }
        return !config::served_model_id.empty() &&
               is_flux_klein_lora_model(config::served_model_id) &&
               !canonical_lora_model_id(config::served_model_id);
    }
    return false;
}

bool should_apply_remembered_openwebui_model(
    const std::string& requested_model,
    const std::string& remembered_model) {
    if (is_local_model_alias(requested_model)) {
        return true;
    }
    if (canonical_lora_model_id(requested_model)) {
        return false;
    }
    return normalize_model_target(requested_model) == normalize_model_target(remembered_model);
}


bool contains_string(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

std::vector<std::string> lora_prompt_match_tokens(std::string value) {
    std::vector<std::string> tokens;
    std::string token;

    auto flush_token = [&]() {
        if (token.size() < 3) {
            token.clear();
            return;
        }
        static const std::vector<std::string> ignored = {
            "flux", "klein", "lora", "edit", "model", "safetensors", "fal", "the", "and", "with"
        };
        if (!contains_string(ignored, token) && !contains_string(tokens, token)) {
            tokens.push_back(token);
        }
        token.clear();
    };

    for (unsigned char ch : value) {
        if (std::isalnum(ch)) {
            token.push_back(static_cast<char>(std::tolower(ch)));
        } else {
            flush_token();
        }
    }
    flush_token();
    return tokens;
}

int lora_prompt_match_score(const LoraModelEntry& entry, const std::string& prompt) {
    const auto prompt_tokens = lora_prompt_match_tokens(prompt);
    if (prompt_tokens.empty()) {
        return 0;
    }

    const auto entry_tokens = lora_prompt_match_tokens(
        entry.model_id + " " + entry.lora_source + " " + entry.lora_file);
    int score = 0;
    for (const auto& entry_token : entry_tokens) {
        for (const auto& prompt_token : prompt_tokens) {
            if (entry_token == prompt_token ||
                (entry_token.size() >= 4 && prompt_token.find(entry_token) != std::string::npos) ||
                (prompt_token.size() >= 4 && entry_token.find(prompt_token) != std::string::npos)) {
                ++score;
                break;
            }
        }
    }
    return score;
}

std::optional<std::string> prompt_matched_lora_model_id(
    const std::string& requested_model,
    const std::string& prompt) {
    const std::string requested_target = normalize_model_target(requested_model);
    std::optional<LoraModelEntry> best_entry;
    int best_score = 0;
    bool tied = false;

    for (const auto& entry : list_lora_model_entries()) {
        if (normalize_model_target(entry.model_id) != requested_target) {
            continue;
        }
        const int score = lora_prompt_match_score(entry, prompt);
        if (score > best_score) {
            best_score = score;
            best_entry = entry;
            tied = false;
        } else if (score > 0 && score == best_score) {
            tied = true;
        }
    }

    if (best_score > 0 && best_entry && !tied) {
        return best_entry->model_id;
    }
    return std::nullopt;
}

std::optional<std::string> request_model_from_json_body(const std::string& body) {
    try {
        const json parsed = json::parse(body);
        if (parsed.contains("model") && parsed["model"].is_string()) {
            return parsed["model"].get<std::string>();
        }
    } catch (...) {
    }
    return std::nullopt;
}

void remember_openwebui_model(const std::string& body) {
    const auto model = request_model_from_json_body(body);
    if (!model || is_local_model_alias(*model)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_openwebui_model_mutex);
    const auto canonical_model = canonical_openwebui_model_id(*model);
    if (canonical_model) {
        g_last_openwebui_model = *canonical_model;
        ++g_openwebui_model_generation;
        std::cout << "[INFO] Remembered OpenWebUI model: "
                  << g_last_openwebui_model << "\n";
        return;
    }

    if (!g_last_openwebui_model.empty()) {
        std::cout << "[INFO] Cleared remembered OpenWebUI model after selecting "
                  << *model << "\n";
        g_last_openwebui_model.clear();
        ++g_openwebui_model_generation;
    }
}

RememberedOpenWebUIModel remembered_openwebui_model() {
    std::lock_guard<std::mutex> lock(g_openwebui_model_mutex);
    if (g_last_openwebui_model.empty()) {
        return {std::nullopt, g_openwebui_model_generation};
    }
    return {g_last_openwebui_model, g_openwebui_model_generation};
}

std::optional<RememberedOpenWebUIModel> wait_for_openwebui_model_update(
    const std::string& requested_model,
    std::uint64_t after_generation) {
    for (int attempt = 0; attempt < 60; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto remembered_model = remembered_openwebui_model();
        if (remembered_model.generation == after_generation) {
            continue;
        }
        if (remembered_model.model &&
            should_apply_remembered_openwebui_model(requested_model, *remembered_model.model)) {
            return remembered_model;
        }
        return remembered_model;
    }
    return std::nullopt;
}

void apply_openwebui_image_model_alias(GenParams& params) {
    if (const auto canonical_model = canonical_lora_model_id(params.model)) {
        if (*canonical_model != params.model) {
            std::cout << "[INFO] OpenWebUI sent image model " << params.model
                      << "; using canonical LoRA model " << *canonical_model << "\n";
            params.model = *canonical_model;
        }
        {
            std::lock_guard<std::mutex> lock(g_openwebui_model_mutex);
            if (g_last_openwebui_model != *canonical_model) {
                std::cout << "[INFO] Remembered OpenWebUI image LoRA model: "
                          << *canonical_model << "\n";
            }
            g_last_openwebui_model = *canonical_model;
            ++g_openwebui_model_generation;
        }
        return;
    }

    RememberedOpenWebUIModel remembered_model = remembered_openwebui_model();
    if (remembered_model.model && is_local_model_alias(params.model)) {
        std::cout << "[INFO] OpenWebUI sent image model " << params.model
                  << "; using remembered model " << *remembered_model.model << "\n";
        params.model = *remembered_model.model;
        return;
    }

    if (is_generic_flux_klein_lora_model(params.model)) {
        if (const auto fresh_model =
                wait_for_openwebui_model_update(params.model, remembered_model.generation)) {
            remembered_model = *fresh_model;
            if (remembered_model.model &&
                should_apply_remembered_openwebui_model(params.model, *remembered_model.model)) {
                std::cout << "[INFO] OpenWebUI sent image model " << params.model
                          << "; using fresh model " << *remembered_model.model << "\n";
                params.model = *remembered_model.model;
            }
            return;
        }
        if (const auto prompt_model = prompt_matched_lora_model_id(params.model, params.prompt)) {
            std::cout << "[INFO] OpenWebUI sent image model " << params.model
                      << "; using prompt-matched LoRA model " << *prompt_model << "\n";
            params.model = *prompt_model;
            return;
        }
        if (remembered_model.model) {
            std::cout << "[INFO] OpenWebUI sent generic image model " << params.model
                      << "; ignoring stale remembered model "
                      << *remembered_model.model << "\n";
        }
        return;
    }

    if (!remembered_model.model ||
        !should_apply_remembered_openwebui_model(params.model, *remembered_model.model)) {
        return;
    }

    std::cout << "[INFO] OpenWebUI sent image model " << params.model
              << "; using remembered model " << *remembered_model.model << "\n";
    params.model = *remembered_model.model;
}

http::message_generator handle_get_models(unsigned version, bool keep_alive) {
    // Advertise base models. Catalog entries below add selectable per-LoRA model IDs.
    json data = json::array();
    std::set<std::string> seen;

    add_model_entry(data, seen, "flux.2-klein-4B");
    add_model_entry(data, seen, "flux.2-klein-4B-edit");
    add_model_entry(data, seen, kZImageTurboQ41ModelId);
    add_model_entry(data, seen, kZImageTurboBf16ModelId);

    for (const auto& entry : list_lora_model_entries()) {
        add_model_entry(data, seen, entry.model_id);
    }

    const json body = {{"object", "list"}, {"data", data}};
    return make_json_response(http::status::ok, version, keep_alive, body);
}

http::message_generator handle_get_image_file(
    const std::string& target, unsigned version, bool keep_alive) {
    const std::string filename = target.substr(8);
    if (!is_safe_filename(filename)) {
        return make_json_response(
            http::status::bad_request, version, keep_alive, json{{"error", "invalid filename"}});
    }

    const std::string fullpath = config::output_dir + "/" + filename;
    if (!fs::exists(fullpath)) {
        return make_json_response(
            http::status::not_found, version, keep_alive, json{{"error", "image not found"}});
    }

    auto data = read_binary_file(fullpath);
    const std::string ext = fs::path(fullpath).extension().string();
    std::string mime = "image/png";
    if (ext == ".jpg" || ext == ".jpeg") {
        mime = "image/jpeg";
    } else if (ext == ".webp") {
        mime = "image/webp";
    }

    return make_image_response(version, keep_alive, std::move(data), mime);
}

http::message_generator handle_image_generation(
    GenParams params,
    const std::string& request_body,
    unsigned version,
    bool keep_alive) {
    const std::string response_format = get_response_format(request_body);
    apply_openwebui_image_model_alias(params);
    WorkerResult worker_result;
    {
        std::lock_guard<std::mutex> lock(g_infer_mutex);
        worker_result = run_worker(params);
    }

    const json body = build_image_response_body(params, worker_result);
    remember_last_image_usage(body.at("usage"));
    const bool keep_output = config::keep_images || response_format == "url";
    cleanup_generated_image(worker_result.image_path, keep_output);

    return make_json_response(http::status::ok, version, keep_alive, body);
}

http::message_generator handle_image_edit(
    const http::request<http::string_body>& req, unsigned version, bool keep_alive) {
    if (req[http::field::content_type].find("multipart/form-data") == std::string::npos) {
        return make_json_response(
            http::status::bad_request,
            version,
            keep_alive,
            json{{"error", "Content-Type must be multipart/form-data"}});
    }

    GenParams params = parse_multipart_request(req);
    apply_openwebui_image_model_alias(params);
    WorkerResult worker_result;
    try {
        std::lock_guard<std::mutex> lock(g_infer_mutex);
        worker_result = run_worker(params);
    } catch (...) {
        cleanup_uploaded_images(params.input_images);
        throw;
    }

    const json body = build_image_response_body(params, worker_result);
    remember_last_image_usage(body.at("usage"));
    cleanup_generated_image(worker_result.image_path, config::keep_images);
    cleanup_uploaded_images(params.input_images);

    return make_json_response(http::status::ok, version, keep_alive, body);
}

}  // namespace

http::message_generator handle_request(http::request<http::string_body>&& req) {
    const bool keep_alive = req.keep_alive();
    const std::string target = std::string(req.target());
    std::cout << "[REQ] " << req.method_string() << " " << target << "\n";

    try {
        if (req.method() == http::verb::get && target == "/health") {
            return make_json_response(
                http::status::ok, req.version(), keep_alive, json{{"status", "ok"}});
        }

        if (req.method() == http::verb::get && target == "/v1/models") {
            return handle_get_models(req.version(), keep_alive);
        }

        if (req.method() == http::verb::get && target.rfind("/images/", 0) == 0) {
            return handle_get_image_file(target, req.version(), keep_alive);
        }

        if (req.method() == http::verb::post && target == "/v1/images/generations") {
            return handle_image_generation(
                parse_request(req.body()), req.body(), req.version(), keep_alive);
        }

        if (req.method() == http::verb::post && target == "/v1/images/edits") {
            return handle_image_edit(req, req.version(), keep_alive);
        }

        if (req.method() == http::verb::post && target == "/v1/chat/completions") {
            remember_openwebui_model(req.body());

            const json body = {
                {"id", "chatcmpl-local"},
                {"object", "chat.completion"},
                {"created", static_cast<long long>(std::time(nullptr))},
                {"model", configured_served_model_id()},
                {"choices",
                 json::array(
                     {{{"index", 0},
                       {"message", {{"role", "assistant"}, {"content", " "}}},
                       {"finish_reason", "stop"}}})},
                {"usage", last_image_usage()}};
            return make_json_response(http::status::ok, req.version(), keep_alive, body);
        }

        if (target == "/v1/models" || target == "/v1/images/generations" || target == "/health" ||
            target.rfind("/images/", 0) == 0) {
            return make_json_response(
                http::status::method_not_allowed,
                req.version(),
                keep_alive,
                json{{"error", "method not allowed"}});
        }

        return make_json_response(
            http::status::not_found, req.version(), keep_alive, json{{"error", "not found"}});
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << req.method_string() << " " << target
                  << ": " << e.what() << "\n";
        return make_json_response(
            http::status::internal_server_error,
            req.version(),
            keep_alive,
            json{{"error", e.what()}});
    }
}

}  // namespace server

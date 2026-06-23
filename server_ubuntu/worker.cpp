#include "worker.hpp"

#include "file_utils.hpp"
#include "flux_klein_bf16_export.hpp"
#include "server_config.hpp"
#include "z_image_bf16_export.hpp"

#include <atomic>
#include <algorithm>
#include <boost/process.hpp>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace bp = boost::process;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace server {

namespace {

constexpr std::string_view kZImageTurboLoraPrefix = "z-image-turbo-lora:";
constexpr std::string_view kZImageTurboQ41LoraPrefix = "z-image-turbo-q41-lora:";
constexpr std::string_view kZImageTurboBf16LoraPrefix = "z-image-turbo-bf16-lora:";
constexpr std::string_view kZImageTurboQ41DisplayLoraPrefix = "z-image-turbo-Q4_1-lora:";
constexpr std::string_view kZImageTurboBf16DisplayLoraPrefix = "z-image-turbo-BF16-lora:";
constexpr std::string_view kZImageTurboLoraDashPrefix = "z-image-turbo-lora-";
constexpr std::string_view kZImageTurboQ41LoraDashPrefix = "z-image-turbo-q41-lora-";
constexpr std::string_view kZImageTurboBf16LoraDashPrefix = "z-image-turbo-bf16-lora-";
constexpr std::string_view kZImageTurboQ41DisplayLoraDashPrefix = "z-image-turbo-Q4_1-lora-";
constexpr std::string_view kZImageTurboBf16DisplayLoraDashPrefix = "z-image-turbo-BF16-lora-";
constexpr std::string_view kFluxKleinLoraPrefix = "flux.2-klein-4B-lora:";
constexpr std::string_view kFluxKleinEditLoraPrefix = "flux.2-klein-4B-edit-lora:";
constexpr std::string_view kFluxKleinLoraDashPrefix = "flux.2-klein-4B-lora-";
constexpr std::string_view kFluxKleinEditLoraDashPrefix = "flux.2-klein-4B-edit-lora-";
constexpr std::string_view kFluxKleinShortLoraPrefix = "flux.2-klein-lora:";
constexpr std::string_view kFluxKleinShortEditLoraPrefix = "flux.2-klein-edit-lora:";
constexpr std::string_view kFluxKleinShortLoraDashPrefix = "flux.2-klein-lora-";
constexpr std::string_view kFluxKleinShortEditLoraDashPrefix = "flux.2-klein-edit-lora-";

enum class ZImageLoraVariant { Q41, Bf16 };
enum class FluxKleinLoraVariant { Base, Edit };

constexpr const char* kZImageTurboQ41ModelId = "z-image-turbo-Q4_1-GGUF";
constexpr const char* kZImageTurboQ41LoraModelId = "z-image-turbo-Q4_1-lora";
constexpr const char* kZImageTurboBf16ModelId = "z-image-turbo-BF16";
constexpr const char* kZImageTurboBf16LoraModelId = "z-image-turbo-BF16-lora";

bool starts_with(const std::string& value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

std::string trim_copy(std::string value);
std::optional<std::string> z_image_lora_name_from_model_id(const std::string& model_id);
std::optional<ZImageLoraVariant> z_image_lora_variant_from_model_id(const std::string& model_id);

bool is_z_image_q41_base_model_id(const std::string& model_id) {
    return model_id == "z-image-turbo" ||
           model_id == "z-image-turbo-q41" ||
           model_id == kZImageTurboQ41ModelId;
}

bool is_z_image_bf16_base_model_id(const std::string& model_id) {
    return model_id == "z-image-turbo-bf16" ||
           model_id == kZImageTurboBf16ModelId;
}

const char* z_image_lora_base_model_id(ZImageLoraVariant variant) {
    return variant == ZImageLoraVariant::Bf16
        ? kZImageTurboBf16LoraModelId
        : kZImageTurboQ41LoraModelId;
}

bool configured_prefers_bf16_lora_ids() {
    const std::string served_model_id = !config::served_model_id.empty()
        ? config::served_model_id
        : config::model_id;
    return config::model_id == "z-image-turbo-bf16" ||
           config::model_id == kZImageTurboBf16ModelId ||
           config::model_id == "z-image-turbo-bf16-lora" ||
           config::model_id == kZImageTurboBf16LoraModelId ||
           config::model_id == "z-image-turbo-bf16 (lora)" ||
           config::model_id == "z-image-turbo-bf16(lora)" ||
           served_model_id == "z-image-turbo-bf16" ||
           served_model_id == kZImageTurboBf16ModelId ||
           served_model_id == "z-image-turbo-bf16-lora" ||
           served_model_id == kZImageTurboBf16LoraModelId ||
           served_model_id == "z-image-turbo-bf16 (lora)" ||
           served_model_id == "z-image-turbo-bf16(lora)";
}

std::string z_image_lora_model_id(const std::string& lora_name, ZImageLoraVariant variant) {
    const char* prefix = variant == ZImageLoraVariant::Bf16
        ? "z-image-turbo-BF16-lora-"
        : "z-image-turbo-Q4_1-lora-";
    return std::string(prefix) + lora_name;
}

std::string z_image_lora_model_id(const std::string& lora_name) {
    return z_image_lora_model_id(
        lora_name,
        configured_prefers_bf16_lora_ids() ? ZImageLoraVariant::Bf16 : ZImageLoraVariant::Q41);
}

std::string normalize_z_image_lora_model_id(
    const std::string& value,
    std::optional<ZImageLoraVariant> default_variant = std::nullopt) {
    if (const auto variant = z_image_lora_variant_from_model_id(value)) {
        if (const auto lora_name = z_image_lora_name_from_model_id(value)) {
            return z_image_lora_model_id(*lora_name, *variant);
        }
        return z_image_lora_base_model_id(*variant);
    }
    if (default_variant) {
        return z_image_lora_model_id(value, *default_variant);
    }
    return z_image_lora_model_id(value);
}

std::string flux_klein_lora_model_id(
    const std::string& lora_name,
    FluxKleinLoraVariant variant) {
    const char* prefix = variant == FluxKleinLoraVariant::Edit
        ? "flux.2-klein-4B-edit-lora-"
        : "flux.2-klein-4B-lora-";
    return std::string(prefix) + lora_name;
}

bool has_flux_klein_lora_prefix(const std::string& value) {
    return starts_with(value, kFluxKleinLoraPrefix) ||
           starts_with(value, kFluxKleinEditLoraPrefix) ||
           starts_with(value, kFluxKleinLoraDashPrefix) ||
           starts_with(value, kFluxKleinEditLoraDashPrefix) ||
           starts_with(value, kFluxKleinShortLoraPrefix) ||
           starts_with(value, kFluxKleinShortEditLoraPrefix) ||
           starts_with(value, kFluxKleinShortLoraDashPrefix) ||
           starts_with(value, kFluxKleinShortEditLoraDashPrefix);
}

std::string normalize_flux_klein_lora_model_id(
    const std::string& value,
    FluxKleinLoraVariant default_variant) {
    if (value == "flux.2-klein-edit-lora" ||
        value == "flux.2-klein-edit (lora)" ||
        value == "flux.2-klein-edit(lora)") {
        return "flux.2-klein-4B-edit-lora";
    }
    if (value == "flux.2-klein-lora" ||
        value == "flux.2-klein (lora)" ||
        value == "flux.2-klein(lora)") {
        return "flux.2-klein-4B-lora";
    }
    if (starts_with(value, kFluxKleinEditLoraPrefix) ||
        starts_with(value, kFluxKleinEditLoraDashPrefix) ||
        starts_with(value, kFluxKleinLoraPrefix) ||
        starts_with(value, kFluxKleinLoraDashPrefix)) {
        return value;
    }
    if (starts_with(value, kFluxKleinShortEditLoraPrefix)) {
        return flux_klein_lora_model_id(
            trim_copy(value.substr(kFluxKleinShortEditLoraPrefix.size())),
            FluxKleinLoraVariant::Edit);
    }
    if (starts_with(value, kFluxKleinShortEditLoraDashPrefix)) {
        return flux_klein_lora_model_id(
            trim_copy(value.substr(kFluxKleinShortEditLoraDashPrefix.size())),
            FluxKleinLoraVariant::Edit);
    }
    if (starts_with(value, kFluxKleinShortLoraPrefix)) {
        return flux_klein_lora_model_id(
            trim_copy(value.substr(kFluxKleinShortLoraPrefix.size())),
            FluxKleinLoraVariant::Base);
    }
    if (starts_with(value, kFluxKleinShortLoraDashPrefix)) {
        return flux_klein_lora_model_id(
            trim_copy(value.substr(kFluxKleinShortLoraDashPrefix.size())),
            FluxKleinLoraVariant::Base);
    }
    return flux_klein_lora_model_id(value, default_variant);
}

std::string trim_copy(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string normalize_lora_lookup_text(std::string value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (i + 1 < value.size() &&
            static_cast<unsigned char>(value[i]) == 0xC3 &&
            static_cast<unsigned char>(value[i + 1]) == 0x97) {
            out.push_back('x');
            ++i;
        } else {
            out.push_back(value[i]);
        }
    }
    return trim_copy(std::move(out));
}

std::string lora_loose_lookup_key(std::string value) {
    value = normalize_lora_lookup_text(std::move(value));
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return out;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string url_decode(std::string value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '%' && i + 2 < value.size()) {
            const int high = hex_value(value[i + 1]);
            const int low = hex_value(value[i + 2]);
            if (high >= 0 && low >= 0) {
                out.push_back(static_cast<char>((high << 4) | low));
                i += 2;
                continue;
            }
        }
        out.push_back(value[i] == '+' ? ' ' : value[i]);
    }
    return out;
}


std::optional<double> parse_double_full(std::string value) {
    value = trim_copy(std::move(value));
    if (value.empty()) {
        return std::nullopt;
    }

    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str()) {
        return std::nullopt;
    }
    while (*end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) {
            return std::nullopt;
        }
        ++end;
    }
    return parsed;
}

std::optional<double> parse_elapsed_clock_ms(std::string value) {
    value = trim_copy(std::move(value));
    if (value.empty()) {
        return std::nullopt;
    }

    std::vector<double> parts;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t colon = value.find(':', start);
        const std::size_t end = colon == std::string::npos ? value.size() : colon;
        const auto part = parse_double_full(value.substr(start, end - start));
        if (!part) {
            return std::nullopt;
        }
        parts.push_back(*part);
        if (colon == std::string::npos) {
            break;
        }
        start = colon + 1;
    }

    double seconds = 0.0;
    for (double part : parts) {
        seconds = seconds * 60.0 + part;
    }
    return seconds * 1000.0;
}

std::optional<int> parse_progress_completed(const std::string& record, std::size_t bracket_pos) {
    const std::size_t bar_pos = record.rfind('|', bracket_pos);
    if (bar_pos == std::string::npos) {
        return std::nullopt;
    }

    const std::size_t slash_pos = record.find('/', bar_pos + 1);
    if (slash_pos == std::string::npos || slash_pos > bracket_pos) {
        return std::nullopt;
    }

    std::string value = trim_copy(record.substr(bar_pos + 1, slash_pos - bar_pos - 1));
    if (value.empty()) {
        return std::nullopt;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || parsed <= 0) {
        return std::nullopt;
    }
    return static_cast<int>(parsed);
}

std::optional<double> parse_progress_seconds_per_it(
    const std::string& record,
    std::size_t bracket_pos) {
    const std::size_t comma_pos = record.find(',', bracket_pos);
    if (comma_pos == std::string::npos) {
        return std::nullopt;
    }

    const std::size_t unit_pos = record.find("s/it", comma_pos);
    if (unit_pos == std::string::npos) {
        return std::nullopt;
    }

    return parse_double_full(record.substr(comma_pos + 1, unit_pos - comma_pos - 1));
}

std::optional<double> parse_progress_clock_elapsed_ms(
    const std::string& record,
    std::size_t bracket_pos) {
    const std::size_t less_pos = record.find('<', bracket_pos);
    if (less_pos == std::string::npos) {
        return std::nullopt;
    }
    return parse_elapsed_clock_ms(record.substr(bracket_pos + 1, less_pos - bracket_pos - 1));
}

std::optional<double> parse_progress_elapsed_ms(
    const std::string& record,
    const std::string& stage_name) {
    const std::string prefix = stage_name + ":";
    const std::size_t stage_pos = record.rfind(prefix);
    if (stage_pos == std::string::npos) {
        return std::nullopt;
    }

    const std::size_t bracket_pos = record.find('[', stage_pos);
    if (bracket_pos == std::string::npos) {
        return std::nullopt;
    }

    const auto completed = parse_progress_completed(record, bracket_pos);
    const auto seconds_per_it = parse_progress_seconds_per_it(record, bracket_pos);
    if (completed && seconds_per_it) {
        return static_cast<double>(*completed) * *seconds_per_it * 1000.0;
    }

    return parse_progress_clock_elapsed_ms(record, bracket_pos);
}

std::optional<double> parse_labeled_ms(const std::string& record, const std::string& label) {
    const std::size_t label_pos = record.find(label);
    if (label_pos == std::string::npos) {
        return std::nullopt;
    }

    const std::size_t start = label_pos + label.size();
    const std::size_t unit_pos = record.find("ms", start);
    const std::size_t end = unit_pos == std::string::npos ? record.size() : unit_pos;
    return parse_double_full(record.substr(start, end - start));
}

void parse_worker_output_record(const std::string& record, WorkerTiming& timing) {
    if (const auto ms = parse_progress_elapsed_ms(record, "Text Encoder")) {
        timing.text_encoder_ms = *ms;
    }
    if (const std::size_t stage_pos = record.rfind("Denoising:"); stage_pos != std::string::npos) {
        const std::size_t bracket_pos = record.find('[', stage_pos);
        if (bracket_pos != std::string::npos) {
            if (const auto elapsed_ms = parse_progress_clock_elapsed_ms(record, bracket_pos)) {
                timing.denoising_ms = *elapsed_ms;
            } else if (const auto ms = parse_progress_elapsed_ms(record, "Denoising")) {
                timing.denoising_ms = *ms;
            }
            if (const auto seconds_per_it = parse_progress_seconds_per_it(record, bracket_pos)) {
                timing.denoising_step_ms = *seconds_per_it * 1000.0;
            }
        }
    }
    if (const auto ms = parse_progress_elapsed_ms(record, "VAE Decoder")) {
        timing.vae_decoder_ms = *ms;
    }
    if (const auto ms = parse_labeled_ms(record, "E2E runtime:")) {
        timing.e2e_runtime_ms = *ms;
    }
}

std::optional<std::string> z_image_lora_name_from_model_id(const std::string& model_id) {
    const std::string normalized = trim_copy(url_decode(model_id));
    if (starts_with(normalized, kZImageTurboLoraPrefix)) {
        return trim_copy(normalized.substr(kZImageTurboLoraPrefix.size()));
    }
    if (starts_with(normalized, kZImageTurboQ41LoraPrefix)) {
        return trim_copy(normalized.substr(kZImageTurboQ41LoraPrefix.size()));
    }
    if (starts_with(normalized, kZImageTurboBf16LoraPrefix)) {
        return trim_copy(normalized.substr(kZImageTurboBf16LoraPrefix.size()));
    }
    if (starts_with(normalized, kZImageTurboQ41DisplayLoraPrefix)) {
        return trim_copy(normalized.substr(kZImageTurboQ41DisplayLoraPrefix.size()));
    }
    if (starts_with(normalized, kZImageTurboBf16DisplayLoraPrefix)) {
        return trim_copy(normalized.substr(kZImageTurboBf16DisplayLoraPrefix.size()));
    }
    if (starts_with(normalized, kZImageTurboLoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kZImageTurboLoraDashPrefix.size()));
        if (!name.empty() && name != "q41") {
            return name;
        }
    }
    if (starts_with(normalized, kZImageTurboQ41LoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kZImageTurboQ41LoraDashPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kZImageTurboBf16LoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kZImageTurboBf16LoraDashPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kZImageTurboQ41DisplayLoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kZImageTurboQ41DisplayLoraDashPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kZImageTurboBf16DisplayLoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kZImageTurboBf16DisplayLoraDashPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    return std::nullopt;
}

std::optional<std::string> flux_klein_lora_name_from_model_id(const std::string& model_id) {
    const std::string normalized = trim_copy(url_decode(model_id));
    if (starts_with(normalized, kFluxKleinEditLoraPrefix)) {
        const std::string name = trim_copy(normalized.substr(kFluxKleinEditLoraPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kFluxKleinLoraPrefix)) {
        const std::string name = trim_copy(normalized.substr(kFluxKleinLoraPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kFluxKleinEditLoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kFluxKleinEditLoraDashPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kFluxKleinShortEditLoraPrefix)) {
        const std::string name = trim_copy(normalized.substr(kFluxKleinShortEditLoraPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kFluxKleinShortEditLoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kFluxKleinShortEditLoraDashPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kFluxKleinLoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kFluxKleinLoraDashPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kFluxKleinShortLoraPrefix)) {
        const std::string name = trim_copy(normalized.substr(kFluxKleinShortLoraPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    if (starts_with(normalized, kFluxKleinShortLoraDashPrefix)) {
        const std::string name = trim_copy(normalized.substr(kFluxKleinShortLoraDashPrefix.size()));
        if (!name.empty()) {
            return name;
        }
    }
    return std::nullopt;
}

std::optional<FluxKleinLoraVariant> flux_klein_lora_variant_from_model_id(
    const std::string& model_id) {
    const std::string normalized = trim_copy(url_decode(model_id));
    if (normalized == "flux.2-klein-4B-edit-lora" ||
        normalized == "flux.2-klein-4B-edit (lora)" ||
        normalized == "flux.2-klein-4B-edit(lora)" ||
        normalized == "flux.2-klein-edit-lora" ||
        normalized == "flux.2-klein-edit (lora)" ||
        normalized == "flux.2-klein-edit(lora)" ||
        starts_with(normalized, kFluxKleinEditLoraPrefix) ||
        starts_with(normalized, kFluxKleinEditLoraDashPrefix) ||
        starts_with(normalized, kFluxKleinShortEditLoraPrefix) ||
        starts_with(normalized, kFluxKleinShortEditLoraDashPrefix)) {
        return FluxKleinLoraVariant::Edit;
    }
    if (normalized == "flux.2-klein-4B-lora" ||
        normalized == "flux.2-klein-4B (lora)" ||
        normalized == "flux.2-klein-4B(lora)" ||
        normalized == "flux.2-klein-lora" ||
        normalized == "flux.2-klein (lora)" ||
        normalized == "flux.2-klein(lora)" ||
        starts_with(normalized, kFluxKleinLoraPrefix) ||
        starts_with(normalized, kFluxKleinLoraDashPrefix) ||
        starts_with(normalized, kFluxKleinShortLoraPrefix) ||
        starts_with(normalized, kFluxKleinShortLoraDashPrefix)) {
        return FluxKleinLoraVariant::Base;
    }
    return std::nullopt;
}

std::optional<ZImageLoraVariant> z_image_lora_variant_from_model_id(const std::string& model_id) {
    const std::string normalized = trim_copy(url_decode(model_id));
    if (normalized == "z-image-turbo-bf16-lora" ||
        normalized == kZImageTurboBf16LoraModelId ||
        normalized == "z-image-turbo-bf16 (lora)" ||
        normalized == "z-image-turbo-bf16(lora)" ||
        starts_with(normalized, kZImageTurboBf16LoraPrefix) ||
        starts_with(normalized, kZImageTurboBf16DisplayLoraPrefix) ||
        starts_with(normalized, kZImageTurboBf16LoraDashPrefix) ||
        starts_with(normalized, kZImageTurboBf16DisplayLoraDashPrefix)) {
        return ZImageLoraVariant::Bf16;
    }
    if (normalized == "z-image-turbo-q41-lora" ||
        normalized == kZImageTurboQ41LoraModelId ||
        normalized == "z-image-turbo-q41_lora" ||
        normalized == "z-image-turbo-lora" ||
        normalized == "z-image-turbo-lora-q41" ||
        starts_with(normalized, kZImageTurboLoraPrefix) ||
        starts_with(normalized, kZImageTurboQ41LoraPrefix) ||
        starts_with(normalized, kZImageTurboQ41DisplayLoraPrefix) ||
        starts_with(normalized, kZImageTurboLoraDashPrefix) ||
        starts_with(normalized, kZImageTurboQ41LoraDashPrefix) ||
        starts_with(normalized, kZImageTurboQ41DisplayLoraDashPrefix)) {
        return ZImageLoraVariant::Q41;
    }
    return std::nullopt;
}

ZImageLoraVariant lora_variant_for_request(const std::string& model_id) {
    const auto explicit_variant = z_image_lora_variant_from_model_id(model_id);
    if (explicit_variant) {
        return *explicit_variant;
    }
    return configured_prefers_bf16_lora_ids() ? ZImageLoraVariant::Bf16 : ZImageLoraVariant::Q41;
}

bool is_flux_klein_lora_target(const std::string& raw_target) {
    const auto variant = flux_klein_lora_variant_from_model_id(raw_target);
    return raw_target == "flux.2-klein-4B (lora)" ||
           raw_target == "flux.2-klein-4B(lora)" ||
           raw_target == "flux.2-klein-4B-lora" ||
           raw_target == "flux.2-klein (lora)" ||
           raw_target == "flux.2-klein(lora)" ||
           raw_target == "flux.2-klein-lora" ||
           (variant && *variant == FluxKleinLoraVariant::Base);
}

bool is_flux_klein_edit_lora_target(const std::string& raw_target) {
    const auto variant = flux_klein_lora_variant_from_model_id(raw_target);
    return raw_target == "flux.2-klein-4B-edit (lora)" ||
           raw_target == "flux.2-klein-4B-edit(lora)" ||
           raw_target == "flux.2-klein-4B-edit-lora" ||
           raw_target == "flux.2-klein-edit (lora)" ||
           raw_target == "flux.2-klein-edit(lora)" ||
           raw_target == "flux.2-klein-edit-lora" ||
           (variant && *variant == FluxKleinLoraVariant::Edit);
}

bool is_z_image_bf16_lora_target(const std::string& raw_target) {
    const auto variant = z_image_lora_variant_from_model_id(raw_target);
    return raw_target == "z-image-turbo-bf16 (lora)" ||
           raw_target == "z-image-turbo-bf16(lora)" ||
           raw_target == "z-image-turbo-bf16-lora" ||
           (variant && *variant == ZImageLoraVariant::Bf16);
}

bool is_z_image_q41_lora_target(const std::string& raw_target) {
    const auto variant = z_image_lora_variant_from_model_id(raw_target);
    return raw_target == "z-image-turbo-q41-lora" ||
           raw_target == "z-image-turbo-q41_lora" ||
           raw_target == "z-image-turbo-lora" ||
           raw_target == "z-image-turbo-lora-q41" ||
           (variant && *variant == ZImageLoraVariant::Q41);
}

bool target_accepts_gguf_options(const std::string& raw_target) {
    return is_z_image_q41_base_model_id(raw_target) ||
           is_z_image_q41_lora_target(raw_target);
}

bool target_accepts_lora_options(const std::string& raw_target) {
    return raw_target == "flux.2-klein-4B" ||
           raw_target == "flux.2-klein" ||
           raw_target == "flux.2-klein-4B-edit" ||
           raw_target == "flux.2-klein-edit" ||
           is_flux_klein_lora_target(raw_target) ||
           is_flux_klein_edit_lora_target(raw_target) ||
           is_z_image_bf16_base_model_id(raw_target) ||
           is_z_image_bf16_lora_target(raw_target) ||
           is_z_image_q41_lora_target(raw_target);
}

bool target_accepts_input_images(const std::string& raw_target) {
    return raw_target == "flux.2-klein-4B-edit" ||
           raw_target == "flux.2-klein-edit" ||
           is_flux_klein_edit_lora_target(raw_target);
}

// Only the bf16 and flux klein workers need an explicit --lora_rank; the
// q41 fused worker defaults to its supported rank.
bool target_accepts_bf16_lora_extras(const std::string& raw_target) {
    return raw_target == "flux.2-klein-4B" ||
           raw_target == "flux.2-klein" ||
           raw_target == "flux.2-klein-4B-edit" ||
           raw_target == "flux.2-klein-edit" ||
           is_flux_klein_lora_target(raw_target) ||
           is_flux_klein_edit_lora_target(raw_target) ||
           is_z_image_bf16_base_model_id(raw_target) ||
           is_z_image_bf16_lora_target(raw_target);
}

void append_arg_if_not_empty(
    std::vector<std::string>& args,
    const std::string& name,
    const std::string& value) {
    if (!value.empty()) {
        args.push_back(name);
        args.push_back(value);
    }
}

void append_gguf_args(std::vector<std::string>& args, const GenParams& params) {
    const std::string gguf_path =
        !params.gguf_path.empty() ? params.gguf_path : config::gguf_path;
    const std::string gguf_repo =
        !params.gguf_repo.empty() ? params.gguf_repo : config::gguf_repo;
    const std::string gguf_file =
        !params.gguf_file.empty() ? params.gguf_file : config::gguf_file;
    const std::string gguf_revision =
        !params.gguf_revision.empty() ? params.gguf_revision : config::gguf_revision;
    const std::string gguf_hf_token =
        !params.gguf_hf_token.empty() ? params.gguf_hf_token : config::gguf_hf_token;

    append_arg_if_not_empty(args, "--gguf_path", gguf_path);
    append_arg_if_not_empty(args, "--gguf_repo", gguf_repo);
    append_arg_if_not_empty(args, "--gguf_file", gguf_file);
    append_arg_if_not_empty(args, "--gguf_revision", gguf_revision);
    append_arg_if_not_empty(args, "--gguf_hf_token", gguf_hf_token);

    if (config::force_gguf_download || params.force_gguf_download) {
        args.push_back("--force_gguf_download");
    }
}

void append_lora_args(
    std::vector<std::string>& args,
    const GenParams& params,
    const std::string& raw_target) {
    const std::string lora_dir = !params.lora_dir.empty() ? params.lora_dir : config::lora_dir;
    const std::string lora_source =
        !params.lora_source.empty() ? params.lora_source : config::lora_source;
    const std::string lora_file =
        !params.lora_file.empty() ? params.lora_file : config::lora_file;
    const int lora_rank = params.lora_rank > 0 ? params.lora_rank : config::lora_rank;
    const float lora_scale = params.lora_scale;

    append_arg_if_not_empty(args, "--lora_dir", lora_dir);
    append_arg_if_not_empty(args, "--lora_source", lora_source);

    append_arg_if_not_empty(args, "--lora_file", lora_file);

    const bool lora_requested = !lora_dir.empty() || !lora_source.empty() || !lora_file.empty();

    // --lora_rank only exists on the bf16/flux klein workers; the q41 fused
    // worker defaults to its supported rank. Do not pass it for plain bf16.
    if (lora_requested && target_accepts_bf16_lora_extras(raw_target) && lora_rank > 0) {
        args.push_back("--lora_rank");
        args.push_back(std::to_string(lora_rank));
    }
    if (lora_requested && target_accepts_bf16_lora_extras(raw_target) && lora_scale > 0.0f) {
        args.push_back("--lora_scale");
        args.push_back(std::to_string(lora_scale));
    }

    // if (config::lora_force || params.lora_force) {
    //     args.push_back("--lora_force");
    // }
    // if (config::lora_keep_cache || params.lora_keep_cache) {
    //     args.push_back("--lora_keep_cache");
    // }
}

bool is_lora_bin_file_name(const std::string& name) {
    return name.find("_lora_A_") != std::string::npos ||
           name.find("_lora_B_") != std::string::npos;
}

bool directory_contains_lora_bins(const fs::path& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return false;
    }

    for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        std::error_code entry_ec;
        if (!it->is_regular_file(entry_ec)) {
            continue;
        }
        if (is_lora_bin_file_name(it->path().filename().string())) {
            return true;
        }
    }

    return false;
}

bool is_known_weights_family_dir(const std::string& dirname) {
    return dirname == "FLUX.2-klein-4B" || dirname == "Z-Image-Turbo";
}

std::string configured_weights_root_path() {
    fs::path root = config::weights_path;
    if (is_known_weights_family_dir(root.filename().string())) {
        return root.parent_path().string();
    }
    return root.string();
}

std::string configured_family_weights_path(const char* model_dir_name) {
    return (fs::path(configured_weights_root_path()) / model_dir_name).string();
}

std::string configured_family_lora_catalog_dir(const char* model_dir_name) {
    if (!config::lora_catalog_dir.empty()) {
        return config::lora_catalog_dir;
    }

    return configured_family_weights_path(model_dir_name);
}

bool target_uses_flux_klein_weights(const std::string& raw_target) {
    return raw_target == "flux.2-klein-4B" || raw_target == "flux.2-klein" ||
           raw_target == "flux.2-klein-4B-edit" || raw_target == "flux.2-klein-edit" ||
           raw_target == "flux.2-klein-4B-lora" || raw_target == "flux.2-klein-lora" ||
           raw_target == "flux.2-klein-4B-edit-lora" || raw_target == "flux.2-klein-edit-lora";
}

bool target_uses_z_image_weights(const std::string& raw_target) {
    return starts_with(raw_target, "z-image-turbo");
}

std::string resolve_worker_weights_path(const std::string& raw_target) {
    (void)raw_target;
    return configured_weights_root_path();
}

std::string configured_z_image_lora_catalog_dir() {
    return configured_family_lora_catalog_dir("Z-Image-Turbo");
}

std::string configured_flux_klein_lora_catalog_dir() {
    return configured_family_lora_catalog_dir("FLUX.2-klein-4B");
}

bool file_exists(const fs::path& path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

std::vector<fs::path> z_image_bf16_required_files(const fs::path& z_image_dir) {
    return {
        z_image_dir / "denoising" / "bf16_parts" / "shapes.json",
        z_image_dir / "denoising" / "bf16_parts" / "all_final_layer_2-1_linear_weight_bf16_u16.bin",
        z_image_dir / "denoising" / "bf16_parts" / "t_embedder_mlp_0_weight_bf16_u16.bin",
        z_image_dir / "denoising" / "bf16_parts" / "cap_embedder_1_weight_bf16_u16.bin",
        z_image_dir / "text_encoder" / "tokenizer.json",
        z_image_dir / "text_encoder" / "embed_tokens_weight_bf16_u16.bin",
        z_image_dir / "vae_decoder" / "mid_block-resnets-0-conv1-weight_u16.bin",
    };
}

std::vector<fs::path> missing_z_image_bf16_weight_files(const fs::path& z_image_dir) {
    std::vector<fs::path> missing;
    for (const auto& path : z_image_bf16_required_files(z_image_dir)) {
        if (!file_exists(path)) {
            missing.push_back(path);
        }
    }
    return missing;
}

bool z_image_bf16_weights_are_ready(const fs::path& z_image_dir) {
    return missing_z_image_bf16_weight_files(z_image_dir).empty();
}

std::string describe_missing_files(const std::vector<fs::path>& missing) {
    if (missing.empty()) {
        return {};
    }
    std::ostringstream out;
    out << " Missing required files:";
    for (const auto& path : missing) {
        out << "\n  - " << path.string();
    }
    return out.str();
}

void ensure_z_image_bf16_weights(const std::string& raw_target) {
    if (!target_uses_z_image_weights(raw_target)) {
        return;
    }

    const fs::path z_image_dir = fs::path(configured_weights_root_path()) / "Z-Image-Turbo";
    if (z_image_bf16_weights_are_ready(z_image_dir)) {
        return;
    }

    if (!config::auto_download_z_image_bf16) {
        const auto missing = missing_z_image_bf16_weight_files(z_image_dir);
        throw std::runtime_error(
            "Z-Image BF16 weights are missing under " + z_image_dir.string() + "." +
            describe_missing_files(missing));
    }

    fs::path cache_dir = config::z_image_bf16_cache_dir;
    if (!cache_dir.empty() && cache_dir.is_relative()) {
        cache_dir = fs::path(config::workdir) / cache_dir;
    }

    prepare_z_image_bf16_weights_cpp(ZImageBf16ExportOptions{
        config::z_image_bf16_repo,
        config::z_image_bf16_revision,
        config::z_image_bf16_hf_token,
        cache_dir,
        z_image_dir,
        config::force_z_image_bf16_download,
    });

    if (z_image_bf16_weights_are_ready(z_image_dir)) {
        return;
    }

    const auto missing = missing_z_image_bf16_weight_files(z_image_dir);
    throw std::runtime_error(
        "Z-Image BF16 export completed, but required files are missing under " +
        z_image_dir.string() + "." +
        describe_missing_files(missing));
}

std::vector<fs::path> flux_klein_bf16_required_files(const fs::path& flux_dir) {
    return {
        flux_dir / "denoising" / "context_embedder_weight_bf16_u16.bin",
        flux_dir / "denoising" / "transformer_blocks_0_attn_add_q_proj_weight_bf16_u16.bin",
        flux_dir / "denoising" / "single_transformer_blocks_0_attn_to_qnorm_weight_bf16_u16.bin",
        flux_dir / "denoising" / "single_transformer_blocks_0_attn_to_v_weight_bf16_u16.bin",
        flux_dir / "denoising" / "single_transformer_blocks_0_attn_to_qkv_mlp_proj_weight_bf16_u16.bin",
        flux_dir / "denoising" / "final_layer_adaLN_modulation_1_weight_bf16_u16.bin",
        flux_dir / "denoising" / "final_layer_linear_weight_bf16_u16.bin",
        flux_dir / "text_encoder" / "tokenizer.json",
        flux_dir / "text_encoder" / "embed_tokens_weight_bf16_u16.bin",
        flux_dir / "text_encoder" / "norm_weight_bf16_u16.bin",
        flux_dir / "vae" / "decoder_conv_in_weight_bf16_u16.bin",
        flux_dir / "vae" / "vae_mid_b1_attn1_q_weights_transposed.bin",
    };
}

std::vector<fs::path> missing_flux_klein_bf16_weight_files(const fs::path& flux_dir) {
    std::vector<fs::path> missing;
    for (const auto& path : flux_klein_bf16_required_files(flux_dir)) {
        if (!file_exists(path)) {
            missing.push_back(path);
        }
    }
    return missing;
}

bool flux_klein_bf16_weights_are_ready(const fs::path& flux_dir) {
    return missing_flux_klein_bf16_weight_files(flux_dir).empty();
}

void ensure_flux_klein_bf16_weights(const std::string& raw_target) {
    if (!target_uses_flux_klein_weights(raw_target)) {
        return;
    }

    const fs::path flux_dir = fs::path(configured_weights_root_path()) / "FLUX.2-klein-4B";
    if (flux_klein_bf16_weights_are_ready(flux_dir)) {
        return;
    }

    if (!config::auto_download_flux_klein_bf16) {
        const auto missing = missing_flux_klein_bf16_weight_files(flux_dir);
        throw std::runtime_error(
            "FLUX.2-klein-4B BF16 weights are missing under " + flux_dir.string() + "." +
            describe_missing_files(missing));
    }

    fs::path cache_dir = config::flux_klein_bf16_cache_dir;
    if (!cache_dir.empty() && cache_dir.is_relative()) {
        cache_dir = fs::path(config::workdir) / cache_dir;
    }

    prepare_flux_klein_bf16_weights_cpp(FluxKleinBf16DownloadOptions{
        config::flux_klein_bf16_repo,
        config::flux_klein_bf16_revision,
        config::flux_klein_bf16_hf_token,
        cache_dir,
        flux_dir,
        config::force_flux_klein_bf16_download,
    });

    if (flux_klein_bf16_weights_are_ready(flux_dir)) {
        return;
    }

    const auto missing = missing_flux_klein_bf16_weight_files(flux_dir);
    throw std::runtime_error(
        "FLUX.2-klein-4B BF16 download completed, but required files are missing under " +
        flux_dir.string() + "." +
        describe_missing_files(missing));
}

std::string get_json_string(const json& item, const char* key) {
    if (item.contains(key) && item[key].is_string()) {
        return item[key].get<std::string>();
    }
    return {};
}

int get_json_int(const json& item, const char* key) {
    if (item.contains(key) && item[key].is_number_integer()) {
        return item[key].get<int>();
    }
    return 0;
}

int first_json_int(
    const json& item,
    const char* first,
    const char* second = nullptr,
    const char* third = nullptr) {
    int value = get_json_int(item, first);
    if (value > 0 || second == nullptr) return value;
    value = get_json_int(item, second);
    if (value > 0 || third == nullptr) return value;
    return get_json_int(item, third);
}

float get_json_float(const json& item, const char* key) {
    if (item.contains(key) && item[key].is_number()) {
        return item[key].get<float>();
    }
    return 0.0f;
}

float first_json_float(
    const json& item,
    const char* first,
    const char* second = nullptr,
    const char* third = nullptr) {
    float value = get_json_float(item, first);
    if (value > 0.0f || second == nullptr) return value;
    value = get_json_float(item, second);
    if (value > 0.0f || third == nullptr) return value;
    return get_json_float(item, third);
}

std::string first_json_string(
    const json& item,
    const char* first,
    const char* second = nullptr,
    const char* third = nullptr) {
    std::string value = get_json_string(item, first);
    if (!value.empty() || second == nullptr) return value;
    value = get_json_string(item, second);
    if (!value.empty() || third == nullptr) return value;
    return get_json_string(item, third);
}

std::string safe_lora_id(std::string value) {
    std::string out;
    bool last_was_dash = false;
    for (unsigned char ch : value) {
        const char c = static_cast<char>(std::tolower(ch));
        if (std::isalnum(ch) || c == '_' || c == '.') {
            out.push_back(c);
            last_was_dash = false;
        } else if (!last_was_dash) {
            out.push_back('-');
            last_was_dash = true;
        }
    }
    while (!out.empty() && out.front() == '-') out.erase(out.begin());
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "lora" : out;
}

std::string exporter_safe_lora_dir_name(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    bool last_was_sep = false;
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_') {
            out.push_back(static_cast<char>(c));
            last_was_sep = false;
        } else if (!last_was_sep) {
            out.push_back('_');
            last_was_sep = true;
        }
    }
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? "lora" : out;
}

std::string hf_repo_id_from_lora_source(std::string source) {
    source = trim_copy(url_decode(std::move(source)));
    const auto query_pos = source.find_first_of("?#");
    if (query_pos != std::string::npos) {
        source.erase(query_pos);
    }

    const std::string marker = "huggingface.co/";
    const auto marker_pos = source.find(marker);
    if (marker_pos != std::string::npos) {
        source = source.substr(marker_pos + marker.size());
        while (!source.empty() && source.front() == '/') source.erase(source.begin());
        const auto first_slash = source.find('/');
        if (first_slash == std::string::npos) {
            return source;
        }
        const auto second_slash = source.find('/', first_slash + 1);
        if (second_slash == std::string::npos) {
            return source;
        }
        return source.substr(0, second_slash);
    }

    return source;
}

std::string exported_lora_dir_name(const LoraModelEntry& entry) {
    if (entry.lora_source.empty() || entry.lora_file.empty()) {
        return {};
    }

    const std::string repo_id = hf_repo_id_from_lora_source(entry.lora_source);
    const std::string file_stem = fs::path(entry.lora_file).stem().string();
    if (repo_id.empty() || file_stem.empty()) {
        return {};
    }
    return exporter_safe_lora_dir_name(repo_id + "__" + file_stem);
}

std::string exported_lora_dir_prefix(const LoraModelEntry& entry) {
    if (entry.lora_source.empty()) {
        return {};
    }

    const std::string repo_id = hf_repo_id_from_lora_source(entry.lora_source);
    if (repo_id.empty()) {
        return {};
    }
    return exporter_safe_lora_dir_name(repo_id) + "__";
}

bool vector_contains_string(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::vector<std::string> json_listed_export_dir_names(
    const std::vector<LoraModelEntry>& entries) {
    std::vector<std::string> names;
    for (const auto& entry : entries) {
        std::string dirname = exported_lora_dir_name(entry);
        if (dirname.empty() || vector_contains_string(names, dirname)) {
            continue;
        }
        names.push_back(std::move(dirname));
    }
    return names;
}

std::vector<std::string> json_listed_export_dir_prefixes(
    const std::vector<LoraModelEntry>& entries) {
    std::vector<std::string> prefixes;
    for (const auto& entry : entries) {
        std::string prefix = exported_lora_dir_prefix(entry);
        if (prefix.empty() || vector_contains_string(prefixes, prefix)) {
            continue;
        }
        prefixes.push_back(std::move(prefix));
    }
    return prefixes;
}

bool is_json_listed_export_dir(
    const std::vector<std::string>& json_export_dir_names,
    const std::vector<std::string>& json_export_dir_prefixes,
    const std::string& dirname) {
    if (vector_contains_string(json_export_dir_names, dirname)) {
        return true;
    }
    for (const auto& prefix : json_export_dir_prefixes) {
        if (starts_with(dirname, prefix)) {
            return true;
        }
    }
    return false;
}

void add_lora_entry_if_new(
    std::vector<LoraModelEntry>& entries,
    LoraModelEntry entry) {
    if (entry.model_id.empty()) {
        return;
    }
    const auto existing = std::find_if(entries.begin(), entries.end(), [&](const auto& candidate) {
        return candidate.model_id == entry.model_id;
    });
    if (existing == entries.end()) {
        entries.push_back(std::move(entry));
    }
}

void append_lora_model_list_entries(
    std::vector<LoraModelEntry>& entries,
    const std::string& configured_list,
    std::optional<ZImageLoraVariant> default_z_image_variant,
    std::optional<FluxKleinLoraVariant> default_flux_variant = std::nullopt) {
    if (configured_list.empty()) {
        return;
    }

    fs::path list_path = configured_list;
    if (list_path.is_relative()) {
        list_path = fs::absolute(list_path);
    }

    std::error_code ec;
    if (!fs::exists(list_path, ec)) {
        std::cerr << "LoRA model list does not exist: " << list_path << "\n";
        return;
    }

    try {
        std::ifstream in(list_path);
        json parsed = json::parse(in, nullptr, true, true);

        const json* models = nullptr;
        if (parsed.is_array()) {
            models = &parsed;
        } else if (parsed.contains("models") && parsed["models"].is_array()) {
            models = &parsed["models"];
        }
        if (models == nullptr) {
            std::cerr << "LoRA model list must be an array or contain a models array: "
                      << list_path << "\n";
            return;
        }

        for (const auto& item : *models) {
            if (!item.is_object()) {
                continue;
            }
            if (item.contains("enabled") && item["enabled"].is_boolean() &&
                !item["enabled"].get<bool>()) {
                continue;
            }

            const std::string source = first_json_string(item, "source", "lora_source", "repo");
            std::string lora_dir = first_json_string(item, "lora_dir", "dir", "export_root");
            const std::string file = first_json_string(item, "file", "lora_file", "weight_file");
            const int lora_rank = first_json_int(item, "lora_rank", "rank");
            const float lora_scale = first_json_float(item, "lora_scale", "scale");
            std::string id = first_json_string(item, "model_id", "id", "name");

            if (source.empty() && lora_dir.empty()) {
                continue;
            }
            if (id.empty()) {
                id = safe_lora_id(!source.empty() ? source : lora_dir);
            }
            const auto explicit_flux_variant = flux_klein_lora_variant_from_model_id(id);
            const bool flux_entry = default_flux_variant.has_value() ||
                explicit_flux_variant.has_value() ||
                has_flux_klein_lora_prefix(id);
            if (!source.empty() && lora_dir.empty()) {
                lora_dir = flux_entry
                    ? configured_flux_klein_lora_catalog_dir()
                    : configured_z_image_lora_catalog_dir();
            }

            const std::string model_id = flux_entry
                ? normalize_flux_klein_lora_model_id(
                      id,
                      default_flux_variant.value_or(
                          explicit_flux_variant.value_or(FluxKleinLoraVariant::Base)))
                : normalize_z_image_lora_model_id(id, default_z_image_variant);

            add_lora_entry_if_new(
                entries,
                {model_id,
                 lora_dir,
                 source,
                 file,
                 lora_rank,
                 lora_scale});
        }
    } catch (const std::exception& e) {
        std::cerr << "Could not read LoRA model list " << list_path << ": "
                  << e.what() << "\n";
    }
}

std::string default_lora_model_list(
    const std::string& configured_list,
    const char* default_jsonc,
    const char* default_json) {
    if (!configured_list.empty()) {
        return configured_list;
    }

    const fs::path jsonc_path = fs::path(default_jsonc);
    std::error_code ec;
    if (fs::exists(jsonc_path, ec)) {
        return jsonc_path.string();
    }

    const fs::path json_path = fs::path(default_json);
    ec.clear();
    if (fs::exists(json_path, ec)) {
        return json_path.string();
    }

    return {};
}

std::string default_q41_lora_model_list() {
    return default_lora_model_list(
        config::lora_model_list_q41,
        "./z_image_q41_lora_models.jsonc",
        "./z_image_q41_lora_models.json");
}

std::string default_bf16_lora_model_list() {
    return default_lora_model_list(
        config::lora_model_list_bf16,
        "./z_image_bf16_lora_models.jsonc",
        "./z_image_bf16_lora_models.json");
}

std::string default_flux_lora_model_list() {
    return default_lora_model_list(
        config::lora_model_list_flux,
        "./flux_klein_lora_models.jsonc",
        "./flux_klein_lora_models.json");
}

std::string default_flux_edit_lora_model_list() {
    return default_lora_model_list(
        config::lora_model_list_flux_edit,
        "./flux_klein_edit_lora_models.jsonc",
        "./flux_klein_edit_lora_models.json");
}

std::vector<LoraModelEntry> read_lora_model_list_entries() {
    std::vector<LoraModelEntry> entries;
    append_lora_model_list_entries(entries, default_q41_lora_model_list(), ZImageLoraVariant::Q41);
    append_lora_model_list_entries(entries, default_bf16_lora_model_list(), ZImageLoraVariant::Bf16);
    append_lora_model_list_entries(
        entries, default_flux_lora_model_list(), std::nullopt, FluxKleinLoraVariant::Base);
    append_lora_model_list_entries(
        entries, default_flux_edit_lora_model_list(), std::nullopt, FluxKleinLoraVariant::Edit);
    append_lora_model_list_entries(entries, config::lora_model_list, std::nullopt);
    return entries;
}

bool has_configured_lora_model_list() {
    return !config::lora_model_list.empty() ||
           !default_q41_lora_model_list().empty() ||
           !default_bf16_lora_model_list().empty() ||
           !default_flux_lora_model_list().empty() ||
           !default_flux_edit_lora_model_list().empty();
}

bool is_base_z_image_lora_model_id(const std::string& model_id) {
    const auto variant = z_image_lora_variant_from_model_id(model_id);
    const auto lora_name = z_image_lora_name_from_model_id(model_id);
    return variant && (!lora_name || lora_name->empty());
}

bool is_base_flux_klein_lora_model_id(const std::string& model_id) {
    return model_id == "flux.2-klein-4B-lora" ||
           model_id == "flux.2-klein-4B (lora)" ||
           model_id == "flux.2-klein-4B(lora)" ||
           model_id == "flux.2-klein-lora" ||
           model_id == "flux.2-klein (lora)" ||
           model_id == "flux.2-klein(lora)" ||
           model_id == "flux.2-klein-4B-edit-lora" ||
           model_id == "flux.2-klein-4B-edit (lora)" ||
           model_id == "flux.2-klein-4B-edit(lora)" ||
           model_id == "flux.2-klein-edit-lora" ||
           model_id == "flux.2-klein-edit (lora)" ||
           model_id == "flux.2-klein-edit(lora)";
}

std::string lora_lookup_name_from_model_id(const std::string& model_id) {
    const std::string normalized = trim_copy(url_decode(model_id));
    if (is_base_z_image_lora_model_id(normalized) ||
        is_base_flux_klein_lora_model_id(normalized)) {
        return {};
    }
    if (const auto flux_name = flux_klein_lora_name_from_model_id(normalized)) {
        return *flux_name;
    }
    const auto prefixed_name = z_image_lora_name_from_model_id(normalized);
    return prefixed_name ? *prefixed_name : normalized;
}

bool lora_model_family_matches_request(
    const std::string& requested_model_id,
    const std::string& candidate_model_id) {
    const auto requested_flux_variant = flux_klein_lora_variant_from_model_id(requested_model_id);
    if (requested_flux_variant) {
        const auto candidate_flux_variant = flux_klein_lora_variant_from_model_id(candidate_model_id);
        return candidate_flux_variant && *candidate_flux_variant == *requested_flux_variant;
    }

    const auto requested_z_image_variant = z_image_lora_variant_from_model_id(requested_model_id);
    if (requested_z_image_variant) {
        const auto candidate_z_image_variant = z_image_lora_variant_from_model_id(candidate_model_id);
        return candidate_z_image_variant && *candidate_z_image_variant == *requested_z_image_variant;
    }

    return true;
}

bool has_direct_lora_options(const GenParams& params) {
    return !params.lora_dir.empty() ||
           !params.lora_source.empty() ||
           !config::lora_dir.empty() ||
           !config::lora_source.empty();
}

std::optional<LoraModelEntry> default_lora_entry_for_base_model_id(
    const std::string& model_id) {
    const std::string normalized = trim_copy(url_decode(model_id));
    if (!is_base_z_image_lora_model_id(normalized) &&
        !is_base_flux_klein_lora_model_id(normalized)) {
        return std::nullopt;
    }

    for (const auto& entry : list_lora_model_entries()) {
        if (lora_model_family_matches_request(normalized, entry.model_id)) {
            return entry;
        }
    }

    return std::nullopt;
}

std::vector<std::string> lora_prompt_match_tokens(std::string value) {
    value = normalize_lora_lookup_text(std::move(value));
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
        if (!vector_contains_string(ignored, token) && !vector_contains_string(tokens, token)) {
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

std::optional<LoraModelEntry> lora_entry_for_prompt_match(
    const std::string& model_id,
    const std::string& prompt) {
    const std::string normalized_request = trim_copy(url_decode(model_id));
    std::optional<LoraModelEntry> best_entry;
    int best_score = 0;
    bool tied = false;

    for (const auto& entry : list_lora_model_entries()) {
        if (!lora_model_family_matches_request(normalized_request, entry.model_id)) {
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

    if (best_score > 0 && !tied) {
        return best_entry;
    }
    return std::nullopt;
}

std::optional<LoraModelEntry> lora_entry_for_catalog_model_id(const std::string& model_id) {
    const std::string normalized_request = trim_copy(url_decode(model_id));
    const std::string lookup_request = normalize_lora_lookup_text(normalized_request);
    for (const auto& entry : list_lora_model_entries()) {
        if (entry.model_id == normalized_request ||
            normalize_lora_lookup_text(entry.model_id) == lookup_request) {
            return entry;
        }
    }

    const std::string requested_name =
        normalize_lora_lookup_text(lora_lookup_name_from_model_id(normalized_request));
    if (requested_name.empty()) {
        return std::nullopt;
    }

    const std::string requested_slug = safe_lora_id(requested_name);
    const std::string requested_key = lora_loose_lookup_key(requested_name);
    for (const auto& entry : list_lora_model_entries()) {
        if (!lora_model_family_matches_request(normalized_request, entry.model_id)) {
            continue;
        }

        const std::string entry_name =
            normalize_lora_lookup_text(lora_lookup_name_from_model_id(entry.model_id));
        if (entry_name.empty()) {
            continue;
        }
        if (entry_name == requested_name ||
            safe_lora_id(entry_name) == requested_slug ||
            lora_loose_lookup_key(entry_name) == requested_key) {
            return entry;
        }
    }

    return std::nullopt;
}

bool resolve_z_image_worker_paths(const std::string& raw_target, WorkerPaths& paths) {
    const std::string z_image_npu_target = "z-image-turbo";

    if (is_z_image_q41_base_model_id(raw_target)) {
        paths.npu_files = config::npu_base_dir + "/" + z_image_npu_target;
        paths.exe = config::exe_base_dir + "/" + z_image_npu_target + "/q41/run.exe";
        return true;
    }

    if (is_z_image_q41_lora_target(raw_target)) {
        paths.npu_files = config::npu_base_dir + "/" + z_image_npu_target;
        paths.exe = config::exe_base_dir + "/" + z_image_npu_target + "/lora_add_q41/run.exe";
        return true;
    }

    if (is_z_image_bf16_base_model_id(raw_target) || is_z_image_bf16_lora_target(raw_target)) {
        paths.npu_files = config::npu_base_dir + "/" + z_image_npu_target;
        paths.exe = config::exe_base_dir + "/" + z_image_npu_target + "/bf16/run.exe";
        return true;
    }

    return false;
}

}  // namespace

std::vector<LoraModelEntry> list_lora_model_entries() {
    std::vector<LoraModelEntry> entries = read_lora_model_list_entries();
    const std::vector<std::string> json_export_dir_names = json_listed_export_dir_names(entries);
    const std::vector<std::string> json_export_dir_prefixes = json_listed_export_dir_prefixes(entries);
    const std::string catalog_dir = configured_z_image_lora_catalog_dir();
    const bool discover_catalog_dirs = !has_configured_lora_model_list();
    if (discover_catalog_dirs && !catalog_dir.empty()) {
        const fs::path root = fs::path(catalog_dir);
        std::error_code ec;
        if (fs::is_directory(root, ec)) {
            if (directory_contains_lora_bins(root)) {
                const std::string lora_name = root.filename().string();
                if (!lora_name.empty() && !is_json_listed_export_dir(json_export_dir_names, json_export_dir_prefixes, lora_name)) {
                    add_lora_entry_if_new(
                        entries,
                        {z_image_lora_model_id(lora_name), root.string(), {}, {}});
                }
            }

            for (fs::directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
                 !ec && it != end;
                 it.increment(ec)) {
                std::error_code entry_ec;
                if (!it->is_directory(entry_ec)) {
                    continue;
                }

                const fs::path child = it->path();
                if (!directory_contains_lora_bins(child)) {
                    continue;
                }

                const std::string lora_name = child.filename().string();
                if (!lora_name.empty() && !is_json_listed_export_dir(json_export_dir_names, json_export_dir_prefixes, lora_name)) {
                    add_lora_entry_if_new(
                        entries,
                        {z_image_lora_model_id(lora_name), child.string(), {}, {}});
                }
            }
        }
    }

    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.model_id < b.model_id;
    });
    return entries;
}

std::optional<LoraModelEntry> resolve_lora_model_entry(const std::string& model_id) {
    return lora_entry_for_catalog_model_id(model_id);
}

void ensure_model_weights_ready(const std::string& model_id) {
    const std::string raw_target = normalize_model_target(model_id);
    ensure_z_image_bf16_weights(raw_target);
    ensure_flux_klein_bf16_weights(raw_target);
}

std::string normalize_model_target(std::string raw_target) {
    if (raw_target == "local-model-1" || raw_target == "edit_model" || raw_target.empty() ||
        (!config::served_model_id.empty() && raw_target == config::served_model_id)) {
        raw_target = config::model_id;
    }

    const size_t colon_pos = raw_target.find(":edit");
    if (colon_pos != std::string::npos) {
        raw_target.replace(colon_pos, 5, "-edit");
    }

    if (raw_target == "flux.2-klein") {
        raw_target = "flux.2-klein-4B";
    } else if (raw_target == "flux.2-klein-edit") {
        raw_target = "flux.2-klein-4B-edit";
    }

    if (is_flux_klein_edit_lora_target(raw_target)) {
        raw_target = "flux.2-klein-4B-edit";
    } else if (is_flux_klein_lora_target(raw_target)) {
        raw_target = "flux.2-klein-4B";
    }

    if (is_z_image_q41_base_model_id(raw_target)) {
        raw_target = "z-image-turbo-q41";
    } else if (is_z_image_bf16_base_model_id(raw_target)) {
        raw_target = "z-image-turbo-bf16";
    }

    if (is_z_image_bf16_lora_target(raw_target)) {
        raw_target = "z-image-turbo-bf16-lora";
    }

    const auto lora_variant = z_image_lora_variant_from_model_id(raw_target);
    if (lora_variant && *lora_variant == ZImageLoraVariant::Bf16) {
        raw_target = "z-image-turbo-bf16-lora";
    } else if (lora_variant && *lora_variant == ZImageLoraVariant::Q41) {
        raw_target = "z-image-turbo-lora";
    }

    return raw_target;
}

WorkerPaths resolve_worker_paths(const std::string& raw_target) {
    WorkerPaths paths;

    if (raw_target == "flux.2-klein-4B-edit" || raw_target == "flux.2-klein-4B-edit-lora") {
        paths.npu_files = config::npu_base_dir + "/flux.2-klein-4B";
        paths.exe = config::exe_base_dir + "/flux.2-klein-4B-edit/run.exe";
    } else if (raw_target == "flux.2-klein-4B-lora") {
        paths.npu_files = config::npu_base_dir + "/flux.2-klein-4B";
        paths.exe = config::exe_base_dir + "/flux.2-klein-4B/run.exe";
    } else if (resolve_z_image_worker_paths(raw_target, paths)) {
        // Handled by the variant-aware Z-Image resolver above.
    } else {
        paths.npu_files = config::npu_base_dir + "/" + raw_target;
        paths.exe = config::exe_base_dir + "/" + raw_target + "/run.exe";
    }

    if (!config::custom_exe.empty()) {
        paths.exe = config::custom_exe;
    }

    return paths;
}

WorkerResult run_worker(const GenParams& p) {
    ensure_output_dir();

    GenParams params = p;
    if (params.model == "local-model-1" || params.model == "edit_model" || params.model.empty() ||
        (!config::served_model_id.empty() && params.model == config::served_model_id)) {
        params.model = config::model_id;
    }

    const std::string requested_model_id = params.model;
    std::optional<LoraModelEntry> selected_lora = lora_entry_for_catalog_model_id(params.model);
    const bool is_generic_flux_lora = flux_klein_lora_variant_from_model_id(params.model).has_value() &&
        lora_lookup_name_from_model_id(params.model).empty();
    bool selected_prompt_lora = false;
    if (!selected_lora && !has_direct_lora_options(params) && is_generic_flux_lora) {
        selected_lora = lora_entry_for_prompt_match(params.model, params.prompt);
        selected_prompt_lora = selected_lora.has_value();
    }
    bool selected_default_lora = false;
    if (!selected_lora && !has_direct_lora_options(params)) {
        selected_lora = default_lora_entry_for_base_model_id(params.model);
        selected_default_lora = selected_lora.has_value();
    }

    if (selected_lora) {
        const auto flux_variant = flux_klein_lora_variant_from_model_id(selected_lora->model_id);
        if (flux_variant) {
            params.model = *flux_variant == FluxKleinLoraVariant::Edit
                ? "flux.2-klein-4B-edit-lora"
                : "flux.2-klein-4B-lora";
        } else {
            const ZImageLoraVariant lora_variant = lora_variant_for_request(selected_lora->model_id);
            params.model = lora_variant == ZImageLoraVariant::Bf16
                ? "z-image-turbo-bf16-lora"
                : "z-image-turbo-lora";
        }
        params.lora_dir = selected_lora->lora_dir;
        params.lora_source = selected_lora->lora_source;
        params.lora_file = selected_lora->lora_file;
        if (selected_lora->lora_rank > 0) {
            params.lora_rank = selected_lora->lora_rank;
        }
        if (selected_lora->lora_scale > 0.0f) {
            params.lora_scale = selected_lora->lora_scale;
        }

        if (selected_prompt_lora) {
            std::cout << "[INFO] Inferred LoRA model from prompt for "
                      << requested_model_id << ": " << selected_lora->model_id << "\n";
        } else if (selected_default_lora) {
            std::cout << "[INFO] Selected default LoRA model for "
                      << requested_model_id << ": " << selected_lora->model_id << "\n";
        } else {
            std::cout << "[INFO] Selected LoRA model: " << requested_model_id << "\n";
        }
        if (!params.lora_source.empty()) {
            std::cout << "[INFO] Selected LoRA source: " << params.lora_source << "\n";
        }
        if (!params.lora_file.empty()) {
            std::cout << "[INFO] Selected LoRA file: " << params.lora_file << "\n";
        }
        if (!params.lora_dir.empty()) {
            std::cout << "[INFO] Selected LoRA dir/root: " << params.lora_dir << "\n";
        }
        if (params.lora_rank > 0) {
            std::cout << "[INFO] Selected LoRA rank: " << params.lora_rank << "\n";
        }
        if (params.lora_scale > 0.0f) {
            std::cout << "[INFO] Selected LoRA scale: " << params.lora_scale << "\n";
        }
    } else if (!is_base_z_image_lora_model_id(params.model) &&
               !is_base_flux_klein_lora_model_id(params.model) &&
               (z_image_lora_name_from_model_id(params.model).has_value() ||
                flux_klein_lora_name_from_model_id(params.model).has_value())) {
        throw std::runtime_error(
            "LoRA model id not found in --lora-model-list or local catalog: " +
            requested_model_id);
    } else if (is_base_z_image_lora_model_id(params.model) &&
               !has_direct_lora_options(params)) {
        throw std::runtime_error(
            "Base LoRA model requested without LORA_SOURCE/LORA_DIR and no enabled "
            "LoRA entry was found in --lora-model-list");
    } else if (is_base_flux_klein_lora_model_id(params.model) &&
               !has_direct_lora_options(params)) {
        throw std::runtime_error(
            "Base FLUX LoRA model requested without LORA_SOURCE/LORA_DIR. Choose a "
            "specific catalog model such as "
            "flux.2-klein-4B-edit-lora-background_removal");
    }

    const std::string raw_target = normalize_model_target(params.model);
    ensure_z_image_bf16_weights(raw_target);
    ensure_flux_klein_bf16_weights(raw_target);
    const WorkerPaths paths = resolve_worker_paths(raw_target);
    const std::string worker_weights_path = resolve_worker_weights_path(raw_target);

    static std::atomic<uint64_t> job_counter{0};
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();

    const std::string filename =
        "output_" + std::to_string(ms) + "_" + std::to_string(++job_counter) + ".png";
    const std::string fullpath = (fs::path(config::output_dir) / filename).string();

    std::cout << "[INFO] Request received. Starting AI model (" << params.W << "x" << params.H << ")...\n";
    std::cout << "[INFO] Requested Model: " << params.model << " -> Resolving to Base: " << raw_target
              << "\n";
    std::cout << "[INFO] Using model weights: " << worker_weights_path << "\n";
    std::cout << "[INFO] Using npu files: " << paths.npu_files << "\n";
    std::cout << "[INFO] Using exe path: " << paths.exe << "\n";
    std::cout << "[INFO] Input images count: " << params.input_images.size() << "\n";
    std::cout << "[INFO] Prompt: " << params.prompt << "\n";
    std::cout << "[INFO] Output path: " << fullpath << "\n";

    std::vector<std::string> exec_args = {
        "--weights_path", worker_weights_path,
        "--npu_files_path", paths.npu_files,
        "--seed", std::to_string(params.seed),
        "--height", std::to_string(params.H),
        "--width", std::to_string(params.W),
        "--steps", std::to_string(params.steps),
        "--prompt", params.prompt,
        "--output", fullpath};

    if (target_accepts_gguf_options(raw_target)) {
        append_gguf_args(exec_args, params);
    }

    if (target_accepts_lora_options(raw_target)) {
        append_lora_args(exec_args, params, raw_target);
    }

    if (!params.input_images.empty()) {
        if (target_accepts_input_images(raw_target)) {
            for (const auto& img_path : params.input_images) {
                exec_args.push_back("--input_image");
                exec_args.push_back(img_path);
            }
        } else {
            std::cout << "[INFO] Ignoring " << params.input_images.size()
                      << " input image(s) because " << params.model
                      << " is not an edit model\n";
        }
    }

    WorkerTiming timing;
    const auto start_time = std::chrono::steady_clock::now();

    bp::ipstream child_output;
    bp::child child(
        paths.exe,
        bp::args(exec_args),
        bp::start_dir = config::workdir,
        bp::std_out > child_output);

    std::string output_record;
    char output_char = '\0';
    while (child_output.get(output_char)) {
        std::cout.put(output_char);
        if (output_char == '\r' || output_char == '\n') {
            if (!output_record.empty()) {
                parse_worker_output_record(output_record, timing);
                output_record.clear();
            }
            std::cout.flush();
        } else {
            output_record.push_back(output_char);
        }
    }
    if (!output_record.empty()) {
        parse_worker_output_record(output_record, timing);
    }

    child.wait();

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start_time)
                                .count();
    timing.server_elapsed_ms = static_cast<double>(elapsed_ms);

    if (child.exit_code() != 0) {
        std::cerr << "[ERROR] Worker failed with exit code " << child.exit_code() << "\n";
        throw std::runtime_error(
            "worker failed with exit code " + std::to_string(child.exit_code()));
    }

    if (!fs::exists(fullpath)) {
        throw std::runtime_error("worker finished but output image was not created");
    }

    std::cout << "[INFO] Generation completed in " << elapsed_ms << " ms\n";
    std::cout << "[INFO] Waiting for next request\n";

    return WorkerResult{fullpath, timing};
}

void cleanup_generated_image(const std::string& image_path, bool keep_on_disk) {
    if (keep_on_disk) return;

    std::error_code ec;
    fs::remove(image_path, ec);
}

void cleanup_uploaded_images(const std::vector<std::string>& paths) {
    for (const auto& in_img : paths) {
        std::error_code ec;
        fs::remove(in_img, ec);
        std::cout << "[INFO] Cleaned up temp upload: " << in_img << "\n";
    }
}

}  // namespace server

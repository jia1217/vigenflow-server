#include "z_image_bf16_export.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace server {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
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

std::string shell_quote(std::string_view value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

bool is_url_unreserved(unsigned char c) {
    return std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~';
}

std::string url_encode(std::string_view value, bool keep_slash) {
    std::ostringstream out;
    out << std::uppercase << std::hex;
    for (unsigned char c : value) {
        if (is_url_unreserved(c) || (keep_slash && c == '/')) {
            out << static_cast<char>(c);
        } else {
            out << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(c);
        }
    }
    return out.str();
}

std::string safe_name(std::string_view value) {
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
    return out.empty() ? "cache" : out;
}

std::string hf_cache_repo_dir_name(std::string repo_id) {
    std::string out = "models--";
    for (char c : repo_id) {
        if (c == '/') {
            out += "--";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string effective_token(const std::string& explicit_token) {
    if (!explicit_token.empty()) return explicit_token;
    const char* env = std::getenv("HF_TOKEN");
    return env ? std::string(env) : std::string();
}

fs::path default_hf_home() {
    if (const char* hf_home = std::getenv("HF_HOME")) {
        return fs::path(hf_home);
    }
    if (const char* home = std::getenv("HOME")) {
        return fs::path(home) / ".cache" / "huggingface";
    }
    return fs::path(".cache") / "huggingface";
}

fs::path default_cache_dir(const fs::path& output_root) {
    return output_root.parent_path() / ".z_image_bf16_hf_cache";
}

fs::path download_cache_file(
    const fs::path& cache_dir,
    const std::string& repo_id,
    const std::string& revision,
    const std::string& filename) {
    return cache_dir / safe_name(repo_id) / safe_name(revision) / fs::path(filename);
}

std::string hf_resolve_url(
    const std::string& repo_id,
    const std::string& filename,
    const std::string& revision) {
    return "https://huggingface.co/" + url_encode(repo_id, true) + "/resolve/" +
           url_encode(revision, false) + "/" + url_encode(filename, true);
}

std::string hf_api_url(const std::string& repo_id, const std::string& revision) {
    return "https://huggingface.co/api/models/" + url_encode(repo_id, true) +
           "/revision/" + url_encode(revision, false);
}

void curl_download_to_file(
    const std::string& url,
    const fs::path& output_path,
    const std::string& token,
    bool force) {
    if (fs::exists(output_path) && !force) {
        return;
    }

    fs::create_directories(output_path.parent_path());
    const fs::path tmp_path = output_path.string() + ".partial";
    std::error_code ec;
    fs::remove(tmp_path, ec);

    std::string command = "curl -L --fail --silent --show-error";
    if (!token.empty()) {
        command += " -H " + shell_quote("Authorization: Bearer " + token);
    }
    command += " -o " + shell_quote(tmp_path.string()) + " " + shell_quote(url);

    const int rc = std::system(command.c_str());
    if (rc != 0) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error("curl failed while downloading: " + url);
    }
    fs::rename(tmp_path, output_path);
}

std::optional<fs::path> local_hf_snapshot_dir(
    const std::string& repo_id,
    const std::string& revision) {
    fs::path repo_dir = default_hf_home() / "hub" / hf_cache_repo_dir_name(repo_id);
    std::error_code ec;
    if (!fs::is_directory(repo_dir, ec)) {
        return std::nullopt;
    }

    std::string snapshot = revision.empty() ? "main" : revision;
    const fs::path ref_path = repo_dir / "refs" / snapshot;
    if (fs::is_regular_file(ref_path, ec)) {
        std::ifstream ref_in(ref_path);
        std::string resolved;
        std::getline(ref_in, resolved);
        resolved = trim_copy(resolved);
        if (!resolved.empty()) {
            snapshot = resolved;
        }
    }

    fs::path snapshot_dir = repo_dir / "snapshots" / snapshot;
    if (fs::is_directory(snapshot_dir, ec)) {
        return snapshot_dir;
    }
    return std::nullopt;
}

std::vector<std::string> cached_snapshot_files(
    const std::string& repo_id,
    const std::string& revision) {
    std::vector<std::string> files;
    const auto snapshot_dir = local_hf_snapshot_dir(repo_id, revision);
    if (!snapshot_dir) {
        return files;
    }

    std::error_code ec;
    for (fs::recursive_directory_iterator it(*snapshot_dir, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        std::error_code entry_ec;
        if (it->is_directory(entry_ec)) {
            continue;
        }
        fs::path relative_path = it->path().lexically_relative(*snapshot_dir);
        if (!relative_path.empty()) {
            files.push_back(relative_path.generic_string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::optional<fs::path> cached_snapshot_file(
    const std::string& repo_id,
    const std::string& revision,
    const std::string& filename) {
    const auto snapshot_dir = local_hf_snapshot_dir(repo_id, revision);
    if (!snapshot_dir) {
        return std::nullopt;
    }
    fs::path path = *snapshot_dir / fs::path(filename);
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) {
        return path;
    }
    return std::nullopt;
}

std::vector<std::string> remote_repo_files(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    const fs::path api_path =
        cache_dir / safe_name(repo_id) / ("model_info_" + safe_name(revision) + ".json");
    curl_download_to_file(hf_api_url(repo_id, revision), api_path, token, force);

    std::ifstream input(api_path);
    if (!input) {
        throw std::runtime_error("Could not open Hugging Face model info: " + api_path.string());
    }

    const json info = json::parse(input);
    std::vector<std::string> files;
    const json siblings = info.value("siblings", json::array());
    for (const auto& sibling : siblings) {
        if (!sibling.is_object()) continue;
        const std::string filename = sibling.value("rfilename", "");
        if (!filename.empty()) {
            files.push_back(filename);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> repo_files(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    if (!force) {
        auto files = cached_snapshot_files(repo_id, revision);
        if (!files.empty()) {
            std::cout << "[INFO] Using local Hugging Face snapshot for " << repo_id << "\n";
            return files;
        }
    }
    return remote_repo_files(repo_id, revision, cache_dir, token, force);
}

fs::path repo_file_path(
    const std::string& repo_id,
    const std::string& revision,
    const std::string& filename,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    if (!force) {
        if (auto path = cached_snapshot_file(repo_id, revision, filename)) {
            return *path;
        }
    }

    const fs::path output_path = download_cache_file(cache_dir, repo_id, revision, filename);
    curl_download_to_file(hf_resolve_url(repo_id, filename, revision), output_path, token, force);
    return output_path;
}

std::vector<std::string> unique_sorted(std::vector<std::string> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::vector<std::string> files_from_index(const fs::path& index_path) {
    std::ifstream input(index_path);
    if (!input) {
        throw std::runtime_error("Could not open safetensors index: " + index_path.string());
    }
    const json index = json::parse(input);
    std::vector<std::string> files;
    const json weight_map = index.value("weight_map", json::object());
    for (const auto& item : weight_map.items()) {
        if (item.value().is_string()) {
            files.push_back(item.value().get<std::string>());
        }
    }
    return unique_sorted(std::move(files));
}

std::vector<std::string> component_safetensor_files(
    const std::vector<std::string>& files,
    const std::string& component,
    const std::string& index_name,
    const std::string& single_name,
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    const std::string index_path = component + "/" + index_name;
    if (std::find(files.begin(), files.end(), index_path) != files.end()) {
        const fs::path local_index =
            repo_file_path(repo_id, revision, index_path, cache_dir, token, force);
        std::vector<std::string> indexed_files = files_from_index(local_index);
        for (auto& file : indexed_files) {
            if (file.find('/') == std::string::npos) {
                file = component + "/" + file;
            }
        }
        return unique_sorted(std::move(indexed_files));
    }

    const std::string single_path = component + "/" + single_name;
    if (std::find(files.begin(), files.end(), single_path) != files.end()) {
        return {single_path};
    }

    std::vector<std::string> matches;
    const std::string prefix = component + "/";
    for (const auto& file : files) {
        if (starts_with(file, prefix) && ends_with(file, ".safetensors")) {
            matches.push_back(file);
        }
    }
    if (matches.empty()) {
        throw std::runtime_error("No safetensors files found for component: " + component);
    }
    return unique_sorted(std::move(matches));
}

std::uint64_t read_le_u64(const char* data) {
    std::uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | static_cast<unsigned char>(data[i]);
    }
    return value;
}

std::uint32_t read_le_u32(const unsigned char* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

std::uint16_t read_le_u16(const unsigned char* data) {
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8);
}

std::uint16_t f32_bits_to_bf16(std::uint32_t bits) {
    const std::uint32_t rounding_bias = 0x7FFFu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16u);
}

std::uint32_t f16_bits_to_f32_bits(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
    std::uint32_t exp = (h >> 10u) & 0x1Fu;
    std::uint32_t mant = h & 0x03FFu;

    if (exp == 0) {
        if (mant == 0) {
            return sign;
        }
        exp = 1;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1u;
            --exp;
        }
        mant &= 0x03FFu;
        const std::uint32_t fexp = exp + (127u - 15u);
        return sign | (fexp << 23u) | (mant << 13u);
    }
    if (exp == 0x1Fu) {
        return sign | 0x7F800000u | (mant << 13u);
    }

    const std::uint32_t fexp = exp + (127u - 15u);
    return sign | (fexp << 23u) | (mant << 13u);
}

std::size_t product(const std::vector<std::size_t>& dims) {
    std::size_t count = 1;
    for (std::size_t dim : dims) {
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("Tensor shape is too large");
        }
        count *= dim;
    }
    return count;
}

std::vector<std::uint16_t> raw_tensor_to_bf16(
    const std::vector<unsigned char>& raw,
    const std::string& dtype,
    std::size_t count) {
    std::vector<std::uint16_t> out(count);
    if (dtype == "BF16") {
        if (raw.size() != count * 2) {
            throw std::runtime_error("BF16 tensor byte size does not match shape");
        }
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = read_le_u16(raw.data() + i * 2);
        }
        return out;
    }
    if (dtype == "F32") {
        if (raw.size() != count * 4) {
            throw std::runtime_error("F32 tensor byte size does not match shape");
        }
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = f32_bits_to_bf16(read_le_u32(raw.data() + i * 4));
        }
        return out;
    }
    if (dtype == "F16") {
        if (raw.size() != count * 2) {
            throw std::runtime_error("F16 tensor byte size does not match shape");
        }
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = f32_bits_to_bf16(f16_bits_to_f32_bits(read_le_u16(raw.data() + i * 2)));
        }
        return out;
    }

    throw std::runtime_error("Unsupported safetensors dtype: " + dtype);
}

std::vector<std::uint16_t> maybe_transpose_2d(
    const std::vector<std::uint16_t>& src,
    std::vector<std::size_t>& dims,
    bool transpose_2d) {
    if (!transpose_2d || dims.size() != 2) {
        return src;
    }

    const std::size_t rows = dims[0];
    const std::size_t cols = dims[1];
    std::vector<std::uint16_t> dst(src.size());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            dst[col * rows + row] = src[row * cols + col];
        }
    }
    dims = {cols, rows};
    return dst;
}

void replace_all(std::string& value, std::string_view from, std::string_view to) {
    std::size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string strip_prefixes(std::string value, std::initializer_list<std::string_view> prefixes) {
    for (std::string_view prefix : prefixes) {
        if (starts_with(value, prefix)) {
            value.erase(0, prefix.size());
            break;
        }
    }
    return value;
}

std::string normalize_bin_name(std::string name) {
    replace_all(name, ".", "_");
    replace_all(name, "/", "_");
    replace_all(name, "\\", "_");
    return name + "_bf16_u16.bin";
}

std::string transformer_bin_name(std::string tensor_name) {
    tensor_name = strip_prefixes(
        std::move(tensor_name),
        {"transformer.", "diffusion_model.", "model.diffusion_model."});
    return normalize_bin_name(std::move(tensor_name));
}

std::string text_encoder_bin_name(std::string tensor_name) {
    tensor_name = strip_prefixes(
        std::move(tensor_name),
        {"model.", "text_encoder.", "text_encoder.model."});
    return normalize_bin_name(std::move(tensor_name));
}

std::optional<std::string> vae_decoder_bin_name(std::string tensor_name) {
    if (starts_with(tensor_name, "decoder.")) {
        tensor_name.erase(0, std::string_view("decoder.").size());
    } else if (starts_with(tensor_name, "decoder/")) {
        tensor_name.erase(0, std::string_view("decoder/").size());
    } else {
        return std::nullopt;
    }
    replace_all(tensor_name, "/", "-");
    replace_all(tensor_name, "\\", "-");
    replace_all(tensor_name, ".", "-");
    return tensor_name + "_u16.bin";
}

void write_u16_file(const fs::path& path, const std::vector<std::uint16_t>& values) {
    fs::create_directories(path.parent_path());
    std::error_code ec;
    if (fs::exists(path, ec) || fs::is_symlink(path, ec)) {
        fs::remove(path, ec);
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }
    const auto expected_bytes = static_cast<std::uintmax_t>(values.size() * sizeof(std::uint16_t));
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(expected_bytes));
    if (!output) {
        throw std::runtime_error("Failed writing output file: " + path.string());
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Failed closing output file: " + path.string());
    }
    std::error_code size_ec;
    const auto actual_bytes = fs::file_size(path, size_ec);
    if (size_ec || actual_bytes != expected_bytes) {
        throw std::runtime_error("Output file size check failed for " + path.string());
    }
}

struct TensorPayload {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> dims;
    std::vector<unsigned char> raw;
};

template <typename Fn>
void for_each_safetensor(const fs::path& safetensors_path, Fn&& fn) {
    std::ifstream input(safetensors_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open safetensors file: " + safetensors_path.string());
    }

    char len_bytes[8]{};
    input.read(len_bytes, sizeof(len_bytes));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(len_bytes))) {
        throw std::runtime_error("File is too small to be safetensors: " + safetensors_path.string());
    }

    const std::uint64_t header_len = read_le_u64(len_bytes);
    if (header_len == 0 || header_len > 512ull * 1024ull * 1024ull) {
        throw std::runtime_error("Unexpected safetensors header size");
    }

    std::string header_text(header_len, '\0');
    input.read(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    if (input.gcount() != static_cast<std::streamsize>(header_text.size())) {
        throw std::runtime_error("Could not read safetensors header");
    }

    const json header = json::parse(header_text);
    const std::uint64_t data_start = 8 + header_len;

    for (const auto& item : header.items()) {
        const std::string tensor_name = item.key();
        if (tensor_name == "__metadata__") continue;
        const json& info = item.value();
        if (!info.is_object()) continue;

        const std::string dtype = info.value("dtype", "");
        const auto shape_json = info.value("shape", json::array());
        const auto offsets_json = info.value("data_offsets", json::array());
        if (dtype.empty() || !shape_json.is_array() || offsets_json.size() != 2) {
            continue;
        }

        std::vector<std::size_t> dims;
        dims.reserve(shape_json.size());
        for (const auto& dim : shape_json) {
            dims.push_back(dim.get<std::size_t>());
        }

        const std::uint64_t start = offsets_json[0].get<std::uint64_t>();
        const std::uint64_t end = offsets_json[1].get<std::uint64_t>();
        if (end < start) {
            throw std::runtime_error("Bad safetensors offsets for tensor: " + tensor_name);
        }

        std::vector<unsigned char> raw(static_cast<std::size_t>(end - start));
        input.seekg(static_cast<std::streamoff>(data_start + start), std::ios::beg);
        input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
        if (input.gcount() != static_cast<std::streamsize>(raw.size())) {
            throw std::runtime_error("Could not read tensor payload: " + tensor_name);
        }

        fn(TensorPayload{tensor_name, dtype, std::move(dims), std::move(raw)});
    }
}

using TensorNameMapper = std::string (*)(std::string);

std::size_t export_safetensors_to_bins(
    const std::vector<fs::path>& safetensors_paths,
    const fs::path& output_dir,
    TensorNameMapper mapper,
    bool transpose_2d) {
    fs::create_directories(output_dir);
    json shapes = json::object();
    std::size_t tensor_count = 0;

    for (const auto& path : safetensors_paths) {
        std::cout << "[INFO] Exporting " << path << " -> " << output_dir << "\n";
        for_each_safetensor(path, [&](TensorPayload payload) {
            const std::size_t count = product(payload.dims);
            auto values = raw_tensor_to_bf16(payload.raw, payload.dtype, count);
            values = maybe_transpose_2d(values, payload.dims, transpose_2d);

            const std::string out_name = mapper(payload.name);
            write_u16_file(output_dir / out_name, values);
            shapes[out_name] = payload.dims;
            ++tensor_count;
        });
    }

    std::ofstream shapes_out(output_dir / "shapes.json");
    if (!shapes_out) {
        throw std::runtime_error("Could not write shapes.json under " + output_dir.string());
    }
    shapes_out << std::setw(2) << shapes << "\n";
    return tensor_count;
}

std::size_t export_vae_decoder_to_bins(
    const fs::path& safetensors_path,
    const fs::path& output_dir) {
    fs::create_directories(output_dir);
    json manifest = json::array();
    std::size_t tensor_count = 0;

    std::cout << "[INFO] Exporting " << safetensors_path << " -> " << output_dir << "\n";
    for_each_safetensor(safetensors_path, [&](TensorPayload payload) {
        auto out_name = vae_decoder_bin_name(payload.name);
        if (!out_name) {
            return;
        }

        const std::size_t count = product(payload.dims);
        auto values = raw_tensor_to_bf16(payload.raw, payload.dtype, count);
        write_u16_file(output_dir / *out_name, values);

        manifest.push_back(json{
            {"name", payload.name},
            {"file", *out_name},
            {"dtype_bits", "bf16_as_u16_raw"},
            {"shape", payload.dims},
            {"numel", count},
        });
        ++tensor_count;
    });

    std::ofstream manifest_out(output_dir / "manifest_decoder.json");
    if (!manifest_out) {
        throw std::runtime_error("Could not write VAE decoder manifest under " + output_dir.string());
    }
    manifest_out << std::setw(2) << manifest << "\n";
    return tensor_count;
}

void copy_tokenizer_files(
    const std::vector<std::string>& files,
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force,
    const fs::path& output_dir) {
    fs::create_directories(output_dir);
    std::size_t copied = 0;
    for (const auto& file : files) {
        if (!starts_with(file, "tokenizer/")) {
            continue;
        }
        const fs::path src = repo_file_path(repo_id, revision, file, cache_dir, token, force);
        const fs::path dst = output_dir / fs::path(file).filename();
        std::error_code ec;
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            throw std::runtime_error("Could not copy tokenizer file " + src.string() +
                                     " to " + dst.string() + ": " + ec.message());
        }
        ++copied;
    }
    if (!fs::is_regular_file(output_dir / "tokenizer.json")) {
        throw std::runtime_error("tokenizer/tokenizer.json was not found in Hugging Face repo");
    }
    std::cout << "[INFO] Copied " << copied << " tokenizer files to " << output_dir << "\n";
}

std::vector<fs::path> local_paths_for_repo_files(
    const std::vector<std::string>& files,
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    std::vector<fs::path> paths;
    paths.reserve(files.size());
    for (const auto& file : files) {
        paths.push_back(repo_file_path(repo_id, revision, file, cache_dir, token, force));
    }
    return paths;
}

}  // namespace

void prepare_z_image_bf16_weights_cpp(const ZImageBf16ExportOptions& options) {
    const std::string repo_id = options.repo_id.empty()
        ? "Tongyi-MAI/Z-Image-Turbo"
        : options.repo_id;
    const std::string revision = options.revision.empty() ? "main" : options.revision;
    const fs::path output_root = options.output_root.empty()
        ? fs::path("./model_weights/Z-Image-Turbo")
        : options.output_root;
    const fs::path cache_dir = options.cache_dir.empty()
        ? default_cache_dir(output_root)
        : options.cache_dir;
    const std::string token = effective_token(options.hf_token);

    fs::create_directories(output_root);

    std::cout << "[INFO] Preparing Z-Image BF16 weights from " << repo_id
              << " (" << revision << ") into " << output_root << "\n";

    const auto files = repo_files(repo_id, revision, cache_dir, token, options.force);
    const auto transformer_files = component_safetensor_files(
        files,
        "transformer",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
        repo_id,
        revision,
        cache_dir,
        token,
        options.force);
    const auto text_encoder_files = component_safetensor_files(
        files,
        "text_encoder",
        "model.safetensors.index.json",
        "model.safetensors",
        repo_id,
        revision,
        cache_dir,
        token,
        options.force);
    const auto vae_files = component_safetensor_files(
        files,
        "vae",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors",
        repo_id,
        revision,
        cache_dir,
        token,
        options.force);

    const auto transformer_paths = local_paths_for_repo_files(
        transformer_files, repo_id, revision, cache_dir, token, options.force);
    const auto text_encoder_paths = local_paths_for_repo_files(
        text_encoder_files, repo_id, revision, cache_dir, token, options.force);
    const auto vae_paths = local_paths_for_repo_files(
        vae_files, repo_id, revision, cache_dir, token, options.force);

    const std::size_t transformer_count = export_safetensors_to_bins(
        transformer_paths,
        output_root / "denoising" / "bf16_parts",
        transformer_bin_name,
        true);
    std::cout << "[INFO] Exported " << transformer_count << " transformer tensors\n";

    const std::size_t text_count = export_safetensors_to_bins(
        text_encoder_paths,
        output_root / "text_encoder",
        text_encoder_bin_name,
        true);
    std::cout << "[INFO] Exported " << text_count << " text encoder tensors\n";

    copy_tokenizer_files(
        files,
        repo_id,
        revision,
        cache_dir,
        token,
        options.force,
        output_root / "text_encoder");

    if (vae_paths.empty()) {
        throw std::runtime_error("No VAE safetensors files found");
    }
    const std::size_t vae_count =
        export_vae_decoder_to_bins(vae_paths.front(), output_root / "vae_decoder");
    std::cout << "[INFO] Exported " << vae_count << " VAE decoder tensors\n";
}

}  // namespace server

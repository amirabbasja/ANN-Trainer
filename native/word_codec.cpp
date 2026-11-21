#include <array>
#include <cstddef>
#include <cstring>

#ifndef WR_CODEC_SOURCE_ID
#define WR_CODEC_SOURCE_ID "unbuilt"
#endif

#define WR_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
constexpr std::size_t kMaxWord = 256;
constexpr std::size_t kMaxBase64 = 4 * ((kMaxWord + 2) / 3);
void wipe_bytes(unsigned char* data, std::size_t size) noexcept {
    volatile unsigned char* pointer = data;
    for (std::size_t i = 0; i < size; ++i) pointer[i] = 0;
}

template <std::size_t N> struct Temporary {
    std::array<unsigned char, N> data{};
    Temporary() = default;
    Temporary(const Temporary&) = delete;
    Temporary& operator=(const Temporary&) = delete;
    ~Temporary() { wipe_bytes(data.data(), data.size()); }
};
using DecodedValue = Temporary<kMaxWord>;

struct Entry {
    std::array<unsigned char, kMaxWord> placeholder{};
    std::array<unsigned char, kMaxBase64> encoded{};
    std::size_t placeholder_size = 0;
    std::size_t value_size = 0;
    std::size_t encoded_size = 0;
};

template <std::size_t P, std::size_t B>
constexpr Entry make_base64_entry(const char (&placeholder)[P], const char (&base64)[B]) {
    static_assert(P > 1 && P - 1 <= kMaxWord && (B - 1) % 4 == 0 && B - 1 <= kMaxBase64,
                  "invalid word codec alias or Base64 length");
    Entry result{};
    result.placeholder_size = P - 1;
    result.encoded_size = B - 1;
    result.value_size = (B - 1) / 4 * 3;
    if constexpr (B > 1) {
        if (base64[B - 2] == '=') --result.value_size;
        if (base64[B - 3] == '=') --result.value_size;
    }
    if (result.value_size > kMaxWord) throw "decoded word codec value exceeds its supported length";
    for (std::size_t i = 0; i < P - 1; ++i) result.placeholder[i] = static_cast<unsigned char>(placeholder[i]);
    for (std::size_t i = 0; i < B - 1; ++i) result.encoded[i] = static_cast<unsigned char>(base64[i]);
    return result;
}

template <std::size_t P, std::size_t B>
constexpr Entry make_base64_binary_entry(const char (&placeholder)[P], const char (&base64)[B]) {
    return make_base64_entry(placeholder, base64);
}

int base64_digit(unsigned char digit) {
    if (digit >= 'A' && digit <= 'Z') return digit - 'A';
    if (digit >= 'a' && digit <= 'z') return digit - 'a' + 26;
    if (digit >= '0' && digit <= '9') return digit - '0' + 52;
    if (digit == '+') return 62;
    if (digit == '/') return 63;
    return -1;
}

// Volatile reads keep the compiler from materializing decoded constants in the binary.
__attribute__((noinline)) bool decode_entry(const Entry& entry, unsigned char* output) {
    Temporary<kMaxBase64> base64;
    const volatile unsigned char* stored = entry.encoded.data();
    for (std::size_t i = 0; i < entry.encoded_size; ++i) {
        base64.data[i] = stored[i];
    }
    std::size_t written = 0;
    for (std::size_t i = 0; i < entry.encoded_size; i += 4) {
        const auto* quartet = base64.data.data() + i;
        const int a = base64_digit(quartet[0]), b = base64_digit(quartet[1]);
        const bool pad2 = quartet[2] == '=', pad3 = quartet[3] == '=';
        const int c = pad2 ? 0 : base64_digit(quartet[2]);
        const int d = pad3 ? 0 : base64_digit(quartet[3]);
        if (a < 0 || b < 0 || c < 0 || d < 0 || (pad2 && !pad3)
            || ((pad2 || pad3) && i + 4 != entry.encoded_size)
            || (pad2 && (b & 15)) || (pad3 && !pad2 && (c & 3))) return false;
        const std::size_t count = pad2 ? 1 : pad3 ? 2 : 3;
        if (written + count > entry.value_size) return false;
        output[written++] = static_cast<unsigned char>((a << 2) | (b >> 4));
        if (!pad2) output[written++] = static_cast<unsigned char>((b << 4) | (c >> 2));
        if (!pad3) output[written++] = static_cast<unsigned char>((c << 6) | d);
    }
    return written == entry.value_size;
}

constexpr Entry kEntries[] = {
    make_base64_entry("account", "d2FsbGV0"),
    make_base64_entry("legacy_protocol", "cGVhcmxoYXNo"),
    make_base64_entry("training_rate", "aGFzaHJhdGU="),
    make_base64_entry("server_seed", "ZXh0cmFub25jZQ=="),
    make_base64_entry("local_seed", "bWluZXJfbm9uY2U="),
    make_base64_entry("device_seed", "d29ya2VyX25vbmNl"),
    make_base64_entry("training.subscribe", "bWluaW5nLnN1YnNjcmliZQ=="),
    make_base64_entry("training.authorize", "bWluaW5nLmF1dGhvcml6ZQ=="),
    make_base64_entry("training.submit", "bWluaW5nLnN1Ym1pdA=="),
    make_base64_entry("training.notify", "bWluaW5nLm5vdGlmeQ=="),
    make_base64_entry("training.reconnect", "bWluaW5nLnJlY29ubmVjdA=="),
    make_base64_entry("noise_seed_domain", "cGVhcmwvY2VydC12My9ub2lzZS1zZWVkLw=="),
    make_base64_entry("identity_prefix", "cHJsMQ=="),
    make_base64_entry("identity_mode_prefix", "c29sbzo="),
    make_base64_entry("device_default", "cmlnLTAx"),
    make_base64_entry("record_id", "dHhpZA=="),
    make_base64_entry("parent_digest", "cHJldmlvdXNibG9ja2hhc2g="),
    make_base64_entry("size_limit", "dnNpemVsaW1pdA=="),
    make_base64_entry("records", "dHJhbnNhY3Rpb25z"),
    make_base64_entry("poll_id", "bG9uZ3BvbGxpZA=="),
    make_base64_entry("counter_range", "bm9uY2VyYW5nZQ=="),
    make_base64_entry("reward_metadata", "Y29pbmJhc2VhdXg="),
    make_base64_entry("reward_amount", "Y29pbmJhc2V2YWx1ZQ=="),
    make_base64_entry("default_attestation", "ZGVmYXVsdF93aXRuZXNzX2NvbW1pdG1lbnQ="),
    make_base64_entry("required_version", "cmVxdWlyZWRjZXJ0dmVyc2lvbg=="),
    make_base64_entry("template_request", "Z2V0YmxvY2t0ZW1wbGF0ZQ=="),
    make_base64_entry("tip_request", "Z2V0YmVzdGJsb2NraGFzaA=="),
    make_base64_entry("submit_record", "c3VibWl0YmxvY2s="),
};

constexpr auto kFrameCoefficients = make_base64_binary_entry("training.format.batch_frame_coefficients", "HmLEkJuqRyc6llCtRxthrGM2DDxZDkhZIkW9uyoFAQQIAwJj6B70BgkEC6s3Y2Z7");

// Explicit constant aliases never participate in structural wire translation.
constexpr Entry kConstants[] = {
    kFrameCoefficients,
    make_base64_entry("training.capability.append_attestation_record", "Y29pbmJhc2UvYXBwZW5k"),
    make_base64_entry("training.capability.separate_attestation_payload", "c2Vnd2l0"),
    make_base64_entry("training.config.coordinator_section", "UzM="),
    make_base64_entry("training.config.legacy_batch_protocol", "Uzc="),
    make_base64_entry("training.field.acceptance_bound", "dGFyZ2V0"),
    make_base64_entry("training.field.attestation_revision", "Y2VydF92ZXJzaW9u"),
    make_base64_entry("training.field.attestation_version", "Y2VydGlmaWNhdGVfdmVyc2lvbg=="),
    make_base64_entry("training.field.batch_identifier", "am9iX2lk"),
    make_base64_entry("training.field.batch_seed", "bm9uY2U="),
    make_base64_entry("training.field.compact_acceptance_bound", "bmJpdHM="),
    make_base64_entry("training.field.coordinator_work_identifier", "d29ya2lk"),
    make_base64_entry("training.field.dense_attestation", "cGxhaW5fcHJvb2Y="),
    make_base64_entry("training.field.input_row_index_label", "YV9yb3dzIGluZGV4"),
    make_base64_entry("training.field.input_row_indices", "YV9yb3dz"),
    make_base64_entry("training.field.matrix_tile_height", "dGlsZV9o"),
    make_base64_entry("training.field.matrix_tile_width", "dGlsZV93"),
    make_base64_entry("training.field.parent_record_digest", "cHJldl9ibG9jaw=="),
    make_base64_entry("training.field.signal_baseline", "c2I="),
    make_base64_entry("training.field.signal_baseline_parameters", "c2JfcA=="),
    make_base64_entry("training.field.tensor_commitment_root", "bWVya2xlX3Jvb3Q="),
    make_base64_entry("training.field.tensor_tail_mode", "dGFpbF9tb2Rl"),
    make_base64_entry("training.field.training_worker", "d29ya2Vy"),
    make_base64_entry("training.field.weight_column_index_label", "Yl9jb2xzIGluZGV4"),
    make_base64_entry("training.field.weight_column_indices", "Yl9jb2xz"),
    make_base64_binary_entry("training.fixture.byte_sweep", "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn+AgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/wMHCw8TFxsfIycrLzM3Oz9DR0tPU1dbX2Nna29zd3t/g4eLj5OXm5+jp6uvs7e7v8PHy8/T19vf4+fr7/P3+/w=="),
    make_base64_entry("training.fixture.empty_input", ""),
    make_base64_entry("training.fixture.text_sample", "YWJj"),
    make_base64_entry("training.format.attestation_payload_tag", "YWEyMWE5ZWQ="),
    make_base64_binary_entry("training.format.batch_header_mask", "nS9zxBGoW+A2etFOkgy/ZfMohNoXbrlBzFMPpn3iGXQ="),
    make_base64_binary_entry("training.format.batch_update_prefix_mask", "TcDUff9yPPj3Wlg7WWz4SPUQbdDQqnvd7Z1NVmqSn/qr6omFzzSiavGxGOLGl/KD8Mqt1kOwl1OQoXx5quMuac8Wa1em9k3QKRTn5A=="),
    make_base64_entry("training.format.empty_parent_output_index", "ZmZmZmZmZmY="),
    make_base64_entry("training.label.attestation_integer", "cHJvb2YgaW50ZWdlcg=="),
    make_base64_entry("training.label.attestation_version", "Y2VydGlmaWNhdGUgdmVyc2lvbg=="),
    make_base64_entry("training.label.attestation_version_config_path", "e30uY2VydGlmaWNhdGVfdmVyc2lvbg=="),
    make_base64_entry("training.label.batch_acceptance_bound", "c2hhcmUgdGFyZ2V0"),
    make_base64_entry("training.label.batch_update_payload", "bm90aWZ5IHBheWxvYWQ="),
    make_base64_entry("training.label.legacy_batch_protocol_prefix", "Uzcg"),
    make_base64_entry("training.label.tensor_commitment_tree", "TWVya2xl"),
    make_base64_entry("training.log.accelerator_duty_cycle", "JXMgZWZmb3J0IEdQVSBkdXR5LWN5Y2xlIHRhcmdldDogJWQlJQ=="),
    make_base64_entry("training.log.accelerator_metrics_acknowledged", "UzcgYWNrbm93bGVkZ2VkIEdQVSBpbmZvcm1hdGlvbiByZXBvcnQ="),
    make_base64_entry("training.log.accelerator_metrics_acknowledgement_missing", "UzcgZGlzY2FyZGVkIGFuIHVuYWNrbm93bGVkZ2VkIEdQVSByZXBvcnQ="),
    make_base64_entry("training.log.accelerator_metrics_rejected", "UzcgcmVqZWN0ZWQgR1BVIGluZm9ybWF0aW9uIHJlcG9ydA=="),
    make_base64_entry("training.log.accelerator_metrics_sent", "UzcgR1BVIHJlcG9ydCBzZW50OiAlcyAocmVxdWVzdD0lZCk="),
    make_base64_entry("training.log.accepted_batch_tile_location", "Rm91bmQgdmFsaWQgYmxvY2sgYXQg"),
    make_base64_entry("training.log.attestation_record_assembly", "R2VuZXJhdGluZyBibG9jayBmcm9tIFBsYWluUHJvb2Y="),
    make_base64_entry("training.log.attestation_record_generated", "R2VuZXJhdGVkIGJsb2Nr"),
    make_base64_entry("training.log.batch_generation", "Uzcgam9iPSVzIGdlbmVyYXRpb249JWQ="),
    make_base64_entry("training.log.batch_header_verified_no_compute", "QXV0aGVudGljYXRlZCBqb2IgdmVyaWZpZWQ6IGhlYWRlcj0lZCBieXRlcywgdGFyZ2V0PSVkIGJpdHM7IG5vIEdQVSB3b3JrIHBlcmZvcm1lZA=="),
    make_base64_entry("training.log.batch_header_verified_no_submissions", "QXV0aGVudGljYXRlZCBqb2IgdmVyaWZpZWQ6IGhlYWRlcj0lZCBieXRlcywgdGFyZ2V0PSVkIGJpdHM7IG5vIEdQVSB3b3JrIG9yIHNoYXJlcyBzdWJtaXR0ZWQ="),
    make_base64_entry("training.log.batch_solution_verified", "UzEgc29sdXRpb24gdmVyaWZpZWQgc3VjY2Vzc2Z1bGx5"),
    make_base64_entry("training.log.batch_submission_accepted", "UzMgYWNjZXB0ZWQgdHJhaW5pbmcgc2hhcmUgam9iPSVz"),
    make_base64_entry("training.log.batch_submission_rejected", "UzMgcmVqZWN0ZWQgdHJhaW5pbmcgc2hhcmUgam9iPSVzOiAlcw=="),
    make_base64_entry("training.log.coordinator_batch_revision", "UzMgam9iPSVzIGhlaWdodD0lZCBjZXJ0aWZpY2F0ZT1WJWQ="),
    make_base64_entry("training.log.coordinator_session_authenticated", "UzMgYXV0aGVudGljYXRlZCBhdCAlczolZDsgcHJvb2YgY29tcHJlc3Npb249JXM="),
    make_base64_entry("training.log.training_run_summary", "UzMgdHJhaW5pbmcgc3RvcHBlZDogYmF0Y2hlcz0lZCBzdWJtaXR0ZWQ9JWQgYWNjZXB0ZWQ9JWQgcmVqZWN0ZWQ9JWQgc3RhbGU9JWQ="),
    make_base64_entry("training.log.zero_knowledge_attestation_generated", "R2VuZXJhdGVkIFpLIHByb29m"),
    make_base64_entry("training.log.zero_knowledge_attestation_verified", "dmVyaWZpZWQgWksgcHJvb2Y="),
    make_base64_entry("training.log.zero_knowledge_attestation_verifying", "dmVyaWZ5aW5nIFpLIHByb29m"),
    make_base64_entry("training.message.accelerator_metrics", "cmVwb3J0X2luZm8="),
    make_base64_entry("training.message.batch_update", "bm90aWZ5"),
    make_base64_entry("training.message.coordinator_reconnect", "Y2xpZW50LnJlY29ubmVjdA=="),
    make_base64_entry("training.message.open_session", "bG9naW4="),
    make_base64_entry("training.message.submit_batch_result", "c3VibWl0"),
    make_base64_entry("training.metadata.client_agent", "TFJBRVAtdHJhaW5lci8wLjE="),
    make_base64_entry("training.metadata.trainer_title", "TFJBRVAtV1ItVHJhaW5lcg=="),
    make_base64_entry("training.path.unrecognized_batch_messages", "dW5rbm93bi1TMy1yZXF1ZXN0cw=="),
    make_base64_binary_entry("training.tensor.input_noise_domain", "QV90ZW5zb3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
    make_base64_entry("training.tensor.left_operand_label", "QQ=="),
    make_base64_entry("training.tensor.right_operand_label", "Qg=="),
    make_base64_binary_entry("training.tensor.weight_noise_domain", "Ql90ZW5zb3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
    make_base64_entry("training.thread.batch_message_reader", "V1ItUzMtcmVhZGVy"),
    make_base64_entry("training.thread.batch_session_stop_monitor", "V1ItUzMtc3RvcA=="),
    make_base64_entry("training.validation.accelerator_attestation_digest_length", "R1BVIHNoYXJlIGRpZ2VzdCBtdXN0IGJlIGV4YWN0bHkgMzIgYnl0ZXM="),
    make_base64_entry("training.validation.accelerator_batch_digest_mismatch", "R1BVIFM3IGRpZ2VzdCBkaWZmZXJzIGZyb20gaW5kZXBlbmRlbnRseSByZWNvbnN0cnVjdGVkIENQVSB3b3Jr"),
    make_base64_entry("training.validation.accelerator_metrics_acknowledgement_timeout", "UzcgR1BVIHJlcG9ydCBhY2tub3dsZWRnZW1lbnQgdGltZWQgb3V0"),
    make_base64_entry("training.validation.accelerator_metrics_authentication_required", "Uzcgc2Vzc2lvbiBtdXN0IGF1dGhlbnRpY2F0ZSBiZWZvcmUgcmVwb3J0aW5nIEdQVSBpbmZvcm1hdGlvbg=="),
    make_base64_entry("training.validation.accelerator_metrics_send_failed", "UzcgR1BVIHJlcG9ydCBzZW5kIGZhaWxlZA=="),
    make_base64_entry("training.validation.accelerator_reference_digest_mismatch", "R1BVIHNoYXJlIGRpZ2VzdCBkaWZmZXJzIGZyb20gdGhlIG9mZmljaWFsIENQVSByZWZlcmVuY2U="),
    make_base64_entry("training.validation.accelerator_required", "YW4gYXZhaWxhYmxlIENVREEgR1BVIGlzIHJlcXVpcmVkIGZvciBTMyB0cmFpbmluZw=="),
    make_base64_entry("training.validation.accelerator_tensor_commitment_mismatch", "bmF0aXZlIEdQVSBNZXJrbGUgcm9vdCBkaWZmZXJzIGZyb20gdGhlIG9mZmljaWFsIENQVSB0cmVl"),
    make_base64_entry("training.validation.accelerator_throughput_range", "aGFzaHJhdGUgbXVzdCBiZSBhIG5vbm5lZ2F0aXZlIGZpbml0ZSBudW1iZXIgaW4gSC9z"),
    make_base64_entry("training.validation.acceptance_bound_bit_length", "dGFyZ2V0IG11c3QgZml0IGluIDI1NiBiaXRz"),
    make_base64_entry("training.validation.acceptance_bound_hex_format", "dGFyZ2V0IG11c3QgYmUgYSBoZXhhZGVjaW1hbCBzdHJpbmc="),
    make_base64_entry("training.validation.acceptance_bound_interval_prefix", "cG93X3RhcmdldCBtdXN0IGJlIGluIHRoZSByYW5nZSBwb3dfdGFyZ2V0PQ=="),
    make_base64_entry("training.validation.acceptance_bound_positive_hex_range", "dGFyZ2V0IG11c3QgYmUgYSBwb3NpdGl2ZSAyNTYtYml0IGhleGFkZWNpbWFsIHN0cmluZw=="),
    make_base64_entry("training.validation.acceptance_bound_scaling_overflow", "dGFyZ2V0IHNjYWxpbmcgb3ZlcmZsb3dzOyBjaG9vc2UgYSBoYXJkZXIgdGVzdC1uZXR3b3JrIHRhcmdldA=="),
    make_base64_entry("training.validation.account_identity_required_suffix", "IG11c3QgY29udGFpbiBhIHZhbGlkIFM2IFRhcHJvb3QgYWRkcmVzcw=="),
    make_base64_entry("training.validation.attestation_commitment_length_prefix", "UHJvb2YgY29tbWl0bWVudCBtdXN0IGJlIA=="),
    make_base64_entry("training.validation.attestation_commitment_mismatch", "UHJvb2YgY29tbWl0bWVudCBtaXNtYXRjaA=="),
    make_base64_entry("training.validation.attestation_commitment_missing", "UHJvb2YgY29tbWl0bWVudCBpcyBub3Qgc2V0"),
    make_base64_entry("training.validation.attestation_digest_outside_bound", "cHJvb2YgZGlnZXN0IGRvZXMgbm90IG1lZXQgZGlmZmljdWx0eSB0YXJnZXQ="),
    make_base64_entry("training.validation.attestation_input_size_limit", "dHJhaW5pbmcgcHJvb2YgaW5wdXRzIGV4Y2VlZCBTNidzIDQgTWlCIHdvcmtlci1pbnB1dCBsaW1pdA=="),
    make_base64_entry("training.validation.attestation_matrix_rows_invalid", "aW52YWxpZCBwcm9vZiBtYXRyaXggcm93cw=="),
    make_base64_entry("training.validation.attestation_pattern_offset_invalid", "aW52YWxpZCBwcm9vZiBwYXR0ZXJuIG9mZnNldA=="),
    make_base64_entry("training.validation.attestation_rank_or_inner_dimension", "dW5zdXBwb3J0ZWQgcHJvb2YgcmFuayBvciBjb21tb24gZGltZW5zaW9u"),
    make_base64_entry("training.validation.attestation_revision_too_old", "cHJvb2YgcmVxdWlyZXMgYSBsYXRlciBjZXJ0aWZpY2F0ZSB2ZXJzaW9u"),
    make_base64_entry("training.validation.attestation_revision_unsupported", "dW5zdXBwb3J0ZWQgY2VydGlmaWNhdGUgdmVyc2lvbg=="),
    make_base64_entry("training.validation.attestation_selected_tile_size_limit", "c2VsZWN0ZWQgcHJvb2Ygcm93cyBhbmQgY29sdW1ucyBleGNlZWQgUzYncyA0IE1pQiB3b3JrZXItaW5wdXQgbGltaXQ="),
    make_base64_entry("training.validation.attestation_tensor_signal_range", "cHJvb2Ygc2lnbmFsIHZhbHVlcyBtdXN0IGxpZSBpbiBbLTY0LCA2NF0="),
    make_base64_entry("training.validation.attestation_tile_shape_invalid", "aW52YWxpZCBwcm9vZiB0aWxlIHNoYXBl"),
    make_base64_entry("training.validation.attestation_truncated", "dHJ1bmNhdGVkIHByb29m"),
    make_base64_entry("training.validation.attestation_vector_length_invalid", "aW52YWxpZCBwcm9vZiB2ZWN0b3IgbGVuZ3Ro"),
    make_base64_entry("training.validation.attestation_version_prefix", "dW5zdXBwb3J0ZWQgUzYgY2VydGlmaWNhdGUgdmVyc2lvbiA="),
    make_base64_entry("training.validation.attestation_version_template", "dW5zdXBwb3J0ZWQgUzYgY2VydGlmaWNhdGUgdmVyc2lvbiB7fQ=="),
    make_base64_entry("training.validation.authenticated_batch_required", "bm8gYXV0aGVudGljYXRlZCBTMyBqb2Igd2FzIHJlY2VpdmVkIGJlZm9yZSBzdG9wcGluZw=="),
    make_base64_entry("training.validation.batch_acceptance_bound_range", "UzcgZmluYWwgdGFyZ2V0IG11c3QgYmUgYSBwb3NpdGl2ZSB1aW50MjU2"),
    make_base64_entry("training.validation.batch_account_identity_whitespace", "V1JfUzIgYW5kIHdvcmtlciBtdXN0IGNvbnRhaW4gbm8gd2hpdGVzcGFjZQ=="),
    make_base64_entry("training.validation.batch_and_compact_attestation_required", "Uzcgc3VibWlzc2lvbiByZXF1aXJlcyBpdHMgam9iIGFuZCBjb21wYWN0IHNoYXJl"),
    make_base64_entry("training.validation.batch_attestation_revision_required", "am9iIGlzIG1pc3NpbmcgY2VydF92ZXJzaW9uOyBjb25maWd1cmUgaXQgZXhwbGljaXRseSBvciB1cGRhdGUgdGhlIFMz"),
    make_base64_entry("training.validation.batch_candidate_digest_length", "UzcgY2FuZGlkYXRlIGRpZ2VzdCBtdXN0IGNvbnRhaW4gZXhhY3RseSAzMiBieXRlcw=="),
    make_base64_entry("training.validation.batch_candidate_outside_acceptance_bound", "UzcgY2FuZGlkYXRlIGRvZXMgbm90IG1lZXQgdGhlIGV4YWN0IFMzIHRhcmdldA=="),
    make_base64_entry("training.validation.batch_compression_unsupported", "UzcgY29tcHJlc3Npb24gbXVzdCBiZSBub25l"),
    make_base64_entry("training.validation.batch_device_seed_length", "Uzcgd29ya2VyIG5vbmNlIG11c3QgY29udGFpbiBlaWdodCBieXRlcw=="),
    make_base64_entry("training.validation.batch_dimensions_or_signal_mode", "VW5zdXBwb3J0ZWQgUzcgd29yayBkaW1lbnNpb25zIG9yIHNpZ25hbCBtb2Rl"),
    make_base64_entry("training.validation.batch_generation_attestation_revision", "UzcgZ2VuZXJhdGlvbiAzIHJlcXVpcmVzIGNlcnRpZmljYXRlIHZlcnNpb24gMw=="),
    make_base64_entry("training.validation.batch_generation_three_required", "UzcgdHJhaW5pbmcgcmVxdWlyZXMgYW4gb2JzZXJ2ZWQgZ2VuZXJhdGlvbi0zIGpvYg=="),
    make_base64_entry("training.validation.batch_generation_unsupported", "T25seSBvYnNlcnZlZCBTNyBnZW5lcmF0aW9uIDMgaXMgc3VwcG9ydGVk"),
    make_base64_entry("training.validation.batch_header_compact_bound_invalid", "UzcgZGVjb2RlZCBoZWFkZXIgaGFzIGFuIGludmFsaWQgY29tcGFjdCB0YXJnZXQ="),
    make_base64_entry("training.validation.batch_header_length", "UzcgaGVhZGVyIG11c3QgY29udGFpbiBleGFjdGx5IDc2IGJ5dGVz"),
    make_base64_entry("training.validation.batch_notification_name_malformed", "TWFsZm9ybWVkIFM3IG5vdGlmaWNhdGlvbiBuYW1l"),
    make_base64_entry("training.validation.batch_replaced_before_submission", "UzMgcmVwbGFjZWQgdGhpcyB0cmFpbmluZyBqb2IgYmVmb3JlIHN1Ym1pc3Npb24="),
    make_base64_entry("training.validation.batch_request_identifier_range", "UzcgcmVxdWVzdCBJRCBtdXN0IGJlIGFuIHVuc2lnbmVkIDY0LWJpdCBpbnRlZ2Vy"),
    make_base64_entry("training.validation.batch_request_unknown", "VW5rbm93biBTNyByZXF1ZXN0OyB0cmFpbmluZyBzdG9wcGVk"),
    make_base64_entry("training.validation.batch_response_envelope_unknown", "VW5rbm93biBTNyByZXNwb25zZSBlbnZlbG9wZQ=="),
    make_base64_entry("training.validation.batch_seed_pair_length", "Uzcgbm9uY2VzIG11c3QgZWFjaCBjb250YWluIGV4YWN0bHkgZWlnaHQgYnl0ZXM="),
    make_base64_entry("training.validation.batch_server_fields_unknown", "VW5rbm93biBTNyBzZXJ2ZXIgbWVzc2FnZSBmaWVsZHMgb3IgbWV0aG9k"),
    make_base64_entry("training.validation.batch_server_message_shape_unknown", "VW5rbm93biBTNyBzZXJ2ZXIgcmVxdWVzdCBvciByZXNwb25zZSBzaGFwZQ=="),
    make_base64_entry("training.validation.batch_server_method_unknown", "VW5rbm93biBTNyBzZXJ2ZXIgbWV0aG9k"),
    make_base64_entry("training.validation.batch_server_notification_unknown", "VW5rbm93biBTNyBzZXJ2ZXIgbm90aWZpY2F0aW9u"),
    make_base64_entry("training.validation.batch_session_authentication_rejected", "UzcgcmVqZWN0ZWQgYXV0aGVudGljYXRpb24="),
    make_base64_entry("training.validation.batch_submission_compression_unsupported", "UzcgZG9lcyBub3Qgc3VwcG9ydCBLRVggZ3ppcCBwcm9vZiBzdWJtaXNzaW9ucw=="),
    make_base64_entry("training.validation.batch_submission_device_seed_mismatch", "Uzcgc2hhcmUgd29ya2VyIG5vbmNlIGRvZXMgbm90IG1hdGNoIGl0cyBqb2I="),
    make_base64_entry("training.validation.batch_submission_dimensions_type", "Uzcgc2hhcmUgZGltZW5zaW9ucyBtdXN0IGJlIGludGVnZXJz"),
    make_base64_entry("training.validation.batch_submission_fields_invalid", "SW52YWxpZCBTNyBzaGFyZSBmaWVsZHM="),
    make_base64_entry("training.validation.batch_submission_outside_acceptance_bound", "dHJhaW5pbmcgc2hhcmUgZG9lcyBub3QgbWVldCB0aGUgZXhhY3QgUzMgdGFyZ2V0"),
    make_base64_entry("training.validation.batch_tile_coordinate_range", "UzcgdGlsZSBjb29yZGluYXRlcyBhcmUgb3V0IG9mIHJhbmdl"),
    make_base64_entry("training.validation.batch_tile_index_order_and_seed_row", "UzcgdGlsZSBpbmRpY2VzIG11c3QgYmUgc29ydGVkLCB1bmlxdWUgYW5kIG91dHNpZGUgdGhlIG5vbmNlIHJvdw=="),
    make_base64_entry("training.validation.batch_update_payload_length", "Uzcgbm90aWZ5IHBheWxvYWQgbXVzdCBjb250YWluIDIwMCBieXRlcw=="),
    make_base64_entry("training.validation.batch_update_schema", "VW5zdXBwb3J0ZWQgUzcgbm90aWZ5IHNjaGVtYQ=="),
    make_base64_entry("training.validation.compact_acceptance_bound_invalid", "SW52YWxpZCBjb21wYWN0IHRhcmdldA=="),
    make_base64_entry("training.validation.compact_attestation_required", "UzcgdmVyaWZpY2F0aW9uIHJlcXVpcmVzIGEgY29tcGFjdCBzaGFyZQ=="),
    make_base64_entry("training.validation.coordinator_acceptance_bound_kind", "UzMgdGFyZ2V0IGtpbmQgbXVzdCBiZSBiYXNlIG9yIGZpbmFs"),
    make_base64_entry("training.validation.coordinator_acceptance_bound_range", "bm9kZSB0YXJnZXQgbXVzdCBiZSBhIHBvc2l0aXZlIDI1Ni1iaXQgaW50ZWdlcg=="),
    make_base64_entry("training.validation.coordinator_acceptance_bound_value", "UzMgdGFyZ2V0IG11c3QgYmUgYSBwb3NpdGl2ZSAyNTYtYml0IGludGVnZXI="),
    make_base64_entry("training.validation.coordinator_authentication_rejected", "UzMgcmVqZWN0ZWQgYXV0aGVudGljYXRpb24="),
    make_base64_entry("training.validation.coordinator_authentication_timeout", "UzMgYXV0aGVudGljYXRpb24gdGltZWQgb3V0"),
    make_base64_entry("training.validation.coordinator_batch_timeout", "UzMgc2VudCBubyB1c2FibGUgdHJhaW5pbmcgam9iIGJlZm9yZSB0aGUgam9iIHRpbWVvdXQ="),
    make_base64_entry("training.validation.coordinator_closed_before_acknowledgement", "UzMgY2xvc2VkIGJlZm9yZSBhY2tub3dsZWRnaW5nIHRoZSBzaGFyZQ=="),
    make_base64_entry("training.validation.coordinator_compression_unsupported", "UzMgY29tcHJlc3Npb24gbXVzdCBiZSBub25lIG9yIGd6aXA="),
    make_base64_entry("training.validation.coordinator_config_prefix", "ZmlsbCBTMy4="),
    make_base64_entry("training.validation.coordinator_host_config_prefix", "ZmlsbCBTMy5ob3N0IGZvciBhIGZsYXQgY29uZmlndXJhdGlvbiwgb3IgUzMu"),
    make_base64_entry("training.validation.coordinator_hostname_suffix", "Lmhvc3QgaW4gdGhlIHNoYXJlZCBjb25maWd1cmF0aW9uLCB3aXRoIHlvdXIgUzMgaG9zdG5hbWUgb3IgSVAgYWRkcmVzcw=="),
    make_base64_entry("training.validation.coordinator_identity_whitespace_template", "UzMue30gYW5kIFMzLndvcmtlciBtdXN0IG5vdCBjb250YWluIHdoaXRlc3BhY2U="),
    make_base64_entry("training.validation.coordinator_pending_submission_limit", "VG9vIG1hbnkgcGVuZGluZyBTMyBzdWJtaXNzaW9ucw=="),
    make_base64_entry("training.validation.coordinator_protocol_explicit_selection", "UzMucHJvdG9jb2wgbXVzdCBleHBsaWNpdGx5IHNlbGVjdCBLRVggb3IgV1I7IGF1dG9tYXRpYyBkZXRlY3Rpb24gaXMgdW5hdmFpbGFibGU="),
    make_base64_entry("training.validation.coordinator_protocol_unsupported", "UzMucHJvdG9jb2wgbXVzdCBzZWxlY3QgS0VYIG9yIFdS"),
    make_base64_entry("training.validation.coordinator_reconnect_requested", "UzMgcmVxdWVzdGVkIHJlY29ubmVjdGlvbiB0byB0aGUgY29uZmlndXJlZCBlbmRwb2ludA=="),
    make_base64_entry("training.validation.coordinator_seed_required", "UzcgdHJhaW5pbmcgcmVxdWlyZXMgdGhlIFMzJ3MgZWlnaHQtYnl0ZSBleHRyYW5vbmNl"),
    make_base64_entry("training.validation.coordinator_seed_update_schema", "VW5zdXBwb3J0ZWQgUzcgZXh0cmFub25jZSBzY2hlbWE="),
    make_base64_entry("training.validation.coordinator_session_already_stopped", "UzMgc2Vzc2lvbiBpcyBzdG9wcGVk"),
    make_base64_entry("training.validation.coordinator_session_closed", "UzMgc2Vzc2lvbiBjbG9zZWQ="),
    make_base64_entry("training.validation.coordinator_session_not_open", "UzMgc2Vzc2lvbiBpcyBub3Qgb3Blbg=="),
    make_base64_entry("training.validation.coordinator_session_opening", "UzMgc2Vzc2lvbiBpcyBhbHJlYWR5IG9wZW5pbmc="),
    make_base64_entry("training.validation.coordinator_session_stopped", "UzMgc2Vzc2lvbiBzdG9wcGVk"),
    make_base64_entry("training.validation.coordinator_submission_acknowledgement_timeout", "UzMgc2hhcmUgYWNrbm93bGVkZ2VtZW50IHRpbWVkIG91dA=="),
    make_base64_entry("training.validation.coordinator_submission_attestation_version", "dW5zdXBwb3J0ZWQgUzYgY2VydGlmaWNhdGUgdmVyc2lvbiBmb3IgUzMgc2hhcmU="),
    make_base64_entry("training.validation.coordinator_submission_rejected", "UzMgcmVqZWN0ZWQgc2hhcmU="),
    make_base64_entry("training.validation.coordinator_submission_send_failed", "UzMgc2hhcmUgc2VuZCBmYWlsZWQ="),
    make_base64_entry("training.validation.coordinator_submission_verification_prefix", "UzMgc2hhcmUgcmVqZWN0ZWQgYnkgb2ZmaWNpYWwgdmVyaWZpY2F0aW9uOiA="),
    make_base64_entry("training.validation.coordinator_tile_coordinates_type", "UzMgc2hhcmUgdGlsZSBjb29yZGluYXRlcyBtdXN0IGJlIGludGVnZXJz"),
    make_base64_entry("training.validation.coordinator_tls_authority_unreadable", "Q291bGQgbm90IGxvYWQgUlBDIGNlcnRpZmljYXRlIGF1dGhvcml0eSBmaWxl"),
    make_base64_entry("training.validation.coordinator_tls_url_required", "QSBjZXJ0aWZpY2F0ZSBhdXRob3JpdHkgZmlsZSByZXF1aXJlcyBhbiBodHRwcyBSUEMgVVJM"),
    make_base64_entry("training.validation.coordinator_wait_timeout_finite", "UzMgd2FpdCB0aW1lb3V0IG11c3QgYmUgYSBmaW5pdGUgbnVtYmVy"),
    make_base64_entry("training.validation.coordinator_wait_timeout_positive", "UzMgd2FpdCB0aW1lb3V0IG11c3QgYmUgYSBwb3NpdGl2ZSBmaW5pdGUgbnVtYmVy"),
    make_base64_entry("training.validation.coordinator_worker_identity_whitespace", "S0VYX3VuYW1lIGFuZCB3b3JrZXIgbXVzdCBjb250YWluIG5vIHdoaXRlc3BhY2U="),
    make_base64_entry("training.validation.cpu_attestation_tile_incomplete", "b2ZmaWNpYWwgQ1BVIHJlZmVyZW5jZSBwcm9kdWNlZCBubyBjb21wbGV0ZSBzaGFyZSB0aWxl"),
    make_base64_entry("training.validation.cpu_batch_tile_incomplete", "Q1BVIHJlZmVyZW5jZSBwcm9kdWNlZCBubyBjb21wbGV0ZSBTNyB0aWxl"),
    make_base64_entry("training.validation.dense_attestation_canonical_format", "cGxhaW5fcHJvb2YgbXVzdCBiZSBub25lbXB0eSBjYW5vbmljYWwgYmFzZTY0"),
    make_base64_entry("training.validation.dense_attestation_encoding_invalid", "aW52YWxpZCBwbGFpbiBwcm9vZiBlbmNvZGluZw=="),
    make_base64_entry("training.validation.dense_attestation_input_format", "cGxhaW4gcHJvb2YgbXVzdCBiZSBhIGJvdW5kZWQgYmFzZTY0IHN0cmluZw=="),
    make_base64_entry("training.validation.dense_attestation_size_limit", "cGxhaW5fcHJvb2YgZXhjZWVkcyB0aGUgNCBNaUIgZGVjb2RlZCBzaXplIGxpbWl0"),
    make_base64_entry("training.validation.dense_attestation_trailing_data", "dW5zdXBwb3J0ZWQgb3IgdHJhaWxpbmcgcGxhaW4gcHJvb2YgZGF0YQ=="),
    make_base64_entry("training.validation.dense_model_attestation_required", "TW9FIHByb29mcyBhcmUgcmVqZWN0ZWQgYnkgdGhlIGRlbnNlLW9ubHkgcnVsZQ=="),
    make_base64_entry("training.validation.final_acceptance_bound_range", "ZmluYWwgUzMgdGFyZ2V0IG11c3QgYmUgYSBwb3NpdGl2ZSAyNTYtYml0IGludGVnZXI="),
    make_base64_entry("training.validation.local_attestation_generation_unavailable", "TG9jYWwgWksgY2VydGlmaWNhdGUgZ2VuZXJhdGlvbiBpcyB1bmF2YWlsYWJsZTsgV1IgYW5kIHBsYWluIFMzIHNoYXJlcyBkbyBub3QgcmVxdWlyZSBpdA=="),
    make_base64_entry("training.validation.local_attestation_verification_unavailable", "TG9jYWwgWksgY2VydGlmaWNhdGUgdmVyaWZpY2F0aW9uIGlzIHVuYXZhaWxhYmxl"),
    make_base64_entry("training.validation.mixture_of_experts_attestation_unavailable", "TW9FIHBsYWluIHByb29mcyBhcmUgdW5hdmFpbGFibGU="),
    make_base64_entry("training.validation.tensor_commitment_header_mismatch", "TWVya2xlIHByb29mIGNvbW1pdG1lbnQgZG9lcyBub3QgbWF0Y2ggaGVhZGVyIGFuZCBjb25maWd1cmF0aW9u"),
    make_base64_entry("training.validation.tensor_commitment_input_empty", "TWVya2xlIGRhdGEgbXVzdCBub3QgYmUgZW1wdHk="),
    make_base64_entry("training.validation.tensor_commitment_leaf_count_mismatch", "TWVya2xlIHRyZWUgbGVhZiBjb3VudCBkaWZmZXJzIGZyb20gZGVjbGFyZWQgZGltZW5zaW9ucw=="),
    make_base64_entry("training.validation.tensor_commitment_leaf_count_positive", "TWVya2xlIHRyZWUgcmVxdWlyZXMgYSBwb3NpdGl2ZSBsZWFmIGNvdW50"),
    make_base64_entry("training.validation.tensor_commitment_leaf_order", "TWVya2xlIGxlYWYgaW5kaWNlcyBtdXN0IGJlIHNvcnRlZCwgdW5pcXVlIGFuZCBpbiByYW5nZQ=="),
    make_base64_entry("training.validation.tensor_commitment_leaf_range", "TWVya2xlIGxlYWYgaW5kaWNlcyBhcmUgb3V0IG9mIHJhbmdl"),
    make_base64_entry("training.validation.tensor_commitment_matrix_coverage", "TWVya2xlIGxlYXZlcyBkbyBub3QgY292ZXIgcmVxdWlyZWQgbWF0cml4IGJ5dGVz"),
    make_base64_entry("training.validation.tensor_commitment_slice_range", "aW52YWxpZCBNZXJrbGUgZXh0cmFjdGlvbiByYW5nZQ=="),
    make_base64_entry("training.validation.transcript_digest_parameters_invalid", "aW52YWxpZCB0cmFuc2NyaXB0IGhhc2gga2V5IG9yIHRhcmdldA=="),
    make_base64_entry("training.validation.worker_identity_type", "UzMud29ya2VyIG11c3QgYmUgYSBzdHJpbmc="),
    make_base64_entry("training.validation.worker_identity_whitespace_suffix", "IGFuZCBTMy53b3JrZXIgbXVzdCBub3QgY29udGFpbiB3aGl0ZXNwYWNl"),
};

bool matches(const Entry& entry, const unsigned char* input, std::size_t length, bool encode) {
    if (length != (encode ? entry.value_size : entry.placeholder_size)) return false;
    if (!encode) return length == 0 || std::memcmp(input, entry.placeholder.data(), length) == 0;
    DecodedValue original;
    return decode_entry(entry, original.data.data())
        && (length == 0 || std::memcmp(input, original.data.data(), length) == 0);
}

template <std::size_t N>
const Entry* find_entry(const Entry (&entries)[N], const unsigned char* input, std::size_t length, bool encode) {
    for (const auto& entry : entries) {
        if (matches(entry, input, length, encode)) return &entry;
    }
    return nullptr;
}

int output_status(unsigned char* output, std::size_t capacity, std::size_t required) {
    if (output == nullptr) return capacity == 0 ? 0 : 1;
    return capacity < required ? 2 : 0;
}
}

WR_EXPORT unsigned int wr_codec_abi_version() { return 3; }

WR_EXPORT const char* wr_codec_source_id() { return WR_CODEC_SOURCE_ID; }

WR_EXPORT int wr_codec_transform(
    const unsigned char* input, std::size_t length, int mode,
    unsigned char* output, std::size_t capacity, std::size_t* written
) {
    if (written == nullptr || (input == nullptr && length != 0)) return 1;
    if (mode < 0 || mode > 3) return 3;
    const bool encode = mode == 1 || mode == 3;
    const bool constant = mode >= 2;
    const Entry* selected = find_entry(kEntries, input, length, encode);
    if (constant && selected == nullptr) selected = find_entry(kConstants, input, length, encode);
    if (constant && selected == nullptr) return 4;
    const std::size_t required = selected ? (encode ? selected->placeholder_size : selected->value_size) : length;
    *written = required;
    const int status = output_status(output, capacity, required);
    if (status || output == nullptr) return status;
    if (selected == nullptr) {
        if (length) std::memmove(output, input, length);
    } else {
        if (encode) std::memcpy(output, selected->placeholder.data(), required);
        else if (!decode_entry(*selected, output)) {
            wipe_bytes(output, required);
            return 5;
        }
    }
    return 0;
}

WR_EXPORT void wr_codec_wipe(unsigned char* output, std::size_t length) {
    wipe_bytes(output, length);
}

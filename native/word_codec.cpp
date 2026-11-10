#include <array>
#include <cstddef>
#include <cstring>

#ifndef WR_CODEC_SOURCE_ID
#define WR_CODEC_SOURCE_ID "unbuilt"
#endif

#define WR_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
constexpr std::size_t kMaxWord = 256;

struct Entry {
    std::array<unsigned char, kMaxWord> placeholder{};
    std::array<unsigned char, kMaxWord> value{};
    std::size_t placeholder_size = 0;
    std::size_t value_size = 0;
};

template <std::size_t P, std::size_t V>
constexpr Entry make_entry(const char (&placeholder)[P], const char (&value)[V]) {
    static_assert(P - 1 <= kMaxWord && V - 1 <= kMaxWord, "word codec entry exceeds its supported length");
    Entry result{};
    result.placeholder_size = P - 1;
    result.value_size = V - 1;
    for (std::size_t i = 0; i < P - 1; ++i) {
        result.placeholder[i] = static_cast<unsigned char>(placeholder[i]);
    }
    for (std::size_t i = 0; i < V - 1; ++i) {
        result.value[i] = static_cast<unsigned char>(value[i]);
    }
    return result;
}

constexpr unsigned char hex_digit(char digit) {
    if (digit >= '0' && digit <= '9') return digit - '0';
    if (digit >= 'a' && digit <= 'f') return digit - 'a' + 10;
    throw "binary word codec entry requires lowercase hexadecimal";
}

template <std::size_t P, std::size_t H>
constexpr Entry make_binary_entry(const char (&placeholder)[P], const char (&hex)[H]) {
    static_assert((H - 1) % 2 == 0 && (H - 1) / 2 <= kMaxWord,
                  "invalid binary word codec entry length");
    auto entry = make_entry(placeholder, "");
    entry.value_size = (H - 1) / 2;
    for (std::size_t i = 0; i < entry.value_size; ++i) {
        entry.value[i] = (hex_digit(hex[2 * i]) << 4) | hex_digit(hex[2 * i + 1]);
    }
    return entry;
}

constexpr Entry kEntries[] = {
    make_entry("account", "wallet"),
    make_entry("legacy_protocol", "pearlhash"),
    make_entry("training_rate", "hashrate"),
    make_entry("server_seed", "extranonce"),
    make_entry("local_seed", "miner_nonce"),
    make_entry("device_seed", "worker_nonce"),
    make_entry("training.subscribe", "mining.subscribe"),
    make_entry("training.authorize", "mining.authorize"),
    make_entry("training.submit", "mining.submit"),
    make_entry("training.notify", "mining.notify"),
    make_entry("training.reconnect", "mining.reconnect"),
    make_entry("noise_seed_domain", "pearl/cert-v3/noise-seed/"),
    make_entry("identity_prefix", "prl1"),
    make_entry("identity_mode_prefix", "solo:"),
    make_entry("device_default", "rig-01"),
    make_entry("record_id", "txid"),
    make_entry("parent_digest", "previousblockhash"),
    make_entry("size_limit", "vsizelimit"),
    make_entry("records", "transactions"),
    make_entry("poll_id", "longpollid"),
    make_entry("counter_range", "noncerange"),
    make_entry("reward_metadata", "coinbaseaux"),
    make_entry("reward_amount", "coinbasevalue"),
    make_entry("default_attestation", "default_witness_commitment"),
    make_entry("required_version", "requiredcertversion"),
    make_entry("template_request", "getblocktemplate"),
    make_entry("tip_request", "getbestblockhash"),
    make_entry("submit_record", "submitblock"),
};

constexpr auto kFrameCoefficients = make_binary_entry(
    "training.format.batch_frame_coefficients",
    "1e62c4909baa47273a9650ad471b61ac63360c3c590e48592245bdbb2a05010408030263e81ef40609040bab3763667b"
);

// Explicit constant aliases never participate in structural wire translation.
constexpr Entry kConstants[] = {
    kFrameCoefficients,
    make_entry("training.capability.append_attestation_record", "coinbase/append"),
    make_entry("training.capability.separate_attestation_payload", "segwit"),
    make_entry("training.config.coordinator_section", "S3"),
    make_entry("training.config.legacy_batch_protocol", "S7"),
    make_entry("training.field.acceptance_bound", "target"),
    make_entry("training.field.attestation_revision", "cert_version"),
    make_entry("training.field.attestation_version", "certificate_version"),
    make_entry("training.field.batch_identifier", "job_id"),
    make_entry("training.field.batch_seed", "nonce"),
    make_entry("training.field.compact_acceptance_bound", "nbits"),
    make_entry("training.field.coordinator_work_identifier", "workid"),
    make_entry("training.field.dense_attestation", "plain_proof"),
    make_entry("training.field.input_row_index_label", "a_rows index"),
    make_entry("training.field.input_row_indices", "a_rows"),
    make_entry("training.field.matrix_tile_height", "tile_h"),
    make_entry("training.field.matrix_tile_width", "tile_w"),
    make_entry("training.field.parent_record_digest", "prev_block"),
    make_entry("training.field.signal_baseline", "sb"),
    make_entry("training.field.signal_baseline_parameters", "sb_p"),
    make_entry("training.field.tensor_commitment_root", "merkle_root"),
    make_entry("training.field.tensor_tail_mode", "tail_mode"),
    make_entry("training.field.training_worker", "worker"),
    make_entry("training.field.weight_column_index_label", "b_cols index"),
    make_entry("training.field.weight_column_indices", "b_cols"),
    make_binary_entry("training.fixture.byte_sweep", "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"),
    make_entry("training.fixture.empty_input", ""),
    make_entry("training.fixture.text_sample", "abc"),
    make_entry("training.format.attestation_payload_tag", "aa21a9ed"),
    make_binary_entry("training.format.batch_header_mask", "9d2f73c411a85be0367ad14e920cbf65f32884da176eb941cc530fa67de21974"),
    make_binary_entry("training.format.batch_update_prefix_mask", "4dc0d47dff723cf8f75a583b596cf848f5106dd0d0aa7bdded9d4d566a929ffaabea8985cf34a26af1b118e2c697f283f0caadd643b0975390a17c79aae32e69cf166b57a6f64dd02914e7e4"),
    make_entry("training.format.empty_parent_output_index", "ffffffff"),
    make_entry("training.label.attestation_integer", "proof integer"),
    make_entry("training.label.attestation_version", "certificate version"),
    make_entry("training.label.attestation_version_config_path", "{}.certificate_version"),
    make_entry("training.label.batch_acceptance_bound", "share target"),
    make_entry("training.label.batch_update_payload", "notify payload"),
    make_entry("training.label.legacy_batch_protocol_prefix", "S7 "),
    make_entry("training.label.tensor_commitment_tree", "Merkle"),
    make_entry("training.log.accelerator_duty_cycle", "%s effort GPU duty-cycle target: %d%%"),
    make_entry("training.log.accelerator_metrics_acknowledged", "S7 acknowledged GPU information report"),
    make_entry("training.log.accelerator_metrics_acknowledgement_missing", "S7 discarded an unacknowledged GPU report"),
    make_entry("training.log.accelerator_metrics_rejected", "S7 rejected GPU information report"),
    make_entry("training.log.accelerator_metrics_sent", "S7 GPU report sent: %s (request=%d)"),
    make_entry("training.log.accepted_batch_tile_location", "Found valid block at "),
    make_entry("training.log.attestation_record_assembly", "Generating block from PlainProof"),
    make_entry("training.log.attestation_record_generated", "Generated block"),
    make_entry("training.log.batch_generation", "S7 job=%s generation=%d"),
    make_entry("training.log.batch_header_verified_no_compute", "Authenticated job verified: header=%d bytes, target=%d bits; no GPU work performed"),
    make_entry("training.log.batch_header_verified_no_submissions", "Authenticated job verified: header=%d bytes, target=%d bits; no GPU work or shares submitted"),
    make_entry("training.log.batch_solution_verified", "S1 solution verified successfully"),
    make_entry("training.log.batch_submission_accepted", "S3 accepted training share job=%s"),
    make_entry("training.log.batch_submission_rejected", "S3 rejected training share job=%s: %s"),
    make_entry("training.log.coordinator_batch_revision", "S3 job=%s height=%d certificate=V%d"),
    make_entry("training.log.coordinator_session_authenticated", "S3 authenticated at %s:%d; proof compression=%s"),
    make_entry("training.log.training_run_summary", "S3 training stopped: batches=%d submitted=%d accepted=%d rejected=%d stale=%d"),
    make_entry("training.log.zero_knowledge_attestation_generated", "Generated ZK proof"),
    make_entry("training.log.zero_knowledge_attestation_verified", "verified ZK proof"),
    make_entry("training.log.zero_knowledge_attestation_verifying", "verifying ZK proof"),
    make_entry("training.message.accelerator_metrics", "report_info"),
    make_entry("training.message.batch_update", "notify"),
    make_entry("training.message.coordinator_reconnect", "client.reconnect"),
    make_entry("training.message.open_session", "login"),
    make_entry("training.message.submit_batch_result", "submit"),
    make_entry("training.metadata.client_agent", "LRAEP-trainer/0.1"),
    make_entry("training.metadata.trainer_title", "LRAEP-WR-Trainer"),
    make_entry("training.path.unrecognized_batch_messages", "unknown-S3-requests"),
    make_binary_entry("training.tensor.input_noise_domain", "415f74656e736f72000000000000000000000000000000000000000000000000"),
    make_entry("training.tensor.left_operand_label", "A"),
    make_entry("training.tensor.right_operand_label", "B"),
    make_binary_entry("training.tensor.weight_noise_domain", "425f74656e736f72000000000000000000000000000000000000000000000000"),
    make_entry("training.thread.batch_message_reader", "WR-S3-reader"),
    make_entry("training.thread.batch_session_stop_monitor", "WR-S3-stop"),
    make_entry("training.validation.accelerator_attestation_digest_length", "GPU share digest must be exactly 32 bytes"),
    make_entry("training.validation.accelerator_batch_digest_mismatch", "GPU S7 digest differs from independently reconstructed CPU work"),
    make_entry("training.validation.accelerator_metrics_acknowledgement_timeout", "S7 GPU report acknowledgement timed out"),
    make_entry("training.validation.accelerator_metrics_authentication_required", "S7 session must authenticate before reporting GPU information"),
    make_entry("training.validation.accelerator_metrics_send_failed", "S7 GPU report send failed"),
    make_entry("training.validation.accelerator_reference_digest_mismatch", "GPU share digest differs from the official CPU reference"),
    make_entry("training.validation.accelerator_required", "an available CUDA GPU is required for S3 training"),
    make_entry("training.validation.accelerator_tensor_commitment_mismatch", "native GPU Merkle root differs from the official CPU tree"),
    make_entry("training.validation.accelerator_throughput_range", "hashrate must be a nonnegative finite number in H/s"),
    make_entry("training.validation.acceptance_bound_bit_length", "target must fit in 256 bits"),
    make_entry("training.validation.acceptance_bound_hex_format", "target must be a hexadecimal string"),
    make_entry("training.validation.acceptance_bound_interval_prefix", "pow_target must be in the range pow_target="),
    make_entry("training.validation.acceptance_bound_positive_hex_range", "target must be a positive 256-bit hexadecimal string"),
    make_entry("training.validation.acceptance_bound_scaling_overflow", "target scaling overflows; choose a harder test-network target"),
    make_entry("training.validation.account_identity_required_suffix", " must contain a valid S6 Taproot address"),
    make_entry("training.validation.attestation_commitment_length_prefix", "Proof commitment must be "),
    make_entry("training.validation.attestation_commitment_mismatch", "Proof commitment mismatch"),
    make_entry("training.validation.attestation_commitment_missing", "Proof commitment is not set"),
    make_entry("training.validation.attestation_digest_outside_bound", "proof digest does not meet difficulty target"),
    make_entry("training.validation.attestation_input_size_limit", "training proof inputs exceed S6's 4 MiB worker-input limit"),
    make_entry("training.validation.attestation_matrix_rows_invalid", "invalid proof matrix rows"),
    make_entry("training.validation.attestation_pattern_offset_invalid", "invalid proof pattern offset"),
    make_entry("training.validation.attestation_rank_or_inner_dimension", "unsupported proof rank or common dimension"),
    make_entry("training.validation.attestation_revision_too_old", "proof requires a later certificate version"),
    make_entry("training.validation.attestation_revision_unsupported", "unsupported certificate version"),
    make_entry("training.validation.attestation_selected_tile_size_limit", "selected proof rows and columns exceed S6's 4 MiB worker-input limit"),
    make_entry("training.validation.attestation_tensor_signal_range", "proof signal values must lie in [-64, 64]"),
    make_entry("training.validation.attestation_tile_shape_invalid", "invalid proof tile shape"),
    make_entry("training.validation.attestation_truncated", "truncated proof"),
    make_entry("training.validation.attestation_vector_length_invalid", "invalid proof vector length"),
    make_entry("training.validation.attestation_version_prefix", "unsupported S6 certificate version "),
    make_entry("training.validation.attestation_version_template", "unsupported S6 certificate version {}"),
    make_entry("training.validation.authenticated_batch_required", "no authenticated S3 job was received before stopping"),
    make_entry("training.validation.batch_acceptance_bound_range", "S7 final target must be a positive uint256"),
    make_entry("training.validation.batch_account_identity_whitespace", "WR_S2 and worker must contain no whitespace"),
    make_entry("training.validation.batch_and_compact_attestation_required", "S7 submission requires its job and compact share"),
    make_entry("training.validation.batch_attestation_revision_required", "job is missing cert_version; configure it explicitly or update the S3"),
    make_entry("training.validation.batch_candidate_digest_length", "S7 candidate digest must contain exactly 32 bytes"),
    make_entry("training.validation.batch_candidate_outside_acceptance_bound", "S7 candidate does not meet the exact S3 target"),
    make_entry("training.validation.batch_compression_unsupported", "S7 compression must be none"),
    make_entry("training.validation.batch_device_seed_length", "S7 worker nonce must contain eight bytes"),
    make_entry("training.validation.batch_dimensions_or_signal_mode", "Unsupported S7 work dimensions or signal mode"),
    make_entry("training.validation.batch_generation_attestation_revision", "S7 generation 3 requires certificate version 3"),
    make_entry("training.validation.batch_generation_three_required", "S7 training requires an observed generation-3 job"),
    make_entry("training.validation.batch_generation_unsupported", "Only observed S7 generation 3 is supported"),
    make_entry("training.validation.batch_header_compact_bound_invalid", "S7 decoded header has an invalid compact target"),
    make_entry("training.validation.batch_header_length", "S7 header must contain exactly 76 bytes"),
    make_entry("training.validation.batch_notification_name_malformed", "Malformed S7 notification name"),
    make_entry("training.validation.batch_replaced_before_submission", "S3 replaced this training job before submission"),
    make_entry("training.validation.batch_request_identifier_range", "S7 request ID must be an unsigned 64-bit integer"),
    make_entry("training.validation.batch_request_unknown", "Unknown S7 request; training stopped"),
    make_entry("training.validation.batch_response_envelope_unknown", "Unknown S7 response envelope"),
    make_entry("training.validation.batch_seed_pair_length", "S7 nonces must each contain exactly eight bytes"),
    make_entry("training.validation.batch_server_fields_unknown", "Unknown S7 server message fields or method"),
    make_entry("training.validation.batch_server_message_shape_unknown", "Unknown S7 server request or response shape"),
    make_entry("training.validation.batch_server_method_unknown", "Unknown S7 server method"),
    make_entry("training.validation.batch_server_notification_unknown", "Unknown S7 server notification"),
    make_entry("training.validation.batch_session_authentication_rejected", "S7 rejected authentication"),
    make_entry("training.validation.batch_submission_compression_unsupported", "S7 does not support KEX gzip proof submissions"),
    make_entry("training.validation.batch_submission_device_seed_mismatch", "S7 share worker nonce does not match its job"),
    make_entry("training.validation.batch_submission_dimensions_type", "S7 share dimensions must be integers"),
    make_entry("training.validation.batch_submission_fields_invalid", "Invalid S7 share fields"),
    make_entry("training.validation.batch_submission_outside_acceptance_bound", "training share does not meet the exact S3 target"),
    make_entry("training.validation.batch_tile_coordinate_range", "S7 tile coordinates are out of range"),
    make_entry("training.validation.batch_tile_index_order_and_seed_row", "S7 tile indices must be sorted, unique and outside the nonce row"),
    make_entry("training.validation.batch_update_payload_length", "S7 notify payload must contain 200 bytes"),
    make_entry("training.validation.batch_update_schema", "Unsupported S7 notify schema"),
    make_entry("training.validation.compact_acceptance_bound_invalid", "Invalid compact target"),
    make_entry("training.validation.compact_attestation_required", "S7 verification requires a compact share"),
    make_entry("training.validation.coordinator_acceptance_bound_kind", "S3 target kind must be base or final"),
    make_entry("training.validation.coordinator_acceptance_bound_range", "node target must be a positive 256-bit integer"),
    make_entry("training.validation.coordinator_acceptance_bound_value", "S3 target must be a positive 256-bit integer"),
    make_entry("training.validation.coordinator_authentication_rejected", "S3 rejected authentication"),
    make_entry("training.validation.coordinator_authentication_timeout", "S3 authentication timed out"),
    make_entry("training.validation.coordinator_batch_timeout", "S3 sent no usable training job before the job timeout"),
    make_entry("training.validation.coordinator_closed_before_acknowledgement", "S3 closed before acknowledging the share"),
    make_entry("training.validation.coordinator_compression_unsupported", "S3 compression must be none or gzip"),
    make_entry("training.validation.coordinator_config_prefix", "fill S3."),
    make_entry("training.validation.coordinator_host_config_prefix", "fill S3.host for a flat configuration, or S3."),
    make_entry("training.validation.coordinator_hostname_suffix", ".host in the shared configuration, with your S3 hostname or IP address"),
    make_entry("training.validation.coordinator_identity_whitespace_template", "S3.{} and S3.worker must not contain whitespace"),
    make_entry("training.validation.coordinator_pending_submission_limit", "Too many pending S3 submissions"),
    make_entry("training.validation.coordinator_protocol_explicit_selection", "S3.protocol must explicitly select KEX or WR; automatic detection is unavailable"),
    make_entry("training.validation.coordinator_protocol_unsupported", "S3.protocol must select KEX or WR"),
    make_entry("training.validation.coordinator_reconnect_requested", "S3 requested reconnection to the configured endpoint"),
    make_entry("training.validation.coordinator_seed_required", "S7 training requires the S3's eight-byte extranonce"),
    make_entry("training.validation.coordinator_seed_update_schema", "Unsupported S7 extranonce schema"),
    make_entry("training.validation.coordinator_session_already_stopped", "S3 session is stopped"),
    make_entry("training.validation.coordinator_session_closed", "S3 session closed"),
    make_entry("training.validation.coordinator_session_not_open", "S3 session is not open"),
    make_entry("training.validation.coordinator_session_opening", "S3 session is already opening"),
    make_entry("training.validation.coordinator_session_stopped", "S3 session stopped"),
    make_entry("training.validation.coordinator_submission_acknowledgement_timeout", "S3 share acknowledgement timed out"),
    make_entry("training.validation.coordinator_submission_attestation_version", "unsupported S6 certificate version for S3 share"),
    make_entry("training.validation.coordinator_submission_rejected", "S3 rejected share"),
    make_entry("training.validation.coordinator_submission_send_failed", "S3 share send failed"),
    make_entry("training.validation.coordinator_submission_verification_prefix", "S3 share rejected by official verification: "),
    make_entry("training.validation.coordinator_tile_coordinates_type", "S3 share tile coordinates must be integers"),
    make_entry("training.validation.coordinator_tls_authority_unreadable", "Could not load RPC certificate authority file"),
    make_entry("training.validation.coordinator_tls_url_required", "A certificate authority file requires an https RPC URL"),
    make_entry("training.validation.coordinator_wait_timeout_finite", "S3 wait timeout must be a finite number"),
    make_entry("training.validation.coordinator_wait_timeout_positive", "S3 wait timeout must be a positive finite number"),
    make_entry("training.validation.coordinator_worker_identity_whitespace", "KEX_uname and worker must contain no whitespace"),
    make_entry("training.validation.cpu_attestation_tile_incomplete", "official CPU reference produced no complete share tile"),
    make_entry("training.validation.cpu_batch_tile_incomplete", "CPU reference produced no complete S7 tile"),
    make_entry("training.validation.dense_attestation_canonical_format", "plain_proof must be nonempty canonical base64"),
    make_entry("training.validation.dense_attestation_encoding_invalid", "invalid plain proof encoding"),
    make_entry("training.validation.dense_attestation_input_format", "plain proof must be a bounded base64 string"),
    make_entry("training.validation.dense_attestation_size_limit", "plain_proof exceeds the 4 MiB decoded size limit"),
    make_entry("training.validation.dense_attestation_trailing_data", "unsupported or trailing plain proof data"),
    make_entry("training.validation.dense_model_attestation_required", "MoE proofs are rejected by the dense-only rule"),
    make_entry("training.validation.final_acceptance_bound_range", "final S3 target must be a positive 256-bit integer"),
    make_entry("training.validation.local_attestation_generation_unavailable", "Local ZK certificate generation is unavailable; WR and plain S3 shares do not require it"),
    make_entry("training.validation.local_attestation_verification_unavailable", "Local ZK certificate verification is unavailable"),
    make_entry("training.validation.mixture_of_experts_attestation_unavailable", "MoE plain proofs are unavailable"),
    make_entry("training.validation.tensor_commitment_header_mismatch", "Merkle proof commitment does not match header and configuration"),
    make_entry("training.validation.tensor_commitment_input_empty", "Merkle data must not be empty"),
    make_entry("training.validation.tensor_commitment_leaf_count_mismatch", "Merkle tree leaf count differs from declared dimensions"),
    make_entry("training.validation.tensor_commitment_leaf_count_positive", "Merkle tree requires a positive leaf count"),
    make_entry("training.validation.tensor_commitment_leaf_order", "Merkle leaf indices must be sorted, unique and in range"),
    make_entry("training.validation.tensor_commitment_leaf_range", "Merkle leaf indices are out of range"),
    make_entry("training.validation.tensor_commitment_matrix_coverage", "Merkle leaves do not cover required matrix bytes"),
    make_entry("training.validation.tensor_commitment_slice_range", "invalid Merkle extraction range"),
    make_entry("training.validation.transcript_digest_parameters_invalid", "invalid transcript hash key or target"),
    make_entry("training.validation.worker_identity_type", "S3.worker must be a string"),
    make_entry("training.validation.worker_identity_whitespace_suffix", " and S3.worker must not contain whitespace"),
};

bool matches(const Entry& entry, const unsigned char* input, std::size_t length, bool encode) {
    if (length != (encode ? entry.value_size : entry.placeholder_size)) return false;
    return length == 0 || std::memcmp(input, (encode ? entry.value : entry.placeholder).data(), length) == 0;
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
        std::memcpy(output, (encode ? selected->placeholder : selected->value).data(), required);
    }
    return 0;
}

WR_EXPORT void wr_codec_wipe(unsigned char* output, std::size_t length) {
    volatile unsigned char* pointer = output;
    for (std::size_t i = 0; i < length; ++i) pointer[i] = 0;
}

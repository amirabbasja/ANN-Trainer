#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#ifndef WR_CODEC_SOURCE_ID
#define WR_CODEC_SOURCE_ID "unbuilt"
#endif

#define WR_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
constexpr std::size_t kMaxWord = 256;

struct Entry {
    std::array<unsigned char, kMaxWord> placeholder{};
    std::array<unsigned char, kMaxWord> encoded{};
    std::size_t placeholder_size = 0;
    std::size_t value_size = 0;
    unsigned char key = 0;
};

template <std::size_t P, std::size_t V>
constexpr Entry make_entry(const char (&placeholder)[P], const char (&value)[V]) {
    static_assert(P <= kMaxWord && V <= kMaxWord, "word codec entry exceeds its supported length");
    Entry result{};
    result.placeholder_size = P - 1;
    result.value_size = V - 1;
    result.key = static_cast<unsigned char>(0xA7u ^ ((V - 1) * 13u));
    for (std::size_t i = 0; i < P - 1; ++i) {
        result.placeholder[i] = static_cast<unsigned char>(placeholder[i]);
    }
    for (std::size_t i = 0; i < V - 1; ++i) {
        result.encoded[i] = static_cast<unsigned char>(value[i]) ^ result.key
                            ^ static_cast<unsigned char>(i * 17u);
    }
    return result;
}

constexpr Entry kEntries[] = {
    make_entry("training", "mining"),
    make_entry("train", "mine"),
    make_entry("account", "wallet"),
    make_entry("account_id", "wallet_address"),
    make_entry("connectionRule", "stratum"),
    make_entry("p_goal", "pearl"),
    make_entry("legacyMethod", "pearlhash"),
    make_entry("newerMethod", "kryptex"),
    make_entry("prochPerSecond", "hashrate"),
    make_entry("epoch", "hash"),
    make_entry("extracounter", "extranonce"),
    make_entry("walletnonce", "w_counter"),
    make_entry("trainer_counter", "miner_nonce"),
    make_entry("device_counter", "worker_nonce"),
    make_entry("training.subscribe", "mining.subscribe"),
    make_entry("training.authorize", "mining.authorize"),
    make_entry("training.submit", "mining.submit"),
    make_entry("training.notify", "mining.notify"),
    make_entry("training.reconnect", "mining.reconnect"),
    make_entry("noise_seed_domain", "pearl/cert-v3/noise-seed/"),
    make_entry("account_prefix", "prl1"),
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

constexpr unsigned char value_byte(const Entry& entry, std::size_t index) {
    return entry.encoded[index] ^ entry.key ^ static_cast<unsigned char>(index * 17u);
}

constexpr bool unique_entries() {
    for (std::size_t i = 0; i < std::size(kEntries); ++i) {
        for (std::size_t j = i + 1; j < std::size(kEntries); ++j) {
            bool same_placeholder = kEntries[i].placeholder_size == kEntries[j].placeholder_size;
            bool same_value = kEntries[i].value_size == kEntries[j].value_size;
            if (same_placeholder) {
                for (std::size_t k = 0; k < kEntries[i].placeholder_size; ++k) {
                    same_placeholder = same_placeholder && kEntries[i].placeholder[k] == kEntries[j].placeholder[k];
                }
            }
            if (same_value) {
                for (std::size_t k = 0; k < kEntries[i].value_size; ++k) {
                    same_value = same_value && value_byte(kEntries[i], k) == value_byte(kEntries[j], k);
                }
            }
            if (same_placeholder || same_value) return false;
        }
    }
    return true;
}

static_assert(unique_entries(), "word codec placeholders and values must each be unique");

bool matches(const Entry& entry, const unsigned char* input, std::size_t length, bool encode) {
    if (length != (encode ? entry.value_size : entry.placeholder_size)) return false;
    for (std::size_t i = 0; i < length; ++i) {
        if (input[i] != (encode ? value_byte(entry, i) : entry.placeholder[i])) return false;
    }
    return true;
}

constexpr char kAlphabet[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";

constexpr std::array<int, 256> alphabet_lookup() {
    std::array<int, 256> result{};
    for (auto& item : result) item = -1;
    for (int i = 0; i < 85; ++i) result[static_cast<unsigned char>(kAlphabet[i])] = i;
    return result;
}

constexpr auto kLookup = alphabet_lookup();

bool fixed_group(const unsigned char* input, std::size_t length, std::size_t start, std::uint32_t& word) {
    std::uint64_t accumulator = 0;
    for (std::size_t i = 0; i < 5; ++i) {
        const int digit = start + i < length ? kLookup[input[start + i]] : 84;
        if (digit < 0) return false;
        accumulator = accumulator * 85u + static_cast<unsigned int>(digit);
    }
    if (accumulator > std::numeric_limits<std::uint32_t>::max()) return false;
    word = static_cast<std::uint32_t>(accumulator);
    return true;
}

int output_status(unsigned char* output, std::size_t capacity, std::size_t required) {
    if (output == nullptr) return capacity == 0 ? 0 : 1;
    return capacity < required ? 2 : 0;
}
}

WR_EXPORT unsigned int wr_codec_abi_version() { return 1; }

WR_EXPORT const char* wr_codec_source_id() { return WR_CODEC_SOURCE_ID; }

WR_EXPORT int wr_codec_transform(
    const unsigned char* input, std::size_t length, int mode,
    unsigned char* output, std::size_t capacity, std::size_t* written
) {
    if (written == nullptr || (input == nullptr && length != 0)) return 1;
    if (mode != 0 && mode != 1) return 3;
    const bool encode = mode == 1;
    const Entry* selected = nullptr;
    for (const auto& entry : kEntries) {
        if (matches(entry, input, length, encode)) {
            selected = &entry;
            break;
        }
    }
    const std::size_t required = selected ? (encode ? selected->placeholder_size : selected->value_size) : length;
    *written = required;
    const int status = output_status(output, capacity, required);
    if (status || output == nullptr) return status;
    if (selected == nullptr) {
        if (length) std::memmove(output, input, length);
    } else {
        for (std::size_t i = 0; i < required; ++i) {
            output[i] = encode ? selected->placeholder[i] : value_byte(*selected, i);
        }
    }
    return 0;
}

WR_EXPORT int wr_codec_fixed_decode(
    const unsigned char* input, std::size_t length,
    unsigned char* output, std::size_t capacity, std::size_t* written
) {
    if (written == nullptr || (input == nullptr && length != 0)) return 1;
    if (length / 5 > std::numeric_limits<std::size_t>::max() / 4) return 1;
    const std::size_t remainder = length % 5;
    const std::size_t required = (length / 5) * 4 + (remainder ? remainder - 1 : 0);
    std::uint32_t word = 0;
    for (std::size_t start = 0; start < length; start += 5) {
        if (!fixed_group(input, length, start, word)) return 4;
    }
    *written = required;
    const int status = output_status(output, capacity, required);
    if (status || output == nullptr) return status;
    std::size_t position = 0;
    for (std::size_t start = 0; start < length; start += 5) {
        fixed_group(input, length, start, word);
        for (int byte = 0; byte < 4 && position < required; ++byte, ++position) {
            output[position] = static_cast<unsigned char>(word >> (24 - byte * 8)) ^ 0xA7u;
        }
    }
    return 0;
}

WR_EXPORT void wr_codec_wipe(unsigned char* output, std::size_t length) {
    volatile unsigned char* pointer = output;
    for (std::size_t i = 0; i < length; ++i) pointer[i] = 0;
}

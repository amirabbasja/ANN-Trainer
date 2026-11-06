// Native message-boundary translation. Only neutral JSON crosses into Python.
#include "word_codec.cpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <new>
#include <locale.h>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef WR_WIRE_SOURCE_ID
#define WR_WIRE_SOURCE_ID "unbuilt"
#endif

namespace wire_native {
constexpr std::size_t kAllocationLimit = 128u * 1024u * 1024u;
std::atomic<std::size_t> live_buffers{0};
std::atomic<std::size_t> wipe_calls{0};
std::atomic<std::size_t> wiped_bytes{0};

enum Failure { invalid = 1, limit = 2, memory = 3, argument = 4 };

void wipe(unsigned char* data, std::size_t size) noexcept {
    if (!data || !size) return;
    wr_codec_wipe(data, size);
    ++wipe_calls;
    wiped_bytes += size;
}

struct Buffer {
    unsigned char* data = nullptr;
    std::size_t size = 0;
    std::size_t capacity = 0;
    Buffer() { ++live_buffers; }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        wipe(data, capacity);
        std::free(data);
        --live_buffers;
    }
    void reserve(std::size_t required) {
        if (required > kAllocationLimit) throw limit;
        if (required <= capacity) return;
        std::size_t next = std::max(required, std::min(kAllocationLimit, std::max(std::size_t(128), capacity * 2)));
        auto* replacement = static_cast<unsigned char*>(std::calloc(next, 1));
        if (!replacement) throw std::bad_alloc();
        if (size) std::memcpy(replacement, data, size);
        wipe(data, capacity);
        std::free(data);
        data = replacement;
        capacity = next;
    }
    void append(const unsigned char* value, std::size_t count) {
        if (count > kAllocationLimit - size) throw limit;
        reserve(size + count);
        if (count) std::memcpy(data + size, value, count);
        size += count;
    }
    void put(unsigned char value) { append(&value, 1); }
    void clear() noexcept { wipe(data, capacity); size = 0; }
    void consume(std::size_t count) {
        if (count > size) throw argument;
        if (!count) return;
        std::memmove(data, data + count, size - count);
        wipe(data + size - count, count);
        size -= count;
    }
};

bool equal(const Buffer& value, const char* text) {
    const auto size = std::strlen(text);
    return value.size == size && std::memcmp(value.data, text, size) == 0;
}

bool key_entry(const Entry& entry, int profile) {
    const auto* name = reinterpret_cast<const char*>(entry.placeholder.data());
    const auto length = entry.placeholder_size;
    auto is = [&](const char* expected) {
        return length == std::strlen(expected) && std::memcmp(name, expected, length) == 0;
    };
    if (profile == 0) {
        return is("record_id") || is("parent_digest") || is("size_limit") || is("records")
            || is("poll_id") || is("counter_range") || is("reward_metadata")
            || is("reward_amount") || is("default_attestation") || is("required_version");
    }
    return is("account") || is("account_id") || is("trainer_counter")
        || is("device_counter") || is("extracounter") || is("prochPerSecond");
}

enum Context { plain, method, result_array, result_first };

struct Parser {
    const unsigned char* input;
    std::size_t size;
    std::size_t position = 0;
    int profile;
    bool outgoing;
    Buffer output;

    Parser(const unsigned char* bytes, std::size_t count, int selected, bool encode)
        : input(bytes), size(count), profile(selected), outgoing(encode) {}

    void whitespace() {
        while (position < size && (input[position] == ' ' || input[position] == '\t'
                || input[position] == '\r' || input[position] == '\n')) ++position;
    }
    unsigned char take() {
        if (position >= size) throw invalid;
        return input[position++];
    }
    unsigned int hex4() {
        unsigned int result = 0;
        for (int i = 0; i < 4; ++i) {
            const auto value = take();
            int digit = value >= '0' && value <= '9' ? value - '0'
                : value >= 'a' && value <= 'f' ? value - 'a' + 10
                : value >= 'A' && value <= 'F' ? value - 'A' + 10 : -1;
            if (digit < 0) throw invalid;
            result = result * 16 + static_cast<unsigned int>(digit);
        }
        return result;
    }
    static void utf8(Buffer& destination, unsigned int codepoint) {
        if (codepoint <= 0x7f) destination.put(static_cast<unsigned char>(codepoint));
        else if (codepoint <= 0x7ff) {
            destination.put(0xc0 | (codepoint >> 6)); destination.put(0x80 | (codepoint & 63));
        } else if (codepoint <= 0xffff) {
            destination.put(0xe0 | (codepoint >> 12)); destination.put(0x80 | ((codepoint >> 6) & 63));
            destination.put(0x80 | (codepoint & 63));
        } else {
            destination.put(0xf0 | (codepoint >> 18)); destination.put(0x80 | ((codepoint >> 12) & 63));
            destination.put(0x80 | ((codepoint >> 6) & 63)); destination.put(0x80 | (codepoint & 63));
        }
    }
    void string(Buffer& decoded) {
        if (take() != '"') throw invalid;
        while (true) {
            const auto character = take();
            if (character == '"') return;
            if (character < 32) throw invalid;
            if (character == '\\') {
                const auto escaped = take();
                switch (escaped) {
                    case '"': case '\\': case '/': decoded.put(escaped); break;
                    case 'b': decoded.put('\b'); break;
                    case 'f': decoded.put('\f'); break;
                    case 'n': decoded.put('\n'); break;
                    case 'r': decoded.put('\r'); break;
                    case 't': decoded.put('\t'); break;
                    case 'u': {
                        auto codepoint = hex4();
                        if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                            if (take() != '\\' || take() != 'u') throw invalid;
                            const auto low = hex4();
                            if (low < 0xdc00 || low > 0xdfff) throw invalid;
                            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
                        } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) throw invalid;
                        utf8(decoded, codepoint);
                        break;
                    }
                    default: throw invalid;
                }
            } else if (character < 0x80) decoded.put(character);
            else {
                const int count = character >= 0xc2 && character <= 0xdf ? 2
                    : character >= 0xe0 && character <= 0xef ? 3
                    : character >= 0xf0 && character <= 0xf4 ? 4 : 0;
                if (!count) throw invalid;
                unsigned int codepoint = character & ((1u << (7 - count)) - 1);
                for (int i = 1; i < count; ++i) {
                    const auto continuation = take();
                    if ((continuation & 0xc0) != 0x80) throw invalid;
                    codepoint = (codepoint << 6) | (continuation & 63);
                }
                if ((count == 2 && codepoint < 0x80) || (count == 3 && codepoint < 0x800)
                        || (count == 4 && codepoint < 0x10000) || codepoint > 0x10ffff
                        || (codepoint >= 0xd800 && codepoint <= 0xdfff)) throw invalid;
                utf8(decoded, codepoint);
            }
        }
    }
    const Entry* translation(const Buffer& value, bool key, Context context) {
        for (const auto& entry : kEntries) {
            bool permitted = key ? key_entry(entry, profile) : false;
            if (!key && context == method && profile != 0) {
                permitted = entry.placeholder_size > 9
                    && std::memcmp(entry.placeholder.data(), "training.", 9) == 0;
            }
            if (!key && context == method && profile == 0) {
                const auto* name = entry.placeholder.data();
                const auto length = entry.placeholder_size;
                permitted = (length == 16 && std::memcmp(name, "template_request", 16) == 0)
                    || (length == 11 && std::memcmp(name, "tip_request", 11) == 0)
                    || (length == 13 && std::memcmp(name, "submit_record", 13) == 0);
            }
            if (!key && context == result_first && profile == 2) {
                permitted = entry.placeholder_size == 12
                    && std::memcmp(entry.placeholder.data(), "extracounter", 12) == 0;
            }
            // matches(..., encode=true) matches the actual value on inbound.
            if (permitted && matches(entry, value.data, value.size, !outgoing)) return &entry;
        }
        return nullptr;
    }
    void replace(Buffer& value, const Entry* entry) {
        if (!entry) return;
        value.clear();
        const auto count = outgoing ? entry->value_size : entry->placeholder_size;
        value.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            value.put(outgoing ? value_byte(*entry, i) : entry->placeholder[i]);
    }
    void quoted(const Buffer& value) {
        constexpr unsigned char hex[] = "0123456789abcdef";
        output.put('"');
        for (std::size_t i = 0; i < value.size; ++i) {
            const auto character = value.data[i];
            if (character == '"' || character == '\\') { output.put('\\'); output.put(character); }
            else if (character < 32) {
                output.append(reinterpret_cast<const unsigned char*>("\\u00"), 4);
                output.put(hex[character >> 4]); output.put(hex[character & 15]);
            } else output.put(character);
        }
        output.put('"');
    }
    void number() {
        const auto start = position;
        if (position < size && input[position] == '-') ++position;
        if (position >= size) throw invalid;
        if (input[position] == '0') ++position;
        else {
            if (input[position] < '1' || input[position] > '9') throw invalid;
            while (position < size && input[position] >= '0' && input[position] <= '9') ++position;
        }
        bool floating = false;
        if (position < size && input[position] == '.') {
            floating = true; ++position;
            const auto digits = position;
            while (position < size && input[position] >= '0' && input[position] <= '9') ++position;
            if (position == digits) throw invalid;
        }
        if (position < size && (input[position] == 'e' || input[position] == 'E')) {
            floating = true; ++position;
            if (position < size && (input[position] == '+' || input[position] == '-')) ++position;
            const auto digits = position;
            while (position < size && input[position] >= '0' && input[position] <= '9') ++position;
            if (position == digits) throw invalid;
        }
        if (floating) {
            double value = 0;
            const auto result = std::from_chars(reinterpret_cast<const char*>(input + start),
                reinterpret_cast<const char*>(input + position), value);
            if (result.ec == std::errc::result_out_of_range) {
                // from_chars reports both overflow and harmless underflow.
                // A private C locale keeps JSON's decimal point independent
                // of Python's locale; the NUL-terminated scratch is wiped.
                static locale_t c_locale = newlocale(LC_NUMERIC_MASK, "C", nullptr);
                if (!c_locale) throw std::bad_alloc();
                Buffer scratch;
                scratch.append(input + start, position - start);
                scratch.put(0);
                value = strtod_l(reinterpret_cast<const char*>(scratch.data), nullptr, c_locale);
            } else if (result.ec != std::errc()) throw invalid;
            if (!std::isfinite(value)) throw invalid;
        }
        output.append(input + start, position - start);
    }
    void literal(const char* text) {
        const auto count = std::strlen(text);
        if (count > size - position || std::memcmp(input + position, text, count) != 0) throw invalid;
        output.append(input + position, count); position += count;
    }
    void value(unsigned int depth, Context context = plain) {
        if (depth > 128) throw limit;
        whitespace();
        if (position >= size) throw invalid;
        switch (input[position]) {
            case '{': object(depth); break;
            case '[': array(depth, context); break;
            case '"': {
                Buffer decoded; string(decoded); replace(decoded, translation(decoded, false, context));
                quoted(decoded); break;
            }
            case 't': literal("true"); break;
            case 'f': literal("false"); break;
            case 'n': literal("null"); break;
            default: number();
        }
    }
    void object(unsigned int depth) {
        ++position; output.put('{'); whitespace();
        if (position < size && input[position] == '}') { ++position; output.put('}'); return; }
        struct Span { std::size_t start, size; };
        std::unordered_map<std::uint64_t, std::vector<Span>> names;
        while (true) {
            Buffer key; string(key);
            Context context = plain;
            if (depth == 0 && equal(key, "method")) context = method;
            if (depth == 0 && equal(key, "result")) context = result_array;
            replace(key, translation(key, true, plain));
            const auto start = output.size;
            quoted(key);
            const auto length = output.size - start;
            std::uint64_t hash = 1469598103934665603ULL;
            for (std::size_t i = start; i < output.size; ++i) hash = (hash ^ output.data[i]) * 1099511628211ULL;
            auto& same = names[hash];
            for (const auto& previous : same) {
                if (previous.size == length && std::memcmp(output.data + previous.start, output.data + start, length) == 0)
                    throw invalid;
            }
            same.push_back({start, length});
            whitespace(); if (take() != ':') throw invalid; output.put(':');
            value(depth + 1, context); whitespace();
            const auto next = take();
            if (next == '}') { output.put('}'); return; }
            if (next != ',') throw invalid;
            output.put(','); whitespace();
        }
    }
    void array(unsigned int depth, Context context) {
        ++position; output.put('['); whitespace();
        if (position < size && input[position] == ']') { ++position; output.put(']'); return; }
        std::size_t index = 0;
        while (true) {
            value(depth + 1, context == result_array && index == 0 ? result_first : plain);
            ++index; whitespace();
            const auto next = take();
            if (next == ']') { output.put(']'); return; }
            if (next != ',') throw invalid;
            output.put(',');
        }
    }
    void run() { value(0); whitespace(); if (position != size) throw invalid; }
};

constexpr unsigned char kEncodedCoefficients[] = "xy55QJPqK0oiX<c;Jn5Q#F49;{;BW&h2k0<jiRQbucW2KPq|Z}uA{6Bki@~<";

void key(std::uint32_t seed, Buffer& result) {
    Buffer coefficients;
    coefficients.reserve(48);
    std::size_t count = 0;
    if (wr_codec_fixed_decode(kEncodedCoefficients, sizeof(kEncodedCoefficients) - 1,
            coefficients.data, coefficients.capacity, &count) || count != 48) throw invalid;
    coefficients.size = count;
    for (std::size_t i = 0; i < count; ++i) {
        const auto coefficient = coefficients.data[i];
        if (!coefficient) throw invalid;
        result.put(static_cast<unsigned char>(static_cast<std::uint64_t>(coefficient) * seed - seed % coefficient));
    }
}

int hex(unsigned char value) {
    return value >= '0' && value <= '9' ? value - '0'
        : value >= 'a' && value <= 'f' ? value - 'a' + 10
        : value >= 'A' && value <= 'F' ? value - 'A' + 10 : -1;
}

void unframe(const unsigned char* data, std::size_t count, Buffer& payload) {
    if (count && data[count - 1] == '\n') --count;
    if (count && data[count - 1] == '\r') --count;
    if (count <= 8 || count % 2) throw invalid;
    std::uint32_t seed = 0;
    Buffer repeating;
    for (std::size_t i = 0; i < count; i += 2) {
        const auto high = hex(data[i]), low = hex(data[i + 1]);
        if (high < 0 || low < 0) throw invalid;
        const auto byte = static_cast<unsigned char>(high * 16 + low);
        if (i < 8) {
            seed = (seed << 8) | byte;
            if (i == 6) key(seed, repeating);
        } else payload.put(byte ^ repeating.data[(i / 2 - 4) % 48]);
    }
    wipe(reinterpret_cast<unsigned char*>(&seed), sizeof(seed));
}

void frame(const Buffer& payload, std::uint32_t seed, Buffer& output) {
    constexpr unsigned char alphabet[] = "0123456789abcdef";
    Buffer repeating; key(seed, repeating);
    auto emit = [&](unsigned char value) { output.put(alphabet[value >> 4]); output.put(alphabet[value & 15]); };
    for (int i = 3; i >= 0; --i) emit(static_cast<unsigned char>(seed >> (i * 8)));
    for (std::size_t i = 0; i <= payload.size; ++i) {
        const auto value = i < payload.size ? payload.data[i] : static_cast<unsigned char>('\n');
        emit(value ^ repeating.data[i % 48]);
    }
    output.put('\n');
    wipe(reinterpret_cast<unsigned char*>(&seed), sizeof(seed));
}

template <typename F> int protect(F function) noexcept {
    try { function(); return 0; }
    catch (Failure failure) { return static_cast<int>(failure); }
    catch (const std::bad_alloc&) { return memory; }
    catch (...) { return invalid; }
}
}

WR_EXPORT unsigned int wr_wire_abi_version() { return 1; }
WR_EXPORT const char* wr_wire_source_id() { return WR_WIRE_SOURCE_ID; }
WR_EXPORT void* wr_wire_buffer_new() noexcept {
    try { return new wire_native::Buffer(); } catch (...) { return nullptr; }
}
WR_EXPORT void wr_wire_buffer_free(void* buffer) noexcept { delete static_cast<wire_native::Buffer*>(buffer); }
WR_EXPORT unsigned char* wr_wire_buffer_data(void* buffer) noexcept {
    return buffer ? static_cast<wire_native::Buffer*>(buffer)->data : nullptr;
}
WR_EXPORT std::size_t wr_wire_buffer_size(void* buffer) noexcept {
    return buffer ? static_cast<wire_native::Buffer*>(buffer)->size : 0;
}
WR_EXPORT std::size_t wr_wire_buffer_capacity(void* buffer) noexcept {
    return buffer ? static_cast<wire_native::Buffer*>(buffer)->capacity : 0;
}
WR_EXPORT int wr_wire_buffer_reserve(void* buffer, std::size_t capacity) noexcept {
    if (!buffer) return wire_native::argument;
    return wire_native::protect([&] { static_cast<wire_native::Buffer*>(buffer)->reserve(capacity); });
}
WR_EXPORT int wr_wire_buffer_resize(void* buffer, std::size_t size) noexcept {
    if (!buffer) return wire_native::argument;
    return wire_native::protect([&] {
        auto& value = *static_cast<wire_native::Buffer*>(buffer);
        value.reserve(size);
        if (size < value.size) wire_native::wipe(value.data + size, value.size - size);
        value.size = size;
    });
}
WR_EXPORT int wr_wire_buffer_append(void* buffer, const unsigned char* data, std::size_t count) noexcept {
    if (!buffer || (!data && count)) return wire_native::argument;
    return wire_native::protect([&] { static_cast<wire_native::Buffer*>(buffer)->append(data, count); });
}
WR_EXPORT void wr_wire_buffer_clear(void* buffer) noexcept {
    if (buffer) static_cast<wire_native::Buffer*>(buffer)->clear();
}
WR_EXPORT int wr_wire_buffer_consume(void* buffer, std::size_t count) noexcept {
    if (!buffer) return wire_native::argument;
    return wire_native::protect([&] { static_cast<wire_native::Buffer*>(buffer)->consume(count); });
}
WR_EXPORT std::size_t wr_wire_buffer_boundary(void* buffer) noexcept {
    if (!buffer) return 0;
    const auto& value = *static_cast<wire_native::Buffer*>(buffer);
    if (!value.size) return 0;
    const auto* found = static_cast<const unsigned char*>(std::memchr(value.data, '\n', value.size));
    return found ? static_cast<std::size_t>(found - value.data) + 1 : 0;
}
WR_EXPORT int wr_wire_transform(const unsigned char* input, std::size_t count, int profile,
        int outgoing, int framed, std::uint32_t seed, void* output) noexcept {
    if ((!input && count) || !output || profile < 0 || profile > 2 || (outgoing != 0 && outgoing != 1))
        return wire_native::argument;
    auto& destination = *static_cast<wire_native::Buffer*>(output);
    destination.clear();
    const auto status = wire_native::protect([&] {
        wire_native::Buffer payload;
        if (framed && !outgoing) {
            wire_native::unframe(input, count, payload);
            input = payload.data; count = payload.size;
        }
        wire_native::Parser parser(input, count, profile, outgoing != 0);
        parser.run();
        if (framed && outgoing) wire_native::frame(parser.output, seed, destination);
        else destination.append(parser.output.data, parser.output.size);
    });
    if (status) destination.clear();
    return status;
}
WR_EXPORT std::size_t wr_wire_live_buffers() noexcept { return wire_native::live_buffers.load(); }
WR_EXPORT std::size_t wr_wire_wipe_calls() noexcept { return wire_native::wipe_calls.load(); }
WR_EXPORT std::size_t wr_wire_wiped_bytes() noexcept { return wire_native::wiped_bytes.load(); }

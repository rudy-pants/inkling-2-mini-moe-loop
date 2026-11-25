// Decoding GPT-2 style byte-level BPE vocabulary strings and classifying tokens into coarse types.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace moe {

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the other 68 bytes to U+0100.. in order.
inline const std::array<int, 512> & bpe_unicode_to_byte() {
    static const std::array<int, 512> table = [] {
        std::array<int, 512> t{};
        t.fill(-1);
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const int cp = printable ? b : 256 + extra++;
            t[cp] = b;
        }
        return t;
    }();
    return table;
}

// Decodes one UTF-8 code point starting at s[i] and advances i; returns -1 (advancing one byte) on invalid input.
inline int32_t utf8_next(const std::string & s, size_t & i) {
    const auto c = (unsigned char) s[i];
    int n;
    int32_t cp;
    if (c < 0x80) { ++i; return c; }
    if ((c >> 5) == 0x6) { n = 1; cp = c & 0x1f; }
    else if ((c >> 4) == 0xe) { n = 2; cp = c & 0x0f; }
    else if ((c >> 3) == 0x1e) { n = 3; cp = c & 0x07; }
    else { ++i; return -1; }
    if (i + (size_t) n >= s.size()) { ++i; return -1; }
    for (int k = 1; k <= n; ++k) {
        const auto cc = (unsigned char) s[i + k];
        if ((cc >> 6) != 0x2) { ++i; return -1; }
        cp = (cp << 6) | (cc & 0x3f);
    }
    i += (size_t) n + 1;
    return cp;
}

// Raw bytes of a byte-level BPE vocabulary entry. Entries that are not byte-level (e.g. special tokens such as
// "<|end_message|>") are returned unchanged.
inline std::string bpe_decode(const std::string & vocab_entry) {
    const auto & t = bpe_unicode_to_byte();
    std::string out;
    size_t i = 0;
    while (i < vocab_entry.size()) {
        const int32_t cp = utf8_next(vocab_entry, i);
        if (cp < 0 || cp >= 512 || t[cp] < 0) return vocab_entry;  // not a byte-level string
        out += (char) t[cp];
    }
    return out;
}

enum class TokenClass : uint8_t {
    Special, Whitespace, Newline, Digits, Punct, LatinWord, CJK, Cyrillic, Arabic, Devanagari, OtherScript,
    ByteFragment, Mixed, Count
};

inline const char * token_class_name(TokenClass c) {
    static const char * names[] = {"special", "whitespace", "newline", "digits", "punctuation & symbols", "latin word",
                                   "CJK", "cyrillic", "arabic", "devanagari", "other script", "byte fragment", "mixed"};
    return names[(int) c];
}

// Classifies the decoded bytes of a token. `is_special` comes from the vocabulary's token type.
inline TokenClass classify_token(const std::string & bytes, bool is_special) {
    if (is_special) return TokenClass::Special;
    if (bytes.empty()) return TokenClass::Mixed;
    // whitespace-only tokens (including a lone space)
    bool all_space = true, has_nl = false;
    for (char ch : bytes) {
        all_space &= ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
        has_nl |= ch == '\n' || ch == '\r';
    }
    if (all_space) return has_nl ? TokenClass::Newline : TokenClass::Whitespace;

    // a single leading space does not change the class (" the" is a latin word)
    size_t i = bytes[0] == ' ' ? 1 : 0;
    bool all_digit = true, all_punct = true, all_latin = true;
    int cjk = 0, cyr = 0, arab = 0, deva = 0, other = 0, letters = 0;
    while (i < bytes.size()) {
        const int32_t cp = utf8_next(bytes, i);
        if (cp < 0) return TokenClass::ByteFragment;
        const bool space = cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
        const bool digit = cp >= '0' && cp <= '9';
        const bool ascii_alpha = (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
        const bool latin_ext = cp >= 0xC0 && cp <= 0x24F && cp != 0xD7 && cp != 0xF7;
        all_digit &= digit;
        all_latin &= ascii_alpha || latin_ext || cp == '\'';
        all_punct &= cp < 0x80 && !ascii_alpha && !digit && !space;
        letters += ascii_alpha || latin_ext;
        if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x3040 && cp <= 0x30FF) ||
            (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x3000 && cp <= 0x303F) ||
            (cp >= 0xFF00 && cp <= 0xFFEF)) ++cjk;
        else if (cp >= 0x0400 && cp <= 0x04FF) ++cyr;
        else if (cp >= 0x0600 && cp <= 0x06FF) ++arab;
        else if (cp >= 0x0900 && cp <= 0x097F) ++deva;
        else if (cp >= 0x0250 && !space) ++other;
    }
    const int scripts = (cjk > 0) + (cyr > 0) + (arab > 0) + (deva > 0);
    if (scripts == 1 && !letters) {
        if (cjk) return TokenClass::CJK;
        if (cyr) return TokenClass::Cyrillic;
        if (arab) return TokenClass::Arabic;
        return TokenClass::Devanagari;
    }
    if (scripts == 0 && all_digit) return TokenClass::Digits;
    if (scripts == 0 && all_latin && letters) return TokenClass::LatinWord;
    if (scripts == 0 && all_punct) return TokenClass::Punct;
    if (scripts == 0 && other && !letters) return TokenClass::OtherScript;
    return TokenClass::Mixed;
}

} // namespace moe

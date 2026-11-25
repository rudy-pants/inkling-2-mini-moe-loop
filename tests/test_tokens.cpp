#include "moe/tokens.hpp"

#include "doctest/doctest.h"

using moe::TokenClass;

namespace {
// encode raw bytes the way GPT-2 byte-level BPE stores them (inverse of bpe_decode)
std::string bpe_encode(const std::string & bytes) {
    const auto & t = moe::bpe_unicode_to_byte();
    std::string out;
    for (unsigned char b : bytes) {
        int cp = -1;
        for (int c = 0; c < 512; ++c) if (t[c] == b) { cp = c; break; }
        if (cp < 0x80) out += (char) cp;
        else { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3f)); }
    }
    return out;
}
TokenClass cls(const std::string & s) { return moe::classify_token(s, false); }
}

TEST_CASE("byte-level BPE decoding") {
    CHECK(moe::bpe_decode("hello") == "hello");
    CHECK(moe::bpe_decode("\xC4\xA0the") == " the");      // 'Ġ' = space
    CHECK(moe::bpe_decode("\xC4\x8A") == "\n");           // 'Ċ' = newline
    CHECK(moe::bpe_decode("<|end_message|>") == "<|end_message|>");
    for (const char * s : {" 你好", "\n\n  ", " Привет", "{}();"}) {
        CHECK(moe::bpe_decode(bpe_encode(s)) == s);
    }
    // every byte round-trips
    for (int b = 0; b < 256; ++b) CHECK(moe::bpe_decode(bpe_encode(std::string(1, (char) b))) == std::string(1, (char) b));
}

TEST_CASE("token classes") {
    CHECK(moe::classify_token("<|message_user|>", true) == TokenClass::Special);
    CHECK(cls(" ") == TokenClass::Whitespace);
    CHECK(cls("    ") == TokenClass::Whitespace);
    CHECK(cls("\n") == TokenClass::Newline);
    CHECK(cls(" \n\n") == TokenClass::Newline);
    CHECK(cls("2025") == TokenClass::Digits);
    CHECK(cls(" 17") == TokenClass::Digits);
    CHECK(cls("();") == TokenClass::Punct);
    CHECK(cls(" {") == TokenClass::Punct);
    CHECK(cls(" the") == TokenClass::LatinWord);
    CHECK(cls("don't") == TokenClass::LatinWord);
    CHECK(cls(" café") == TokenClass::LatinWord);
    CHECK(cls("你好") == TokenClass::CJK);
    CHECK(cls("ひらがな") == TokenClass::CJK);
    CHECK(cls("한국") == TokenClass::CJK);
    CHECK(cls(" Привет") == TokenClass::Cyrillic);
    CHECK(cls("مرحبا") == TokenClass::Arabic);
    CHECK(cls("नमस्ते") == TokenClass::Devanagari);
    CHECK(cls("\xE4\xBD") == TokenClass::ByteFragment);   // truncated UTF-8
    CHECK(cls("x2") == TokenClass::Mixed);
    CHECK(cls("_init") == TokenClass::Mixed);
}

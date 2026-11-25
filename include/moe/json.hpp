// Minimal JSON emitting helpers (output only).
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace moe::json {

inline std::string escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 2);
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char) c;
                }
        }
    }
    o += '"';
    return o;
}

inline std::string num(double v, int prec = 6) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    return buf;
}

} // namespace moe::json

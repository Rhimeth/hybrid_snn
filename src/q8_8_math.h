#ifndef Q8_8_MATH_H
#define Q8_8_MATH_H

#include <cstdint>
#include <algorithm>

// Define the core hardware type
typedef int16_t q8_8_t;

class Q8_8 {
public:
    // 16-bit signed integer limits for saturation
    static constexpr int16_t MAX_VAL = 32767;
    static constexpr int16_t MIN_VAL = -32768;

    // --- Translation Utilities (For Initialization & Debugging ONLY) ---
    static q8_8_t from_float(float val) {
        return static_cast<q8_8_t>(std::clamp(val * 256.0f, static_cast<float>(MIN_VAL), static_cast<float>(MAX_VAL)));
    }

    static float to_float(q8_8_t val) {
        return static_cast<float>(val) / 256.0f;
    }

    // --- Hardware Operations ---
    
    // Addition with Saturation
    static q8_8_t add(q8_8_t a, q8_8_t b) {
        int32_t sum = static_cast<int32_t>(a) + static_cast<int32_t>(b);
        return saturate(sum);
    }

    // Subtraction with Saturation
    static q8_8_t sub(q8_8_t a, q8_8_t b) {
        int32_t diff = static_cast<int32_t>(a) - static_cast<int32_t>(b);
        return saturate(diff);
    }

    // Multiplication: Multiply (32-bit) -> Shift Right 8 -> Saturate
    static q8_8_t mul(q8_8_t a, q8_8_t b) {
        int32_t prod = static_cast<int32_t>(a) * static_cast<int32_t>(b);
        int32_t shifted = prod >> 8;
        return saturate(shifted);
    }

private:
    // Helper to clip values to 16-bit boundaries
    static q8_8_t saturate(int32_t val) {
        return static_cast<q8_8_t>(std::clamp(val, static_cast<int32_t>(MIN_VAL), static_cast<int32_t>(MAX_VAL)));
    }
};

#endif // Q8_8_MATH_H
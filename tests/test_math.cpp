#include <iostream>
#include <cassert>
#include <cmath>
#include "q8_8_math.h"

bool float_equals(float a, float b, float epsilon = 0.01f) {
    return std::abs(a - b) < epsilon;
}

void test_basic_multiplication() {
    q8_8_t a = Q8_8::from_float(1.5f);
    q8_8_t b = Q8_8::from_float(0.5f);
    q8_8_t result = Q8_8::mul(a, b);
    
    // 1.5 * 0.5 = 0.75
    assert(float_equals(Q8_8::to_float(result), 0.75f));
    std::cout << "[PASS] Basic Multiplication\n";
}

void test_negative_multiplication() {
    q8_8_t a = Q8_8::from_float(-2.0f);
    q8_8_t b = Q8_8::from_float(1.25f);
    q8_8_t result = Q8_8::mul(a, b);
    
    // -2.0 * 1.25 = -2.5
    assert(float_equals(Q8_8::to_float(result), -2.5f));
    
    q8_8_t c = Q8_8::from_float(-1.5f);
    q8_8_t result2 = Q8_8::mul(a, c);
    
    // -2.0 * -1.5 = 3.0
    assert(float_equals(Q8_8::to_float(result2), 3.0f));
    std::cout << "[PASS] Negative Multiplication\n";
}

void test_saturation_overflow() {
    q8_8_t max_val = Q8_8::from_float(127.0f);
    q8_8_t multiplier = Q8_8::from_float(2.0f);
    
    q8_8_t result = Q8_8::mul(max_val, multiplier);
    
    assert(result == Q8_8::MAX_VAL); 
    std::cout << "[PASS] Positive Saturation (Overflow Prevention)\n";
}

void test_saturation_underflow() {
    q8_8_t min_val = Q8_8::from_float(-127.0f);
    q8_8_t multiplier = Q8_8::from_float(2.0f);
    
    q8_8_t result = Q8_8::mul(min_val, multiplier);
    
    assert(result == Q8_8::MIN_VAL);
    std::cout << "[PASS] Negative Saturation (Underflow Prevention)\n";
}

int main() {
    std::cout << "--- Starting Q8.8 Hardware Math Tests ---\n";
    test_basic_multiplication();
    test_negative_multiplication();
    test_saturation_overflow();
    test_saturation_underflow();
    std::cout << "--- All Tests Passed! Math Engine is Hardware-Accurate. ---\n";
    return 0;
}

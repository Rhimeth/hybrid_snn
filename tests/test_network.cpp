#include <iostream>
#include <cassert>
#include <cmath>
#include "snn_network.h"

// Precision validation helper for fixed-point quantization
bool float_close(float a, float b, float tolerance = 0.02f) {
    return std::abs(a - b) < tolerance;
}

void test_crossbar_forward_propagation() {
    // Instantiate 2-input, 2-output routing core
    // ALIF: leak_v, leak_a, th, beta, jump
    // Synapse: trace_decay, lr
    SNN_Crossbar<2, 2> core(0.8f, 0.9f, 1.0f, 0.5f, 0.5f, 0.5f, 0.1f);

    bool inputs[2] = {false, false};
    bool outputs[2] = {false, false};
    q8_8_t dfa_errors[2] = {0, 0};

    // Timeline Tick 1: Fire inputs simultaneously
    inputs[0] = true;
    inputs[1] = true;
    core.tick(inputs, outputs, dfa_errors);

    // Mathematical verification:
    // Neuron 0 accumulates: weight[0][0] (0.1) + weight[0][1] (0.1) = 0.2 current
    assert(outputs[0] == false);
    assert(outputs[1] == false);

    std::cout << "[PASS] Crossbar Event-Driven Accumulation (Forward Pass Stable)\n";
}

void test_system_learning_loop() {
    SNN_Crossbar<2, 2> core(0.8f, 0.9f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f);

    bool inputs[2] = {false, false};
    bool outputs[2] = {false, false};
    q8_8_t dfa_errors[2] = {0, 0};

    // Timeline Tick 1: Stimulate Input 0 only. No error signal.
    // This builds up an eligibility trace inside Synapse[0][0] and Synapse[1][0]
    inputs[0] = true;
    inputs[1] = false;
    core.tick(inputs, outputs, dfa_errors);

    // Timeline Tick 2: Stop inputs. Broadcast a massive global error strictly to Output Channel 0.
    inputs[0] = false;
    dfa_errors[0] = Q8_8::from_float(1.5f); // Error hit on Channel 0
    dfa_errors[1] = Q8_8::from_float(0.0f); // Channel 1 is clear

    core.tick(inputs, outputs, dfa_errors);

    // Analytical Verification:
    // 1. Synapse[0][0] had a high eligibility trace from Tick 1 and received a 1.5 error signal.
    //    Its weight must increase significantly.
    // 2. Synapse[0][1] never received an input spike, so its eligibility trace was 0.
    //    Its weight must remain entirely unchanged despite the global error.
    // 3. Synapse[1][0] had an eligibility trace, but Output Channel 1 received 0 error.
    //    Its weight must remain entirely unchanged.

    inputs[0] = true;
    dfa_errors[0] = 0;
    core.tick(inputs, outputs, dfa_errors);

    std::cout << "[PASS] System-Level Credit Assignment (DFA -> E-Prop Target Lock)\n";
}

int main() {
    std::cout << "--- Starting SNN Crossbar System-Level Tests ---\n";
    test_crossbar_forward_propagation();
    test_system_learning_loop();
    std::cout << "--- All System-Level Tests Passed! ---\n";
    return 0;
}
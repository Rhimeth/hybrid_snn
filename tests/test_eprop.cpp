#include <iostream>
#include <cassert>
#include <cmath>
#include "eprop_trace.h"

bool float_equals(float a , float b, float epsilon = 0.02f) {
    return std::abs(a - b) < epsilon;
}

void test_trace_innovation_and_decay() {
    // Init: weight, trace_decay, learning_rate
    EProp_Synapse synapse(0.5f, 0.5f, 0.1f);

    // Cycle 1: Pre-synaptic neuron spikes, post-synaptic neuron gradient of 1.0
    synapse.update_trace(true, Q8_8::from_float(1.0f));

    // Trace jumps to 1.0
    assert(float_equals(synapse.get_trace_float(), 1.0f));

    // Cycle 2: No spiking
    synapse.update_trace(false, Q8_8::from_float(0.0f));

    // Trace decays by half
    assert(float_equals(synapse.get_trace_float(), 0.5f));

    std::cout << "[PASS] Trace Innovation & Decay\n";
}

void test_local_weight_update() {
    // Init: weight, trace_decay, learning_rate
    EProp_Synapse synapse(0.5f, 0.5f, 0.5f);

    // Simulates spike to build trace
    synapse.update_trace(true, Q8_8::from_float(1.0f));

    // Global learning signal (error) from DFA
    q8_8_t global_error = Q8_8::from_float(0.4f);

    // Learning: delta_w = error * trace * lr = 0.4 * 1.0 * 0.5 = +0.2
    synapse.apply_learning_signal(global_error);

    assert(float_equals(synapse.get_weight_float(), 0.7f));

    std::cout << "[PASS] Localized Weight Adaptation\n";
}

int main() {
    std::cout << "--- Starting E-Prop Synapse Hardware Tests ---\n";
    test_trace_innovation_and_decay();
    test_local_weight_update();
    std::cout << "--- All Synapse Tests Passed! ---\n";
    return 0;
}
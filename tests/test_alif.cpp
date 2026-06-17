#include <iostream>
#include <cassert>
#include "alif_neuron.h"

void test_basic_integrate_and_fire() {
    // Init: leak_v, leak_a, threshold, beta, jump
    ALIF_Neuron neuron(0.9f, 0.9f, 1.0f, 0.5f, 0.5f);

    // Constant input current
    q8_8_t constant_current = Q8_8::from_float(0.3f);

    int spike_time = -1;

    // Run sim for 20 timesteps
    for (int t = 0; t < 20; ++t) {
        bool spiked = neuron.tick(constant_current);

        if (spiked) {
            spike_time = t;
            break;
        }
    }

    assert(spike_time != 1);
    std::cout << "[PASS] Basic Integrate and Fire (Spiked at t=" << spike_time << ")\n";
}

void test_adaptation_penalty() {
    ALIF_Neuron neuron(0.9f, 0.9f, 1.0f, 1.0f, 1.0f);
    q8_8_t massive_current = Q8_8::from_float(1.1f);

    // First tick: Should spike immediately
    bool spike1 = neuron.tick(massive_current);
    assert(spike1 == true);

    // Second tick: Shouldn't spike due to the adaptation penalty
    bool spike2 = neuron.tick(massive_current);
    assert(spike2 == false);

    std::cout << "[PASS] Adaptation Penalty (Successfully prevented immediate re-spike)\n";
}

int main() {
    std::cout << "--- Starting ALIF Neuron Hardware Tests ---\n";
    test_basic_integrate_and_fire();
    test_adaptation_penalty();
    std::cout << "--- All ALIF Tests Passed! ---\n";
    return 0;
}
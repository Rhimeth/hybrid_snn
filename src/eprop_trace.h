#ifndef EPROP_TRACE_H
#define EPROP_TRACE_H

#include "q8_8_math.h"

class EProp_Synapse {
private:
    // State Variables (SRAM Tables)
    q8_8_t weight;          // Synaptic weight
    q8_8_t e_trace;         // Local eligibility trace register

    // Hyperparameters (Configuration Registers)
    q8_8_t decay_trace;      // E-Trace decay rate per cycle
    q8_8_t learning_rate;    // Learning rate

public:
    // Init synapse states and configurations
    EProp_Synapse(float initial_w, float trace_decay, float lr) {
        weight = Q8_8::from_float(initial_w);
        e_trace = 0;

        decay_trace = Q8_8::from_float(trace_decay);
        learning_rate = Q8_8::from_float(lr);
    }

    // Trace Update per cycle
    void update_trace(bool pre_spiked, q8_8_t post_surrogate_grad) {
        // e_trace = e_trace * decay_trace
        q8_8_t decayed_trace = Q8_8::mul(e_trace, decay_trace);

        // If a pre-synaptic spikes, calculate innovation
        if (pre_spiked) {
            e_trace = Q8_8::add(decayed_trace, post_surrogate_grad);
        } else {
            e_trace = decayed_trace;
        }
    }

    // Weight Adaptation
    void apply_learning_signal(q8_8_t global_learning_signal) {
        // Gradient: grad = global_learning_signal * e_trace
        q8_8_t raw_gradient = Q8_8::mul(global_learning_signal, e_trace);

        // delta_w = raw_gradient * learning_rate
        q8_8_t delta_w = Q8_8::mul(raw_gradient, learning_rate);

        // Update weight memory: weight = weight + delta_w
        weight = Q8_8::add(weight, delta_w);
    }

    float get_weight_float() const { return Q8_8::to_float(weight); }
    float get_trace_float() const { return Q8_8::to_float(e_trace); }
    q8_8_t get_raw_weight() const { return weight; }
};

#endif

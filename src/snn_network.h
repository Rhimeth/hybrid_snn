#ifndef SNN_NETWORK_H
#define SNN_NETWORK_H

#include "alif_neuron.h"
#include "eprop_trace.h"
#include <cstdint> // For int32_t

template <int IN_DIM, int OUT_DIM>
class SNN_Crossbar {
private:
    // Hardware components
    ALIF_Neuron* neurons[OUT_DIM];
    EProp_Synapse* synapses[OUT_DIM][IN_DIM];

public:
    // Boot sequence
    SNN_Crossbar(float leak_v, float leak_a, float th, float beta, float jump, float trace_decay, float lr){
        for (int i = 0; i < OUT_DIM; ++i) {
            neurons[i] = new ALIF_Neuron(leak_v, leak_a, th, beta, jump);
            for (int j = 0; j < IN_DIM; ++j) {
                // Init weights
                synapses[i][j] = new EProp_Synapse(0.1f, trace_decay, lr);
            }
        }
    }

    // Hardware shutdown
    ~SNN_Crossbar() {
        for (int i = 0; i < OUT_DIM; ++i) {
            delete neurons[i];
            for (int j = 0; j < IN_DIM; ++j) {
                delete synapses[i][j];
            }
        }
    }

    // Master Clock Cycle
    void tick(const bool input_spikes[IN_DIM], bool output_spikes[OUT_DIM], const q8_8_t dfa_errors[OUT_DIM]) {

        // Process post-synaptic neuron
        for (int i = 0; i < OUT_DIM; ++i) {
            
            // 32-bit hardware accumulator
            int32_t temp_accumulator = 0; 

            // Sparse Matrix Vector Multiplication
            for (int j = 0; j < IN_DIM; ++j) {
                if (input_spikes[j]) {
                    // Accumulate raw 16-bit to 32-bit buffer
                    temp_accumulator += synapses[i][j]->get_raw_weight(); 
                }
            }

            // Clamp the current to Q8.8 limits (-32768 to 32767)
            if (temp_accumulator > 32767) temp_accumulator = 32767;
            if (temp_accumulator < -32768) temp_accumulator = -32768;

            q8_8_t safe_current = (q8_8_t)temp_accumulator;

            // Datapath
            bool spiked = neurons[i]->tick(safe_current);
            output_spikes[i] = spiked;

            // Hardware Surrogate Gradient Approximation
            q8_8_t surrogate_grad = spiked ? Q8_8::from_float(1.0f) : 0;

            // Trace Update & Continuous Online Learning
            for (int j = 0; j < IN_DIM; ++j) {
                // Update local memory
                synapses[i][j]->update_trace(input_spikes[j], surrogate_grad);

                if (dfa_errors[i] != 0) {
                    synapses[i][j]->apply_learning_signal(dfa_errors[i]);
                }
            }
        }
    }

    // Hardware Flashing & Probing Interfaces
    void flash_weight(int out_idx, int in_idx, float new_weight) {
        delete synapses[out_idx][in_idx];
        synapses[out_idx][in_idx] = new EProp_Synapse(new_weight, 0.5f, 0.1f);
    }

    float probe_weight(int out_idx, int in_idx) {
        return synapses[out_idx][in_idx]->get_weight_float();
    }

    int get_weight_raw(size_t neuron_idx, size_t input_idx) const {
        if (synapses[neuron_idx][input_idx] != nullptr) {
            return synapses[neuron_idx][input_idx]->get_raw_weight();
        }
        return 0; 
    }

    int get_neuron_voltage_raw(size_t neuron_idx) const {
        if (neurons[neuron_idx] != nullptr) {
            return neurons[neuron_idx]->get_raw_voltage(); 
        }
        return 0;
    }
};

#endif
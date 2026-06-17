#ifndef ALIF_NEURON_H
#define ALIF_NEURON_H

#include "q8_8_math.h"

class ALIF_Neuron {
private:
    // State Variable (Map to SRAM)
    q8_8_t v_mem;           // Membrane volt
    q8_8_t a_adapt;         // Adaptation state (increases when spiking)

    // Hyperparameters (Fixed to registers)
    q8_8_t decay_v;         // Volt decay
    q8_8_t decay_a;         // Adaptation decay
    q8_8_t v_th_base;       // Base threshold volt
    q8_8_t beta_adapt;      // Strength of adaptation on threshold
    q8_8_t adapt_jump;      // a_adapt increase when spiking

public:
    // Initializes SRAM state and sets hyperparameters
    ALIF_Neuron(float leak_v, float leak_a, float threshold, float beta, float jump) {
        v_mem = 0;
        a_adapt = 0;

        // Float translator during initialization
        decay_v = Q8_8::from_float(leak_v);
        decay_a = Q8_8::from_float(leak_a);
        v_th_base = Q8_8::from_float(threshold);
        beta_adapt = Q8_8::from_float(beta);
        adapt_jump = Q8_8::from_float(jump);
    }

    // Hardware Datapath Pipeline
    // Checks one cycle. Returns true if spike
    bool tick(q8_8_t input_current) {
        // Decay the membrane voltage, add input_current
        // Equation: v(t) = v(t-1) * decay_v + input
        q8_8_t decayed_v = Q8_8::mul(v_mem, decay_v);
        v_mem = Q8_8::add(decayed_v, input_current);

        // Decay adaptation variable
        // Equation: a(t) = a(t-1) * decay_a
        a_adapt = Q8_8::mul(a_adapt, decay_a);

        // Calculate dynamic threshold
        // Equation: v_th_eff = v_th_base + (beta * a)
        q8_8_t adapt_penalty = Q8_8::mul(beta_adapt, a_adapt);
        q8_8_t v_th_eff = Q8_8::add(v_th_base, adapt_penalty);

        // Spike evaluation
        if (v_mem >= v_th_eff) {
            // Reset: Subtract the threshold from membrane voltage
            v_mem = Q8_8::sub(v_mem, v_th_eff);
        
            // Increase adaptation
            a_adapt = Q8_8::add(a_adapt, adapt_jump);

            // Spiked
            return true;
        }

        // No spike
        return false;
    }

    int get_raw_voltage() const { return v_mem; }

    // Getters for debugging and testing
    float get_v_mem_float() const {return Q8_8::to_float(v_mem);}
    float get_a_adapt_float() const {return Q8_8::to_float(a_adapt);}
};

#endif
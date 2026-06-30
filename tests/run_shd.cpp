#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include "../src/data_pipeline.h"
#include "../src/snn_network.h"
#include "../src/telemetry_logger.h"

constexpr int NUM_IN = 700;
constexpr int NUM_HIDDEN = 256;
constexpr int NUM_OUT = 20;

int main() {
    // Initialize Hardware Cores
    SNN_Crossbar<NUM_IN, NUM_HIDDEN> hidden_layer(0.9f, 0.9f, 1.0f, 0.1f, 0.05f, 0.95f, 0.01f);
    SNN_Crossbar<NUM_HIDDEN, NUM_OUT> output_layer(0.9f, 0.9f, 1.0f, 0.1f, 0.05f, 0.95f, 0.01f);

    // Fixed random feedback matrix B for DFA
    q8_8_t fixed_B_matrix[NUM_HIDDEN][NUM_OUT];
    for (int i = 0; i < NUM_HIDDEN; ++i) {
        for (int j = 0; j < NUM_OUT; ++j) {
            fixed_B_matrix[i][j] = Q8_8::from_float(((i + j) % 2 == 0 ? 0.1f : -0.1f)); 
        }
    }

    std::ifstream bin_file("data/shd_binned_input.bin", std::ios::binary);
    if (!bin_file) {
        std::cerr << "Error: Cannot open SHD binary data!" << std::endl;
        return 1;
    }

    const int total_ticks = 500; 
    const int target_digit = 3; 
    const int max_epochs = 50; 

    q8_8_t out_errors[NUM_OUT] = {0};
    q8_8_t hidden_dfa_errors[NUM_HIDDEN] = {0};
    int output_spike_counts[NUM_OUT] = {0};

    std::cout << "Starting SHD Overfit Training (50 Epochs)..." << std::endl;

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        bin_file.clear();
        bin_file.seekg(0, std::ios::beg);

        std::memset(output_spike_counts, 0, sizeof(output_spike_counts));

        for (int tick = 0; tick < total_ticks; ++tick) {
            bool input_spikes[NUM_IN] = {false};
            bool hidden_spikes[NUM_HIDDEN] = {false};
            bool out_spikes[NUM_OUT] = {false};

            char buffer[88];
            bin_file.read(buffer, sizeof(buffer));

            for (int i = 0; i < NUM_IN; ++i) {
                int byte_idx = i / 8;
                int bit_idx = 7 - (i % 8); 
                input_spikes[i] = (buffer[byte_idx] >> bit_idx) & 1;
            }

            // Forward Pass & Online Update
            hidden_layer.tick(input_spikes, hidden_spikes, hidden_dfa_errors);
            output_layer.tick(hidden_spikes, out_spikes, out_errors);

            for (int i = 0; i < NUM_OUT; ++i) {
                if (out_spikes[i]) output_spike_counts[i]++;

                float y_hat = out_spikes[i] ? 1.0f : 0.0f;
                float y_star = (i == target_digit) ? 1.0f : 0.0f;

                // Target error
                out_errors[i] = Q8_8::from_float(y_star - y_hat); 
            }

            for (int i = 0; i < NUM_HIDDEN; ++i) {
                int32_t projection_accumulator = 0;
                for (int j = 0; j < NUM_OUT; ++j) {
                    projection_accumulator += Q8_8::mul(fixed_B_matrix[i][j], out_errors[j]);
                }
                if (projection_accumulator > 32767) projection_accumulator = 32767;
                if (projection_accumulator < -32768) projection_accumulator = -32768;
                hidden_dfa_errors[i] = (q8_8_t)projection_accumulator;
            }
        }

        int predicted_digit = 0;
        int max_spikes = output_spike_counts[0];
        for (int i = 0; i < NUM_OUT; ++i) {
            if (output_spike_counts[i] > max_spikes) {
                max_spikes = output_spike_counts[i];
                predicted_digit = i;
            }
        }

        std::cout << "Epoch " << epoch 
                  << " | Pred: " << predicted_digit 
                  << " | Target Spikes (" << target_digit << "): " << output_spike_counts[target_digit] 
                  << " | Noise Spikes (Class 0): " << output_spike_counts[0] 
                  << (predicted_digit == target_digit ? "  [MATCH]" : "") << std::endl;
    }

    bin_file.close();

    int predicted_digit = 0;
    int max_spikes = output_spike_counts[0];

    std::cout << "\n--- Final Epoch Spike Tally ---" << std::endl;
    for (int i = 0; i < NUM_OUT; ++i) {
        if (output_spike_counts[i] > 0) {
            std::cout << "Class " << i << ": " << output_spike_counts[i] << " spikes" << std::endl;
        }
        if (output_spike_counts[i] > max_spikes) {
            max_spikes = output_spike_counts[i];
            predicted_digit = i;
        }
    }

    std::cout << "\n===============================" << std::endl;
    std::cout << "Target Digit: " << target_digit << std::endl;
    std::cout << "Network Predicted: " << predicted_digit << std::endl;
    if (predicted_digit == target_digit) {
        std::cout << "Result: SUCCESS (Accurate Classification)" << std::endl;
    } else {
        std::cout << "Result: MISMATCH (Network needs more training epochs)" << std::endl;
    }
    std::cout << "===============================\n" << std::endl;

    return 0;
}

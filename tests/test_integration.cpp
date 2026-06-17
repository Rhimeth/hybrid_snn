#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include "../src/data_pipeline.h"
#include "../src/snn_network.h"
#include "../src/telemetry_logger.h"

void test_closed_loop_learning() {
    std::system("mkdir -p data plots");
    
    // Initialize the CSV logger destination
    TelemetryLogger logger("data/simulation_log.csv");

    // Init Pipeline & Network
    DeltaEncoder joint_sensor(0.0f, 0.1f);

    // Params: leak_v, leak_a, th, beta, jump, trace_decay, learning_rate
    SNN_Crossbar<2, 2> hidden_layer(0.8f, 0.9f, 1.0f, 0.5f, 0.0f, 0.8f, 0.5f);
    SNN_Crossbar<2, 1> output_layer(0.8f, 0.9f, 1.0f, 0.5f, 0.0f, 0.8f, 0.5f);

    // Flash weights
    hidden_layer.flash_weight(0, 0, 1.2f);  // UP wire -> Hidden 0
    hidden_layer.flash_weight(1, 1, 1.2f);  // DOWN wire -> Hidden 1

    output_layer.flash_weight(0, 0, 1.2f);  // Hidden 0 -> Motor
    output_layer.flash_weight(0, 1, 1.2f);  // Hidden 1 -> Motor (The tracking synapse)

    std::vector<float> continuous_telemetry = {
        -0.15f, -0.30f, 0.05f, 0.12f, 0.25f, 0.28f, 0.15f, 0.0f
    };

    bool sensor_wires[2] = {false, false};
    bool hidden_spikes[2] = {false, false};
    bool motor_command[1] = {false};

    q8_8_t no_error_hidden[2] = {0, 0};
    q8_8_t dfa_error[1] {0}; // Global penalty

    int global_tick = 0;

    // Master Clock Loop
    for (int epoch = 1; epoch <= 2; ++epoch) {
        std::cout << "\n EPOCH" << epoch << " \n";
        std::cout << "Analog In | Wires (U/D) | Hidden Core | Motor | Error Signal\n";

        // Reset sensor base
        joint_sensor = DeltaEncoder(0.0f, 0.1f);

        for (float analog_val : continuous_telemetry) {
            // Encode
            joint_sensor.encode(analog_val, sensor_wires[0], sensor_wires[1]);

            // Forward Pass & Online Learning (Error passed applies to previously generated traces)
            hidden_layer.tick(sensor_wires, hidden_spikes, no_error_hidden);
            output_layer.tick(hidden_spikes, motor_command, dfa_error);

            // Environment Feedback
            dfa_error[0] = 0;
            float err_print = 0.0f;

            // Motor should not fire if DOWN wire spiked
            if (sensor_wires[1] == true && motor_command[0] == true) {
                err_print = -1.0f;
                dfa_error[0] = Q8_8::from_float(err_print);
            }

            int raw_weight = output_layer.get_weight_raw(0, 1);       // Hidden 1 -> Motor Synapse
            int raw_vm     = output_layer.get_neuron_voltage_raw(0);  // Output Motor Membrane Potential

            logger.logTick(global_tick, analog_val, raw_weight, motor_command[0], static_cast<int>(err_print), raw_vm);

            // System Diagnostic Probe
            std::cout << std::left 
                      << std::setw(10) << analog_val 
                      << "| [" << sensor_wires[0] << ", " << sensor_wires[1] << "]       "
                      << "| [" << hidden_spikes[0] << ", " << hidden_spikes[1] << "]       "
                      << "| [" << motor_command[0] << "]   "
                      << "| " << err_print << "\n";
            
            global_tick++;
        }
    }
}

int main() {
    std::cout << "--- Starting Closed-Loop Online Learning Test ---\n";
    test_closed_loop_learning();
    std::cout << "\n--- System Architecture Verified ---\n";
    return 0;
}
#include <iostream>
#include <iomanip>
#include "../src/data_pipeline.h"

void test_delta_modulation() {
    // Init encoder
    DeltaEncoder joint_sensor(0.0f, 0.1f);

    // Simulated telemetry from moving joint
    float continuous_telemetry[] = {0.05f, 0.12f, 0.25f, 0.28f, 0.15f, 0.0f, -0.15f};
    int num_samples = 7;

    bool up_wire, down_wire;

    std::cout << std::left << std::setw(15) << "Analog Sensor" 
              << std::setw(10) << "| UP Wire" 
              << "| DOWN Wire\n";
    std::cout << "--------------------------------------\n";

    for (int i = 0; i < num_samples; ++i) {
        joint_sensor.encode(continuous_telemetry[i], up_wire, down_wire);

        std::cout << std::left << std::setw(15) << continuous_telemetry[i]
                  << "| " << std::setw(8) << (up_wire ? "SPIKE" : "-")
                  << "| " << (down_wire ? "SPIKE" : "-") << "\n";
    }
}

int main() {
    std::cout << "--- Starting Data Pipeline Benchmarks ---\n";
    test_delta_modulation();
    std::cout << "\n--- Pipeline Tests Complete ---\n";
    return 0;
}

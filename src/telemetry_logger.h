#ifndef TELEMETRY_LOGGER_H
#define TELEMETRY_LOGGER_H

#include <fstream>
#include <iostream>
#include <string>

class TelemetryLogger {
private:
    std::ofstream file;

public:
    TelemetryLogger(const std::string& filename) {
        file.open(filename);
        if (file.is_open()) {
            // Write CSV Header
            file << "tick,analog_in,weight,motor,error,membrane_voltage\n";
        } else {
            std::cerr << "Error: Could not open telemetry log file: " << filename << "\n";
        }
    }

    ~TelemetryLogger() {
        if (file.is_open()) {
            file.close();
        }
    }

    void logTick(int tick, float analog_in, int raw_weight, int motor, int error, int raw_vm) {
        if (!file.is_open()) return;

        // Convert Q8.8 fixed-point to float (divide by 256.0)
        float float_weight = static_cast<float>(raw_weight) / 256.0f;
        float float_vm     = static_cast<float>(raw_vm) / 256.0f;

        file << tick << ","
             << analog_in << ","
             << float_weight << ","
             << motor << ","
             << error << ","
             << float_vm << "\n";
    }
};

#endif // TELEMETRY_LOGGER_H
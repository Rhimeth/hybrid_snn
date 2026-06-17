#ifndef DATA_PIPELINE_H
#define DATA_PIPELINE_H

class DeltaEncoder {
private:
    float baseline;
    float threshold;

public:
    // Init sensor value and threshold
    DeltaEncoder(float initial_val, float thresh) : baseline(initial_val), threshold(thresh) {}

    // Maps 1 continuous float into 2 binary spike wires
    void encode(float sensor_value, bool& spike_up, bool& spike_down) {
        spike_up = false;
        spike_down = false;

        float delta = sensor_value - baseline;

        if (delta >= threshold) {
            spike_up = true;
            baseline += threshold;
        } else if (delta <= -threshold) {
            spike_down = true;
            baseline -= threshold;
        }
    }
};

#endif
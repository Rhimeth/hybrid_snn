module lif_neuron #(
    parameter WIDTH     = 16,
    parameter THRESHOLD = 16'h2000,
    parameter LEAK      = -6554        // signed leak per cycle
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire signed [WIDTH-1:0] input_current,
    output reg                     spike_out
);
    reg signed [WIDTH-1:0] v_mem;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem     <= 0;
            spike_out <= 0;
        end else begin
            if (v_mem >= THRESHOLD) begin
                v_mem     <= 0;
                spike_out <= 1;
            end else begin
                v_mem     <= v_mem + input_current + LEAK;
                spike_out <= 0;
            end
        end
    end
endmodule
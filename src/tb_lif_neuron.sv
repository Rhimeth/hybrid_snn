module tb_lif_neuron();
    reg clk, rst_n;
    reg signed [15:0] input_current;
    wire spike;

    lif_neuron uut (
        .clk(clk), .rst_n(rst_n),
        .input_current(input_current),
        .spike_out(spike)
    );

    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end

    initial begin
        $dumpfile("tb_lif_neuron.vcd");
        $dumpvars(0, tb_lif_neuron);
        rst_n = 0;
        input_current = 0;
        #20 rst_n = 1;
        input_current = 16'h1000;
        #10000 $finish;
    end
endmodule
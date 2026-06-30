# src/jax_translation/snn_core.py
import jax
import jax.numpy as jnp

@jax.jit
def lif_eprop_step(state, input_spike_frame, W, alpha, threshold):
    v_t_minus_1, trace_t_minus_1 = state
    
    # LIF Membrane Update
    v_t = alpha * v_t_minus_1 + jnp.dot(input_spike_frame, W)
    
    # Heaviside Spike Generation
    spike_t = jnp.where(v_t >= threshold, 1.0, 0.0)
    v_t = jnp.where(spike_t == 1.0, 0.0, v_t) # Hard reset
    
    # Eligibility Trace Update
    pseudo_derivative = jnp.maximum(0.0, 1.0 - jnp.abs(v_t - threshold))
    trace_t = alpha * trace_t_minus_1 + jnp.outer(input_spike_frame, pseudo_derivative)
    
    return (v_t, trace_t), spike_t
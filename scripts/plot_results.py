import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Load simulation data
df = pd.read_csv('data/simulation_log.csv')

# --- CHART 1: SYNAPTIC WEIGHT CONVERGENCE VS ERROR ---
fig, ax1 = plt.subplots(figsize=(10, 5))

color = '#1f77b4'
ax1.set_xlabel('Simulation Ticks', fontweight='bold')
ax1.set_ylabel('Synaptic Weight (W)', color=color, fontweight='bold')
ax1.plot(df['tick'], df['weight'], color=color, linewidth=2.5, label='Down-Wire Weight')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Create a second axis for the error signals
ax2 = ax1.twinx()  
color = '#d62728'
ax2.set_ylabel('Global Error Signal (e)', color=color, fontweight='bold')
ax2.step(df['tick'], df['error'], color=color, linewidth=2, linestyle=':', where='mid', label='DFA Error')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Synaptic Weight Convergence Under Direct Feedback Alignment', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('plots/weight_convergence.png', dpi=300)
plt.close()

# --- CHART 2: LIF MEMBRANE VOLTAGE INTEGRATION ---
plt.figure(figsize=(10, 4))
plt.plot(df['tick'], df['membrane_voltage'], color='#2ca02c', linewidth=2, label='Membrane Voltage (Vm)')
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Firing Threshold (Vth)')

plt.title('Leaky Integrate-and-Fire (LIF) Micro-Dynamics', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Simulation Ticks', fontweight='bold')
plt.ylabel('Membrane Potential (V)', fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/membrane_voltage.png', dpi=300)
plt.close()

print("Graphs successfully compiled and saved to the plots/ folder!")
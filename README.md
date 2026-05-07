# Quantum Noise Modelling & Simulation
This project focuses on developing a accurate and high-speed model of quantum noise for simulating repetition code. The validated model allows futural simulation of other various quantum error correction (QEC) codes, including the surface code.

## Introduction

Accurate noise modeling is essential for developing practical quantum computers, but current approaches have significant limitations. Existing noise models either oversimplify quantum hardware behavior or require exponential computational resources, making them unsuitable for large-scale quantum error correction.

This project presents a universal noise model for superconducting quantum chips that overcomes these limitations through a **physics-based parameter decomposition methodology** built on the Stim library[1]. Our approach systematically decomposes platform-calibrated parameters (T1, T2, gate fidelities, readout errors) into four distinct physical mechanisms: **decoherence processes, gate infidelity, state preparation and measurement (SPAM) errors, and leakage propagation**. The model incorporates **Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization** to automatically fit parameters against experimental quantum error correction data, ensuring maximum fidelity to actual hardware behavior while enabling accurate and efficient simulation across different quantum platforms.

📄 Paper:https://arxiv.org/abs/2603.29439

## Key Advantages
🚀 Superior Accuracy: Outperforms all baseline methods across three error correlation types[2]. Compared to traditional models, our CMA-ES optimized approach achieves 19.5× reduction in time error rates, 9.3× reduction in space error rates, and 5.2× reduction in spacetime error rates. Significantly surpasses Code-capacity models, circuit-level models, phenomenological models, and Google's SD6/SI1000 approaches in all error categories.

⚡ Computational Efficiency: Features favorable polynomial scaling with system size, avoiding the exponential overhead associated with density-matrix-based approaches. Enable simulation of surface codes with distances up to 49 and circuit depths exceeding 10,000 QEC cycles. Supports large-scale systems (2000+ qubits) within practical memory constraints (<32 GB).

🌐 Cross-Platform Universality: Demonstrates consistent performance across IBM QPUs (Brisbane, Sherbrooke, Torino), China Mobile’s Wuyue QPU and QuantumCTeck’s Tianyan QPU without requiring any platform-specific parameter tuning or optimization.

🔬 Rigorous Validation: Extensively tested through single-round cross-platform experiments (5-21 qubits) and multi-round temporal studies (21 qubits over 30 QEC cycles). Uses CMA-ES optimization for automatic parameter fitting against experimental quantum error correction data.

## Performance Comparison
### Correlation Matrix Heat Map Comparison
The correlation matrix computation captures three types of error relationships in quantum error correction: timelike correlations, spacelike 
correlations, and spacetime correlations. This analysis is essential for validating the noise model's ability to reproduce realistic quantum hardware behavior. For detailed methodology, see [2].
<div align="center">
<img width="3493" height="3567" alt="experiment_correlation_matrix" src="https://github.com/user-attachments/assets/94b2bd62-864e-4632-aba9-910e454a5f4a" />
</i></div>
<div align="center"><i>Experiment correlation matrix heat map</i></div>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<div align="center">
<img width="3493" height="3567" alt="SI1000_p=0 01_correlation_matrix" src="https://github.com/user-attachments/assets/239bd599-2921-4460-a293-262fb5b360ad" />
</i></div>
<div align="center"><i>SI1000 noise model correlation matrix heat map, p = 0.01</i></div>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<div align="center">
<img width="3493" height="3567" alt="simulation_correlation_matrix" src="https://github.com/user-attachments/assets/c6f65553-974e-4793-a5a4-41b4d1479370" />
</i></div>
<div align="center"><i>Our universal noise model correlation matrix heat map</i></div>
<br>
<br>
<br>
<br>

### Error Rates Comparison
<div align="center">
<img width="1644" height="811" alt="image" src="https://github.com/user-attachments/assets/c8d34678-7fb4-45ee-a696-b18b77846ccc" />
</i></div>

<div align="center">Our universal noise model significantly outperforms all baseline methods, achieving 19.5×, 9.3×, and 5.2× improvements in time, space, and spacetime error rates respectively.</i></div>


## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/hesonghuan17-netizen/Superconducting-Cross-Platform-Quantum-Noise-Model.git
   cd Superconducting-Cross-Platform-Quantum-Noise-Model
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install the required dependencies: 
   ```
   pip install -r requirements.txt 
   ```
   For more details about the required dependencies, refer to the [requirements.txt](requirements.txt) file.


## Start

### Quantum Noise Simulation

#### 1. Configure Parameters

Edit the `config.py` file with your quantum system parameters:

```python
# config.py
NUM_QUBITS = 21          # Number of qubits for repetition code
NUM_ROUNDS = 30          # Number of measurement rounds
SHOTS_EXP = 8000         # Total experimental shots
SHOTS = 1                # Shots per noise scenario
```
#### 2. Prepare Parameter Files

The simulation requires optimized parameter files in JSON format containing platform-specific quantum hardware parameters. **The parameter file must match the settings in `config.py`** (e.g., `NUM_QUBITS = 21` requires 21 qubits worth of parameters).

**Use parameter files like this:**
```bash
{
   "spam_rates": [
       0.013427734375,          // State preparation and measurement error rates
       0.009752154356554221,
       0.0205078125,
       0.007721985833098121
   ],
   "t1_t2_values": [
       {
           "t1": 6.363399272084333e-05,      // T1: relaxation time (seconds)
           "t2": 1.9816307591254707e-05      // T2: dephasing time (seconds)
       },
       {
           "t1": 0.00024796670678882193,
           "t2": 0.00023600546182412152
       }
   ],
   "ecr_lengths": [5.333333333333332e-07,5.333333333333332e-07],  // Two-qubit gate durations (seconds)
   "sqg_length": [5.69e-08,5.69e-08],     // Single-qubit gate durations (seconds)
   "rd_length": [1.216e-06,1.216e-06],   // Readout operation durations (seconds)
   "ecr_fid": [0.9934625087568738, 0.9981116487192228],  // Two-qubit gate fidelities
   "sqg_fid": [0.9998731637079057, 0.9992618718889338],  // Single-qubit gate fidelities
   "lp": [6.884759180577696e-05, 0.001316455545806526], // Leakage probabilities
   "sp": [0.14833706303566527, 0.01125922715588417]      // Seepage probabilities
}
```
#### 3. Run Quantum Noise Simulation

Execute the main simulation with your parameter file:
```python
# simulation_example.py
import json
import numpy as np
from Simulation.Creation_and_Sampling.sampling import run_sampling
from config import NUM_QUBITS, NUM_ROUNDS, SHOTS, SHOTS_EXP

# Load optimized parameters from JSON file
with open('parameters.json', 'r') as f:
    params = json.load(f)

# Extract parameters
spam_rates = params['spam_rates']
t1_t2_values = [(item['t1'], item['t2']) for item in params['t1_t2_values']]
ecr_fid = params['ecr_fid']
sqg_fid = params['sqg_fid']
ecr_lengths = params['ecr_lengths']
sqg_length_list = params['sqg_length']
sqg_lengths = sqg_length_list[0] if sqg_length_list else 0  # Single-qubit gate duration is short and uniform across qubits, only one value needed instead of a list
rd_lengths = params['rd_length']
lp = params['lp']  # Leakage probabilities
sp = params['sp']  # Seepage probabilities

# Run simulation
results = run_sampling(
    shots=SHOTS,
    shots2=SHOTS_EXP,
    rounds=NUM_ROUNDS,
    num_qubits=NUM_QUBITS,
    lp=lp,
    sp=sp,
    spam_rates=spam_rates,
    sqg_fid=sqg_fid,
    ecr_fid=ecr_fid,
    t1_t2_values=t1_t2_values,
    ecr_lengths=ecr_lengths,
    rd_lengths=rd_lengths,
    sqg_lengths=sqg_lengths
)

print(f"Simulation completed! Generated {results.shape} detection events.")

# Save results
np.save('simulation_results.npy', results)
```
#### 4. Analyze Results
Compute correlation matrices:


```python
# analysis_example.py
import numpy as np
from Simulation.Analysis_and_Plotting.analysis import compute_correlation_matrix
from config import NUM_QUBITS, NUM_ROUNDS, SHOTS_EXP

# Load simulation results
results = np.load('simulation_results.npy')

# Compute correlation matrix
correlation_matrix = compute_correlation_matrix(
    data=results,
    num_qubits=NUM_QUBITS,
    rounds_=NUM_ROUNDS,
    shots=SHOTS_EXP
)

# Save correlation matrix
np.save('correlation_matrix_sim.npy', correlation_matrix)

print(f"Correlation matrix computed: {correlation_matrix.shape}")
```

#### 5. Visualize Results
Plot the correlation matrices:

```python
# Update Plot_mtx.py path to your correlation matrix
# Then run visualization
python Simulation/Analysis_and_Plotting/Plot_mtx.py
```


## Parameter Optimization

Our quantum noise model employs a two-stage CMA-ES optimization framework to calibrate noise parameters against experimental data.

### Optimization Architecture

The framework optimizes four physical noise mechanisms in parallel:

- **ECR Fidelity**: Two-qubit gate errors
- **Leakage Parameters**: LP (leakage) and SP (seepage) probabilities  
- **T1/T2 Times**: Decoherence time constants
- **Measurement Errors**: SPAM (State Preparation And Measurement) errors

### Two-Stage Workflow

#### Stage 1: Average Pre-optimization

**Configure and run average optimization:**

```python
# Edit main_averaged.py
TARGET_DATA_FOLDER = "correlation_matrix_brisbane_layout_1"
INITIAL_CONFIG_FILE = r"path/to/initial_parameters.json"

# Run optimization
python main_averaged.py
```

**Select optimizers when prompted:**
```python
🚀 Select optimizers to run
Please enter: all    # or specific ones like: ecr mea
```

#### Stage 2: Individual Fine-tuning

**Configure batch processing:**
```python
# Edit main.py  
TARGET_DATA_FOLDER = "correlation_matrix_experiment"  # Same folder

# Update initial parameter path to use Stage 1 results
```

Run batch optimization:
```bash

python main.py
```

### Configuration Parameters
CMA-ES Settings:

```python
# Stage 1: Comprehensive optimization
AVERAGE_CMA_CONFIG = {
    'ecr': {
        'popsize': 20,      # Population size
        'maxiter': 200,     # Max generations  
        'sigma': 0.01,      # Initial step size
    }
}

# Stage 2: Fine-tuning
INDIVIDUAL_CMA_CONFIG = {
    'ecr': {
        'popsize': 20,
        'maxiter': 100,     # Fewer iterations
        'sigma': 0.01,
    }
}
```

Parameter Bounds: 
```python

# ECR fidelity bounds
ecr_bounds = [0.9, 0.999]

# Leakage probability bounds  
lp_bounds = [1e-5, 5e-2]
sp_bounds = [1e-3, 5e-1]

# T1/T2 time bounds
t1_bounds = [1e-5, 6e-4]  # 10μs to 600μs
t2_t1_ratio = [0.1, 1.0]  # T2/T1 ratio

# Measurement error bounds
mea_bounds = [1e-4, 2e-1]

```
Output Structure: 
```bash
optimization_results/
└── {layout_name}/
    ├── average_correlation_matrix/           # Stage 1 results
    │   ├── Paramaeter_Loading/
    │   │   └── shared_parameters.json        # Use for Stage 2
    │   └── optimization_logs/
    │
    └── correlation_matrix_sample_*/          # Stage 2 results
        ├── Paramaeter_Loading/
        │   └── shared_parameters.json        # Final optimized params
        └── optimization_logs/

```

## Single-Round Repetition Code Optimization

The framework includes specialized optimization for single-round repetition codes, using the **quantum diamond norm** (total variation distance norm) as the objective function to minimize differences between simulated and experimental measurement distributions.

### Single-Round Code Features

**Dual SPAM Rate Parameters:**
- `spam_rates`: Measurement error rates
- `spam_rates_initial`: State preparation error rates

**Two Simulation Types:**
- **X-basis Repetition Code** (`sim_Repetition_X.py`): Detects logical X errors
- **Z-basis Repetition Code** (`sim_Repetition_Z.py`): Detects logical Z errors

### Quantum Diamond Norm Optimization

**Objective Function (total variation distance norm):**
```python
def calculate_l1_distance(sim_counts, exp_counts, total_shots_sim, total_shots_exp):
    """Calculate quantum diamond norm (L1 norm) between distributions"""
    all_states = set(sim_counts.keys()) | set(exp_counts.keys())
    
    l1_distance = 0.0
    for state in all_states:
        p_sim = sim_counts.get(state, 0) / total_shots_sim
        p_exp = exp_counts.get(state, 0) / total_shots_exp
        l1_distance += abs(p_sim - p_exp)
    
    return l1_distance
```
### Configuration and Execution
#### 1. Configure platforms in **`Optimization_CMA_single_rep.py`**:
```python

# Platform configurations
CHIP_CONFIGS = {
    'ibm_brisbane': {
        'param_file': 'path/to/optimized_parameters_ibm_brisbane.json',
        'exp_data_X_dir': 'path/to/experimental_X_results',
        'exp_data_Z_dir': 'path/to/experimental_Z_results'
    },
    'ibm_sherbrooke': {...},
    'Wuyue': {...}
}

# Optimization settings
NUM_QUBITS_LIST = [5, 9, 13, 17, 21]
TIMES = 10
SHOTS = 4096 * TIMES
```

#### 2. Run single-round optimization:
```bash
python Optimization_CMA_single_rep.py
```

#### 3. Dual simulation execution:
```python
# X-basis simulation
sim_result_array_X = sim.sim_Repetition_X.run_sampling(
    SHOTS_EXP, SHOTS, 0, num_qubits,
    lp, sp, spam_rates, spam_rates_initial,  # Note dual SPAM rates
    sqg_fid, ecr_fid, t1_t2_values, ecr_lengths, rd_lengths
)

# Z-basis simulation
sim_result_array_Z = sim.sim_Repetition_Z.run_sampling(
    SHOTS_EXP, SHOTS, 0, num_qubits,
    lp, sp, spam_rates, spam_rates_initial,  # Note dual SPAM rates
    sqg_fid, ecr_fid, t1_t2_values, ecr_lengths, rd_lengths
)

# Calculate combined L1 distance
total_l1 = l1_distance_X + l1_distance_Z

```

## Data

Research data: https://doi.org/10.5281/zenodo.16983512

## License
Apache License 2.0

## Reference
[1] C. Gidney, "Stim: a fast stabilizer circuit simulator," *Quantum*, vol. 5, p. 497, Jul. 2021. doi: [10.22331/q-2021-07-06-497](https://doi.org/10.22331/q-2021-07-06-497).  
[2] Google Quantum AI. "Exponential suppression of bit or phase errors with cyclic error correction." Nature 595, 383–387 (2021). doi: [10.1038/s41586-021-03588-y](https://doi.org/10.1038/s41586-021-03588-y)








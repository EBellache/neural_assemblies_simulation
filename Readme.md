# Morphogenic Spaces Framework for Neuronal Assemblies

## Overview

This framework implements a novel approach to understanding hippocampal neuronal assemblies using:
- **Tropical Mathematics** for piecewise-linear dynamics
- **Golay Error-Correcting Codes** for robust assembly encoding
- **E8 Exceptional Lie Group** geometry for state space representation
- **Sheaf Theory** and **Category Theory** for hierarchical organization

The framework is designed to test predictions against Buzsáki's HC-5 hippocampal dataset.

## Directory Structure

```
morphogenic_framework/
├── core/                      # Core mathematical modules
│   ├── tropical_math.py      # Tropical (max-plus) algebra operations
│   ├── oscillator.py         # Theta-gamma nested oscillations
│   └── assembly_detector.py  # Assembly detection using tropical clustering
│
├── triple_code/              # Error correction and geometry
│   ├── golay_code.py       # Golay [24,12,8] error correction
│   └── e8_lattice.py       # E8 lattice operations and projections
│
├── networks/                 # Neural network models
│   └── hippocampal_network.py  # CA3-CA1 circuit implementation
│
├── validation/               # Validation metrics
│   └── buzsaki_metrics.py  # Buzsáki-specific metrics and validation
│
├── datasets/                 # Data interfaces
│   └── hc5_interface.py    # HC-5 dataset loader and preprocessor
│
├── visualization/            # Visualization tools
│   └── visualize_results.py # Comprehensive plotting functions
│
├── test_morphogenic_framework.py  # Main test protocol
└── README.md                      # This file
```

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy scipy matplotlib seaborn
pip install h5py pandas scikit-learn

# Optional: JAX for acceleration
pip install jax jaxlib

# Optional: For actual HC-5 data
# Download from CRCNS.org after registration
```

### Setup

```bash
# Clone or extract the framework
cd morphogenic_framework

# Create necessary directories
mkdir -p hc5_data test_results figures cache

# Test installation
python -c "from core.tropical_math import test_tropical_operations; test_tropical_operations()"
```

## Quick Start

### 1. Test Individual Modules

```python
# Test tropical mathematics
from core.tropical_math import test_tropical_operations
test_tropical_operations()

# Test assembly detection
from core.assembly_detector import test_assembly_detector
test_assembly_detector()

# Test Golay codes
from triple_code.golay_code import test_golay_code
test_golay_code()

# Test E8 lattice
from triple_code.e8_lattice import test_e8_lattice
test_e8_lattice()
```

### 2. Run Complete Test Protocol

```python
from test_morphogenic_framework import main

# Run full protocol on mock data
results = main()
```

### 3. Visualize Results

```python
from visualization.visualize_results import visualize_test_results

# Generate comprehensive report
viz = visualize_test_results(
    results_file='./test_results/test_results.json',
    output_dir='./figures'
)
```

## Key Components

### Tropical Mathematics Module

Implements max-plus algebra for piecewise-linear dynamics:

```python
from core.tropical_math import tropical_inner_product, tropical_distance

# Tropical operations
a = np.array([1, 3, 2])
b = np.array([2, 1, 4])

# Inner product: max(a + b)
inner = tropical_inner_product(a, b)

# Distance: max(a-b) - min(a-b)
dist = tropical_distance(a, b)
```

### Assembly Detection

Detects neuronal assemblies using tropical clustering:

```python
from core.assembly_detector import AssemblyDetector

detector = AssemblyDetector(bin_size_ms=25, min_cells=5)

# Detect assemblies in spike trains
assemblies = detector.detect_assemblies_tropical(
    spike_trains,  # Dict[cell_id, spike_times]
    time_window=(0, 1.0)
)
```

### Golay Error Correction

Encodes assembly patterns with error correction:

```python
from triple_code.golay_code import GolayAssemblyEncoder

encoder = GolayAssemblyEncoder()

# Encode 12-bit pattern to 24-bit codeword
codeword = encoder.encode_assembly(pattern)

# Decode with error correction (up to 3 errors)
corrected, n_errors = encoder.decode_assembly(noisy_codeword)
```

### E8 Lattice Projection

Maps assembly states to E8 geometry:

```python
from triple_code.e8_lattice import E8Lattice, assembly_to_e8

lattice = E8Lattice()

# Project to nearest E8 point
e8_coords = assembly_to_e8(assembly_pattern, lattice)

# Compute Casimir invariants
casimirs = lattice.compute_casimir_invariants(e8_coords)
```

### Hippocampal Network

Simulates CA3-CA1 circuit dynamics:

```python
from networks.hippocampal_network import HippocampalNetwork

# Create network
network = HippocampalNetwork(
    n_ca3=500,
    n_ca1=500,
    n_interneurons=100
)

# Simulate theta cycle
results = network.simulate_theta_cycle(external_input)
```

## Testing Protocol

The main test protocol validates:

1. **Assembly Detection**: Using tropical correlation matrices
2. **Error Correction**: Golay code performance under noise
3. **E8 Geometry**: Projection accuracy and trajectory curvature
4. **Oscillations**: Theta-gamma phase-amplitude coupling
5. **Replay**: Sequence replay during sharp-wave ripples
6. **Information Content**: Entropy and synchrony metrics
7. **Validation**: Against Buzsáki's empirical criteria

### Running Tests

```python
from test_morphogenic_framework import MorphogenicTestProtocol

# Initialize protocol
protocol = MorphogenicTestProtocol(
    data_path='./hc5_data',
    output_dir='./test_results'
)

# Run on specific sessions
results = protocol.run_full_protocol(
    session_ids=['session1', 'session2'],
    max_sessions=5
)

# Results include:
# - Assembly compression ratio
# - Tropical correlation strength
# - Golay error correction rate
# - E8 projection error
# - Theta-gamma PAC
# - Ripple detection rate
# - Replay fidelity
# - Information content
```

## Validation Criteria

The framework validates against Buzsáki's findings:

| Metric | Expected Range | Description |
|--------|---------------|-------------|
| Assembly Compression | > 2.0 | Data compression via assemblies |
| Theta-Gamma PAC | 0.2-0.6 | Phase-amplitude coupling strength |
| Ripple Rate | 0.1-2.0 Hz | Sharp-wave ripple frequency |
| Replay Fidelity | > 0.5 | Sequence replay accuracy |
| Mean Firing Rate | 1-5 Hz | Pyramidal cell firing |
| Synchrony Index | > 1.0 | Above-chance synchronization |
| Information Content | > 1 bit | Population code entropy |

## Key Predictions

1. **Assemblies form tropical polytopes** in activity space
2. **Golay codes protect against ~12% spike failures**
3. **E8 coordinates show catastrophes at assembly transitions**
4. **Replay compresses time by factor ~20**
5. **Information peaks at gamma troughs**

## Extending the Framework

### Adding New Metrics

```python
# In validation/custom_metrics.py
def compute_custom_metric(spike_trains, assemblies):
    # Your metric computation
    return metric_value

# Register in test protocol
protocol.custom_metrics['my_metric'] = compute_custom_metric
```

### Custom Assembly Detection

```python
from core.assembly_detector import AssemblyDetector

class CustomDetector(AssemblyDetector):
    def detect_assemblies_custom(self, spike_trains, time_window):
        # Your detection algorithm
        return assemblies
```

### New Visualization

```python
from visualization.visualize_results import MorphogenicVisualizer

class CustomVisualizer(MorphogenicVisualizer):
    def plot_custom_analysis(self, ax, results):
        # Your visualization
        pass
```

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**
   - Reduce `max_sessions` in test protocol
   - Use data chunking in HC5DataLoader

2. **JAX not available**
   - Framework falls back to NumPy automatically
   - Performance will be slower but functional

3. **HC-5 data not found**
   - Framework generates mock data automatically
   - Download real data from CRCNS.org

4. **Convergence issues in assembly detection**
   - Adjust `bin_size_ms` parameter (default: 25ms)
   - Increase `min_cells` threshold

## Citation

If you use this framework, please cite:

```bibtex
@article{morphogenic2025,
  title={Morphogenic Spaces Framework for Neuronal Assemblies},
  author={Based on manuscript by A. Bellachehab},
  journal={In preparation},
  year={2025}
}

@book{buzsaki2006,
  title={Rhythms of the Brain},
  author={Buzsáki, György},
  publisher={Oxford University Press},
  year={2006}
}
```

## References

- Buzsáki, G. (2006). Rhythms of the Brain. Oxford University Press.
- Conway, J. H., & Sloane, N. J. A. (1999). Sphere packings, lattices and groups.
- Golay, M. J. E. (1949). Notes on digital coding. Proc. IRE, 37, 657.
- Souriau, J. M. (1970). Structure des systèmes dynamiques.
- Thurston, W. P. (1997). Three-dimensional geometry and topology.

## License

This framework is provided for academic research purposes.
See LICENSE file for details.

## Support

For questions or issues:
- Check the test functions in each module
- Review the docstrings for detailed parameter descriptions
- Ensure all dependencies are correctly installed

## Acknowledgments

- Allen Institute and IBL for open datasets
- CRCNS.org for hosting HC-5 data
- Buzsáki Lab for hippocampal research

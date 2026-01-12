# FluxGen R&D

This repository contains the complete Python implementation for the **FluxGen Sustainable Technologies R&D**. It includes mathematical models, spatial algorithms, and simulation logic for all four problem statements.

## 📂 Project Structure

```text
fluxgen/
│
├── main.py                 # Entry point: Runs all 4 simulations sequentially
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── src/
    ├── geometry_p1.py      # P1: Reservoir Volume Estimation (Kriging Interpolation)
    ├── spectral_p2.py      # P2: Spectral Validation Logic (False Positive Detection)
    ├── watershed_p3.py     # P3: Watershed Digital Twin (Delay Modeling via Convolution)
    └── groundwater_p4.py   # P4: Spatial Groundwater Dynamics (Cell-to-Cell Interaction)


# RA_Work

This repository contains research assistant work projects, including:

## Projects

### EyeTrack
- **Scenario.py**: Eye tracking scenario implementation with HDF5 data storage
- **analyze_data.py**: Comprehensive data analysis and visualization script
- **quick_view.py**: Simple script to quickly explore HDF5 data

### Pump
- **Pump-Control-Simulation**: Pump control simulation project
  - `Filter_Loc_P-wave.py`: Filter location P-wave analysis
  - `README.md`: Project documentation

## Getting Started

Each project directory contains its own documentation and implementation files. Please refer to individual project README files for specific setup and usage instructions.

### Eye Tracking Data Analysis

The EyeTrack project uses HDF5 format for efficient data storage and analysis:

1. **Data Collection**: Run `Scenario.py` to collect eye tracking data
2. **Quick View**: Use `quick_view.py <filename.h5>` to explore data
3. **Full Analysis**: Use `analyze_data.py <filename.h5>` for comprehensive analysis
4. **AI Preparation**: Export data in various formats for machine learning

**Example Usage:**
```bash
# Collect data
python EyeTrack/Scenario.py

# Quick data exploration
python EyeTrack/quick_view.py eye_tracking_data_20241201_143022.h5

# Full analysis with plots
python EyeTrack/analyze_data.py eye_tracking_data_20241201_143022.h5

# Export for AI/ML
python EyeTrack/analyze_data.py eye_tracking_data_20241201_143022.h5 --export hdf5
```

## Structure

```
RA_Work/
├── EyeTrack/
│   ├── Scenario.py
│   ├── analyze_data.py
│   └── quick_view.py
└── Pump/
    └── Pump-Control-Simulation/
        ├── Filter_Loc_P-wave.py
        └── README.md
```

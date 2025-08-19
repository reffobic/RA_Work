# RA_Work

This repository contains research assistant work projects, including:

## Projects

### EyeTrack
- **Scenario.py**: Eye tracking scenario implementation with HDF5 data storage
- **ScenarioCSV.py**: Eye tracking scenario implementation with CSV data storage
- **analyze_data.py**: Comprehensive data analysis and visualization script
- **quick_view.py**: Simple script to quickly explore HDF5 data

### GSR Sensor
- **sketch_aug11a.ino**: Arduino sketch for GSR sensor with dynamic baseline and phasic threshold detection



## Getting Started

Each project directory contains its own documentation and implementation files. Please refer to individual project README files for specific setup and usage instructions.

### Eye Tracking Data Analysis

The EyeTrack project supports both HDF5 and CSV formats for data storage and analysis:

#### **HDF5 Format (Recommended for Research)**
- **Efficient storage**: 70-90% compression
- **Fast queries**: Optimized for large datasets
- **AI/ML ready**: Direct integration with pandas, numpy

1. **Data Collection**: Run `Scenario.py` to collect eye tracking data
2. **Quick View**: Use `quick_view.py <filename.h5>` to explore data
3. **Full Analysis**: Use `analyze_data.py <filename.h5>` for comprehensive analysis
4. **AI Preparation**: Export data in various formats for machine learning

**Example Usage (HDF5):**
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

#### **CSV Format (Simple & Compatible)**
- **Human readable**: Easy to open in Excel, Google Sheets
- **Universal compatibility**: Works with any data analysis tool
- **Simple structure**: Standard CSV format

**Example Usage (CSV):**
```bash
# Collect data in CSV format
python EyeTrack/ScenarioCSV.py

# Open CSV file in Excel, Google Sheets, or any CSV viewer
# Use pandas for analysis: pd.read_csv('eye_tracking_data_20241201_143022.csv')
```

### GSR Sensor Analysis

The GSR sensor project provides physiological stress detection:

1. **Hardware Setup**: Connect GSR sensor to Arduino A0 pin
2. **Data Collection**: Upload `sketch_aug11a.ino` to Arduino
3. **Stress Detection**: Monitor serial output for stress events
4. **Parameters**: Adjust `PHASIC_THRESHOLD` and `REFRACTORY_PERIOD` for sensitivity

**Features:**
- Dynamic baseline tracking
- Phasic threshold detection
- Kalman filtering for noise reduction
- Refractory period to prevent false positives

## Structure

```
RA_Work/
├── EyeTrack/
│   ├── Scenario.py
│   ├── ScenarioCSV.py
│   ├── analyze_data.py
│   └── quick_view.py
└── GSRsensor/
    └── sketch_aug11a.ino
```

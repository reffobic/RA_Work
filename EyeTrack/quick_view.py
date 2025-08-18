#!/usr/bin/env python3
"""
Quick View Script for HDF5 Eye Tracking Data
Simple script to quickly explore your data
"""

import h5py
import pandas as pd
import sys

def quick_view(filename):
    """Quick overview of HDF5 eye tracking data"""
    print(f"📊 Quick View: {filename}")
    print("=" * 50)
    
    with h5py.File(filename, 'r') as hf:
        # Show metadata
        if 'metadata' in hf:
            print("📋 METADATA:")
            for key, value in hf['metadata'].attrs.items():
                print(f"   {key}: {value}")
            print()
        
        # Show datasets
        print("📈 DATASETS:")
        total_points = 0
        for dataset_name in hf.keys():
            if dataset_name != 'metadata':
                dataset = hf[dataset_name]
                points = dataset.shape[0]
                total_points = max(total_points, points)
                print(f"   {dataset_name}: {points} points")
        
        print(f"\n📊 Total data points: {total_points}")
        
        # Load a sample for quick stats
        if total_points > 0:
            sample_size = min(1000, total_points)
            sample_data = {}
            
            for dataset_name in hf.keys():
                if dataset_name != 'metadata':
                    dataset = hf[dataset_name]
                    sample_data[dataset_name] = dataset[:sample_size]
            
            df_sample = pd.DataFrame(sample_data)
            
            print(f"\n🔍 SAMPLE STATISTICS (first {sample_size} points):")
            if 'avg_pupil_diam' in df_sample.columns:
                print(f"   Average pupil diameter: {df_sample['avg_pupil_diam'].mean():.2f}")
            if 'gaze_valid' in df_sample.columns:
                valid_rate = (df_sample['gaze_valid'].sum() / len(df_sample)) * 100
                print(f"   Gaze validity rate: {valid_rate:.1f}%")
            if 'focus_actor_dist' in df_sample.columns:
                print(f"   Focus distance range: {df_sample['focus_actor_dist'].min():.1f} - {df_sample['focus_actor_dist'].max():.1f}")

def load_specific_data(filename, columns=None, start_idx=0, end_idx=None):
    """Load specific columns or time range from HDF5 file"""
    print(f"\n📥 Loading specific data from {filename}")
    
    with h5py.File(filename, 'r') as hf:
        if columns is None:
            columns = [key for key in hf.keys() if key != 'metadata']
        
        data = {}
        for col in columns:
            if col in hf:
                if end_idx is None:
                    data[col] = hf[col][start_idx:]
                else:
                    data[col] = hf[col][start_idx:end_idx]
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} data points with columns: {list(df.columns)}")
        return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_view.py <hdf5_filename>")
        print("Example: python quick_view.py eye_tracking_data_20241201_143022.h5")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        quick_view(filename)
        
        # Example: Load specific data
        print("\n" + "="*50)
        print("💡 EXAMPLE: Load first 100 points of pupil data")
        df_sample = load_specific_data(filename, 
                                     columns=['timestamp', 'left_pupil_diam', 'right_pupil_diam', 'gaze_valid'],
                                     start_idx=0, 
                                     end_idx=100)
        
        print("\nFirst 5 rows:")
        print(df_sample.head())
        
    except FileNotFoundError:
        print(f"❌ Error: File {filename} not found")
    except Exception as e:
        print(f"❌ Error: {e}")

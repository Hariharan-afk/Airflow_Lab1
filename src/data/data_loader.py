"""
Data Loader Module for Gas Sensor Array Drift Dataset
Handles LibSVM format parsing: class feature_id:value feature_id:value ...
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List
import yaml


def load_config():
    """Load configuration from YAML file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_libsvm_line(line: str, num_features: int = 128) -> Tuple[int, np.ndarray]:
    """
    Parse a single line in LibSVM format.
    
    Format: class feature_id:value feature_id:value ...
    Example: 1 1:15596.162100 2:1.868245 3:2.371604 ...
    
    Args:
        line: String line in LibSVM format
        num_features: Number of features expected (128)
    
    Returns:
        Tuple of (class_label, feature_vector)
    """
    parts = line.strip().split()
    
    # First element is the class label
    class_label = int(parts[0])
    
    # Initialize feature vector with zeros
    features = np.zeros(num_features)
    
    # Parse feature_id:value pairs
    for part in parts[1:]:
        if ':' in part:
            feature_id, value = part.split(':')
            feature_idx = int(feature_id) - 1  # Convert to 0-indexed
            features[feature_idx] = float(value)
    
    return class_label, features


def load_batch_data(batch_id: int, data_dir: str = None) -> pd.DataFrame:
    """
    Load a single batch file and convert to DataFrame.
    
    Args:
        batch_id: Batch number (1-10)
        data_dir: Directory containing .dat files
    
    Returns:
        DataFrame with features and labels
    """
    config = load_config()
    
    if data_dir is None:
        data_dir = config['paths']['raw_data']
    
    file_path = os.path.join(data_dir, f"batch{batch_id}.dat")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Batch file not found: {file_path}")
    
    print(f"📄 Loading batch {batch_id} from {file_path}")
    
    labels = []
    features_list = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                label, features = parse_libsvm_line(line, config['data']['num_features'])
                labels.append(label)
                features_list.append(features)
            except Exception as e:
                print(f"⚠️ Error parsing line {line_num}: {e}")
                continue
    
    # Create DataFrame
    feature_cols = [f"feature_{i+1}" for i in range(config['data']['num_features'])]
    df = pd.DataFrame(features_list, columns=feature_cols)
    df['class'] = labels
    df['batch_id'] = batch_id
    
    print(f"✅ Loaded {len(df)} samples from batch {batch_id}")
    print(f"   Class distribution: {df['class'].value_counts().sort_index().to_dict()}")
    
    return df


def load_multiple_batches(batch_ids: List[int], data_dir: str = None) -> pd.DataFrame:
    """
    Load multiple batches and combine them.
    
    Args:
        batch_ids: List of batch numbers to load
        data_dir: Directory containing .dat files
    
    Returns:
        Combined DataFrame
    """
    all_batches = []
    
    for batch_id in batch_ids:
        df = load_batch_data(batch_id, data_dir)
        all_batches.append(df)
    
    combined_df = pd.concat(all_batches, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"✅ Combined {len(batch_ids)} batches: {batch_ids}")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   Features: {combined_df.shape[1] - 2}")  # Exclude 'class' and 'batch_id'
    print(f"   Class distribution:")
    for cls, count in combined_df['class'].value_counts().sort_index().items():
        class_name = load_config()['data']['classes'][cls]
        print(f"      {cls} ({class_name}): {count}")
    print(f"{'='*60}\n")
    
    return combined_df


def save_processed_data(df: pd.DataFrame, filename: str):
    """Save processed DataFrame to CSV"""
    config = load_config()
    output_dir = config['paths']['processed_data']
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved processed data to: {output_path}")


if __name__ == "__main__":
    # Test the data loader
    print("🧪 Testing Data Loader Module\n")
    
    # Test loading single batch
    batch1 = load_batch_data(1)
    print(f"\nBatch 1 shape: {batch1.shape}")
    print(f"First row class: {batch1['class'].iloc[0]}")
    print(f"First 5 features: {batch1.iloc[0, :5].values}")
    
    # Test loading training batches
    config = load_config()
    train_batches = config['data']['train_batches']
    train_data = load_multiple_batches(train_batches)
    
    print(f"\n✅ Data loader module working correctly!")
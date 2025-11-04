"""
Preprocessing Module for Gas Sensor Array Drift Dataset
Handles data validation, cleaning, and feature scaling
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
import pickle


def load_config():
    """Load configuration from YAML file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    print("🔍 Validating data...")
    
    initial_rows = len(df)
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"⚠️ Found {missing_values} missing values")
        df = df.dropna()
        print(f"   Dropped rows with missing values")
    else:
        print(f"✅ No missing values found")
    
    # Check for infinite values
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    inf_mask = np.isinf(df[feature_cols]).any(axis=1)
    if inf_mask.sum() > 0:
        print(f"⚠️ Found {inf_mask.sum()} rows with infinite values")
        df = df[~inf_mask]
        print(f"   Dropped rows with infinite values")
    else:
        print(f"✅ No infinite values found")
    
    # Check class labels are valid (1-6)
    valid_classes = set(range(1, 7))
    invalid_classes = set(df['class'].unique()) - valid_classes
    if invalid_classes:
        print(f"⚠️ Found invalid class labels: {invalid_classes}")
        df = df[df['class'].isin(valid_classes)]
        print(f"   Removed rows with invalid classes")
    else:
        print(f"✅ All class labels are valid")
    
    final_rows = len(df)
    if final_rows < initial_rows:
        print(f"📊 Data validation complete: {initial_rows} → {final_rows} rows")
    else:
        print(f"✅ Data validation complete: {final_rows} rows")
    
    return df


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for features.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with feature statistics
    """
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    stats = pd.DataFrame({
        'mean': df[feature_cols].mean(),
        'std': df[feature_cols].std(),
        'min': df[feature_cols].min(),
        'max': df[feature_cols].max(),
        'median': df[feature_cols].median()
    })
    
    return stats


def scale_features(train_df: pd.DataFrame, 
                   test_df: pd.DataFrame = None,
                   scaler_path: str = None) -> tuple:
    """
    Scale features to [-1, 1] range as per dataset documentation.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (optional)
        scaler_path: Path to save/load scaler
    
    Returns:
        Tuple of (scaled_train_df, scaled_test_df, scaler)
    """
    config = load_config()
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    
    print("⚙️ Scaling features to [-1, 1] range...")
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=tuple(config['model']['scaler_range']))
    
    # Fit on training data
    train_scaled = train_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    print(f"✅ Scaled {len(feature_cols)} features")
    print(f"   Sample feature ranges after scaling:")
    for i in range(min(3, len(feature_cols))):
        col = feature_cols[i]
        print(f"      {col}: [{train_scaled[col].min():.3f}, {train_scaled[col].max():.3f}]")
    
    # Transform test data if provided
    test_scaled = None
    if test_df is not None:
        test_scaled = test_df.copy()
        test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
        print(f"✅ Applied scaling to test data ({len(test_df)} samples)")
    
    # Save scaler if path provided
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"💾 Saved scaler to: {scaler_path}")
    
    return train_scaled, test_scaled, scaler


def prepare_train_test_split(train_batch_ids: list, test_batch_ids: list):
    """
    Prepare train and test datasets from batch IDs.
    
    Args:
        train_batch_ids: List of batch IDs for training
        test_batch_ids: List of batch IDs for testing
    
    Returns:
        Tuple of (train_df, test_df)
    """
    from .data_loader import load_multiple_batches
    
    print("="*60)
    print("📦 PREPARING TRAIN-TEST SPLIT")
    print("="*60)
    
    # Load training data
    print(f"\n📚 Loading training batches: {train_batch_ids}")
    train_df = load_multiple_batches(train_batch_ids)
    train_df = validate_data(train_df)
    
    # Load test data
    print(f"\n🧪 Loading test batches: {test_batch_ids}")
    test_df = load_multiple_batches(test_batch_ids)
    test_df = validate_data(test_df)
    
    print("\n" + "="*60)
    print("✅ TRAIN-TEST SPLIT COMPLETE")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    print("="*60 + "\n")
    
    return train_df, test_df


def preprocess_pipeline(save_intermediate: bool = True):
    """
    Complete preprocessing pipeline.
    
    Args:
        save_intermediate: Whether to save intermediate processed files
    
    Returns:
        Tuple of (train_scaled, test_scaled, scaler)
    """
    config = load_config()
    
    print("\n" + "="*60)
    print("🚀 STARTING PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load and validate data
    train_df, test_df = prepare_train_test_split(
        config['data']['train_batches'],
        config['data']['test_batches']
    )
    
    # Step 2: Save raw processed data
    if save_intermediate:
        processed_dir = config['paths']['processed_data']
        os.makedirs(processed_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(processed_dir, 'train_raw.csv'), index=False)
        test_df.to_csv(os.path.join(processed_dir, 'test_raw.csv'), index=False)
        print(f"💾 Saved raw processed data to {processed_dir}/")
    
    # Step 3: Feature statistics
    print("\n📊 Training data feature statistics:")
    train_stats = get_feature_statistics(train_df)
    print(train_stats.head())
    
    # Step 4: Scale features
    print("\n")
    scaler_path = os.path.join(config['paths']['models'], 'scaler.pkl')
    train_scaled, test_scaled, scaler = scale_features(train_df, test_df, scaler_path)
    
    # Step 5: Save scaled data
    if save_intermediate:
        train_scaled.to_csv(os.path.join(processed_dir, 'train_scaled.csv'), index=False)
        test_scaled.to_csv(os.path.join(processed_dir, 'test_scaled.csv'), index=False)
        print(f"💾 Saved scaled data to {processed_dir}/")
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return train_scaled, test_scaled, scaler


if __name__ == "__main__":
    # Run preprocessing pipeline
    train_scaled, test_scaled, scaler = preprocess_pipeline(save_intermediate=True)
    
    print(f"✅ Preprocessing complete!")
    print(f"   Train shape: {train_scaled.shape}")
    print(f"   Test shape: {test_scaled.shape}")
"""
Model Training Module for Gas Sensor Drift Detection
Implements Random Forest classifier with proper validation
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import yaml
import json
from datetime import datetime


def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_random_forest(train_df: pd.DataFrame, 
                        model_params: dict = None) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    
    Args:
        train_df: Training DataFrame with scaled features
        model_params: Model hyperparameters (optional)
    
    Returns:
        Trained RandomForestClassifier
    """
    config = load_config()
    
    if model_params is None:
        model_params = config['model']['params']
    
    print("="*60)
    print("🌲 TRAINING RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Separate features and labels
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    X_train = train_df[feature_cols].values
    y_train = train_df['class'].values
    
    print(f"\n📊 Training Data:")
    print(f"   Samples: {len(X_train)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {np.unique(y_train)}")
    print(f"\n   Class distribution:")
    for cls in sorted(np.unique(y_train)):
        count = np.sum(y_train == cls)
        percentage = count / len(y_train) * 100
        class_name = config['data']['classes'][int(cls)]
        print(f"      {cls} ({class_name}): {count} ({percentage:.1f}%)")
    
    # Initialize model
    print(f"\n⚙️ Model Parameters:")
    for param, value in model_params.items():
        print(f"   {param}: {value}")
    
    model = RandomForestClassifier(**model_params)
    
    # Train model
    print(f"\n🚀 Training model...")
    model.fit(X_train, y_train)
    
    # Training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"✅ Training complete!")
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    
    return model


def cross_validate_model(train_df: pd.DataFrame, 
                         model_params: dict = None,
                         n_folds: int = 10) -> dict:
    """
    Perform cross-validation on training data.
    
    Args:
        train_df: Training DataFrame
        model_params: Model hyperparameters
        n_folds: Number of folds for cross-validation
    
    Returns:
        Dictionary with CV results
    """
    config = load_config()
    
    if model_params is None:
        model_params = config['model']['params']
    
    print("\n" + "="*60)
    print(f"🔄 PERFORMING {n_folds}-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Prepare data
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    X_train = train_df[feature_cols].values
    y_train = train_df['class'].values
    
    # Initialize model and cross-validator
    model = RandomForestClassifier(**model_params)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Perform cross-validation
    print(f"🔄 Running {n_folds}-fold stratified cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Calculate statistics
    cv_results = {
        'mean_accuracy': float(cv_scores.mean()),
        'std_accuracy': float(cv_scores.std()),
        'min_accuracy': float(cv_scores.min()),
        'max_accuracy': float(cv_scores.max()),
        'fold_scores': cv_scores.tolist()
    }
    
    print(f"\n✅ Cross-Validation Results:")
    print(f"   Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"   Min Accuracy:  {cv_results['min_accuracy']:.4f}")
    print(f"   Max Accuracy:  {cv_results['max_accuracy']:.4f}")
    print(f"\n   Fold Scores:")
    for i, score in enumerate(cv_results['fold_scores'], 1):
        print(f"      Fold {i:2d}: {score:.4f}")
    
    return cv_results


def get_feature_importance(model: RandomForestClassifier, 
                          feature_cols: list,
                          top_n: int = 20) -> pd.DataFrame:
    """
    Extract and rank feature importances.
    
    Args:
        model: Trained RandomForestClassifier
        feature_cols: List of feature column names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importances
    """
    print(f"\n📊 Extracting Feature Importances (Top {top_n})...")
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance_df.head(top_n)
    
    print(f"\n   Top {top_n} Most Important Features:")
    for idx, row in top_features.iterrows():
        print(f"      {row['feature']}: {row['importance']:.6f}")
    
    return feature_importance_df


def save_model(model: RandomForestClassifier, 
               metadata: dict,
               model_name: str = None):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained model
        metadata: Dictionary with training metadata
        model_name: Name for the saved model file
    """
    config = load_config()
    model_dir = config['paths']['models']
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"random_forest_{timestamp}"
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Saved model to: {model_path}")
    
    # Save metadata
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {metadata_path}")
    
    # Save as 'latest' for easy access
    latest_model_path = os.path.join(model_dir, "latest_model.pkl")
    latest_metadata_path = os.path.join(model_dir, "latest_model_metadata.json")
    
    with open(latest_model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(latest_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved as 'latest_model' for easy access")
    
    return model_path, metadata_path


def load_model(model_path: str = None):
    """
    Load a trained model.
    
    Args:
        model_path: Path to model file (defaults to latest)
    
    Returns:
        Tuple of (model, metadata)
    """
    config = load_config()
    model_dir = config['paths']['models']
    
    if model_path is None:
        model_path = os.path.join(model_dir, "latest_model.pkl")
        metadata_path = os.path.join(model_dir, "latest_model_metadata.json")
    else:
        metadata_path = model_path.replace('.pkl', '_metadata.json')
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"📂 Loaded model from: {model_path}")
    
    # Load metadata
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"📂 Loaded metadata from: {metadata_path}")
    
    return model, metadata


def train_pipeline(train_df: pd.DataFrame, 
                   perform_cv: bool = True,
                   save_model_flag: bool = True) -> tuple:
    """
    Complete model training pipeline.
    
    Args:
        train_df: Training DataFrame with scaled features
        perform_cv: Whether to perform cross-validation
        save_model_flag: Whether to save the model
    
    Returns:
        Tuple of (model, metadata)
    """
    config = load_config()
    
    print("\n" + "="*60)
    print("🚀 STARTING MODEL TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Cross-validation (optional)
    cv_results = None
    if perform_cv:
        cv_results = cross_validate_model(train_df, n_folds=10)
    
    # Step 2: Train final model on all training data
    model = train_random_forest(train_df)
    
    # Step 3: Extract feature importances
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    feature_importance_df = get_feature_importance(
        model, 
        feature_cols, 
        top_n=config['drift']['feature_drift']['top_n_features']
    )
    
    # Step 4: Prepare metadata
    metadata = {
        'model_type': 'RandomForest',
        'training_date': datetime.now().isoformat(),
        'train_batches': config['data']['train_batches'],
        'n_samples': len(train_df),
        'n_features': len(feature_cols),
        'model_params': config['model']['params'],
        'training_accuracy': float(model.score(
            train_df[feature_cols].values, 
            train_df['class'].values
        )),
        'cv_results': cv_results,
        'top_features': feature_importance_df.head(20).to_dict('records')
    }
    
    # Step 5: Save model
    if save_model_flag:
        model_path, metadata_path = save_model(model, metadata)
        metadata['model_path'] = model_path
        metadata['metadata_path'] = metadata_path
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING PIPELINE COMPLETE")
    print("="*60 + "\n")
    
    return model, metadata


if __name__ == "__main__":
    # Test training pipeline
    print("🧪 Testing Model Training Module\n")
    
    # Load preprocessed data
    from src.data.preprocess import preprocess_pipeline
    train_scaled, test_scaled, scaler = preprocess_pipeline(save_intermediate=False)
    
    # Train model
    model, metadata = train_pipeline(train_scaled, perform_cv=True, save_model_flag=True)
    
    print("\n✅ Model training module working correctly!")
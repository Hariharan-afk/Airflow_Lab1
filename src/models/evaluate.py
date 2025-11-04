"""
Model Evaluation Module for Gas Sensor Drift Detection
Provides comprehensive evaluation metrics and visualizations
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
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


def evaluate_model(model, test_df: pd.DataFrame, batch_id: int = None) -> dict:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_df: Test DataFrame with scaled features
        batch_id: Batch ID being evaluated (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    config = load_config()
    
    # Prepare data
    feature_cols = [col for col in test_df.columns if col.startswith('feature_')]
    X_test = test_df[feature_cols].values
    y_test = test_df['class'].values
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'batch_id': batch_id,
        'n_samples': len(y_test),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
    }
    
    # Per-class metrics
    per_class_metrics = {}
    for cls in sorted(np.unique(y_test)):
        cls_int = int(cls)
        class_name = config['data']['classes'][cls_int]
        
        # Binary classification for this class
        y_true_binary = (y_test == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        per_class_metrics[cls_int] = {
            'class_name': class_name,
            'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            'support': int(np.sum(y_test == cls))
        }
    
    metrics['per_class'] = per_class_metrics
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Prediction confidence statistics
    max_probas = y_pred_proba.max(axis=1)
    metrics['confidence'] = {
        'mean': float(max_probas.mean()),
        'std': float(max_probas.std()),
        'min': float(max_probas.min()),
        'max': float(max_probas.max()),
        'median': float(np.median(max_probas))
    }
    
    # Misclassification analysis
    misclassified = y_test != y_pred
    metrics['misclassification'] = {
        'count': int(misclassified.sum()),
        'rate': float(misclassified.mean()),
        'low_confidence_threshold': 0.6,
        'low_confidence_count': int((max_probas < 0.6).sum())
    }
    
    return metrics


def print_evaluation_results(metrics: dict, batch_id: int = None):
    """
    Print evaluation results in a formatted manner.
    
    Args:
        metrics: Dictionary with evaluation metrics
        batch_id: Batch ID being evaluated
    """
    config = load_config()
    
    print("\n" + "="*60)
    if batch_id:
        print(f"📊 EVALUATION RESULTS - BATCH {batch_id}")
    else:
        print("📊 EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Samples:         {metrics['n_samples']}")
    print(f"   Accuracy:        {metrics['accuracy']:.4f}")
    print(f"   Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"   Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"   F1-Score (macro):  {metrics['f1_macro']:.4f}")
    
    print(f"\n🎯 Per-Class Performance:")
    print(f"   {'Class':<5} {'Name':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print(f"   {'-'*70}")
    for cls, cls_metrics in metrics['per_class'].items():
        print(f"   {cls:<5} {cls_metrics['class_name']:<15} "
              f"{cls_metrics['precision']:<10.4f} {cls_metrics['recall']:<10.4f} "
              f"{cls_metrics['f1']:<10.4f} {cls_metrics['support']:<10}")
    
    print(f"\n🔮 Prediction Confidence:")
    print(f"   Mean:   {metrics['confidence']['mean']:.4f}")
    print(f"   Median: {metrics['confidence']['median']:.4f}")
    print(f"   Std:    {metrics['confidence']['std']:.4f}")
    print(f"   Range:  [{metrics['confidence']['min']:.4f}, {metrics['confidence']['max']:.4f}]")
    
    print(f"\n❌ Misclassification Analysis:")
    print(f"   Misclassified: {metrics['misclassification']['count']} "
          f"({metrics['misclassification']['rate']*100:.2f}%)")
    print(f"   Low Confidence (<{metrics['misclassification']['low_confidence_threshold']}): "
          f"{metrics['misclassification']['low_confidence_count']}")
    
    print("\n" + "="*60 + "\n")


def compare_metrics(baseline_metrics: dict, current_metrics: dict) -> dict:
    """
    Compare current metrics with baseline.
    
    Args:
        baseline_metrics: Baseline evaluation metrics
        current_metrics: Current evaluation metrics
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'accuracy_change': current_metrics['accuracy'] - baseline_metrics['accuracy'],
        'f1_macro_change': current_metrics['f1_macro'] - baseline_metrics['f1_macro'],
        'confidence_mean_change': current_metrics['confidence']['mean'] - baseline_metrics['confidence']['mean'],
        'misclassification_rate_change': (
            current_metrics['misclassification']['rate'] - 
            baseline_metrics['misclassification']['rate']
        )
    }
    
    # Per-class changes
    per_class_changes = {}
    for cls in baseline_metrics['per_class'].keys():
        if cls in current_metrics['per_class']:
            per_class_changes[cls] = {
                'f1_change': (
                    current_metrics['per_class'][cls]['f1'] - 
                    baseline_metrics['per_class'][cls]['f1']
                )
            }
    
    comparison['per_class_changes'] = per_class_changes
    
    return comparison


def print_comparison(comparison: dict, baseline_batch: str, current_batch: int):
    """
    Print comparison results.
    
    Args:
        comparison: Comparison results dictionary
        baseline_batch: Baseline batch identifier (e.g., "Training")
        current_batch: Current batch ID
    """
    print("\n" + "="*60)
    print(f"📊 COMPARISON: {baseline_batch} vs Batch {current_batch}")
    print("="*60)
    
    print(f"\n📈 Overall Metric Changes:")
    
    def format_change(value, is_percentage=True):
        sign = "+" if value > 0 else ""
        if is_percentage:
            return f"{sign}{value*100:.2f}%"
        return f"{sign}{value:.4f}"
    
    print(f"   Accuracy:             {format_change(comparison['accuracy_change'])}")
    print(f"   F1-Score (macro):     {format_change(comparison['f1_macro_change'])}")
    print(f"   Confidence (mean):    {format_change(comparison['confidence_mean_change'], False)}")
    print(f"   Misclassif. Rate:     {format_change(comparison['misclassification_rate_change'])}")
    
    # Check for significant drops
    config = load_config()
    thresholds = config['drift']['performance']
    
    print(f"\n⚠️ Performance Degradation Alerts:")
    alerts = []
    
    if abs(comparison['accuracy_change']) > thresholds['accuracy_drop_threshold']:
        if comparison['accuracy_change'] < 0:
            alerts.append(f"   🔴 Accuracy dropped by {abs(comparison['accuracy_change'])*100:.2f}% "
                        f"(threshold: {thresholds['accuracy_drop_threshold']*100:.1f}%)")
    
    if abs(comparison['f1_macro_change']) > thresholds['f1_drop_threshold']:
        if comparison['f1_macro_change'] < 0:
            alerts.append(f"   🔴 F1-Score dropped by {abs(comparison['f1_macro_change'])*100:.2f}% "
                        f"(threshold: {thresholds['f1_drop_threshold']*100:.1f}%)")
    
    if alerts:
        for alert in alerts:
            print(alert)
    else:
        print("   ✅ No significant performance degradation detected")
    
    print("\n" + "="*60 + "\n")


def save_evaluation_report(metrics: dict, report_path: str):
    """
    Save evaluation report to JSON file.
    
    Args:
        metrics: Evaluation metrics dictionary
        report_path: Path to save the report
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Add timestamp
    metrics['evaluation_timestamp'] = datetime.now().isoformat()
    
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Saved evaluation report to: {report_path}")


if __name__ == "__main__":
    # Test evaluation module
    print("🧪 Testing Model Evaluation Module\n")
    
    from src.models.train import load_model
    from src.data.data_loader import load_batch_data
    from src.data.preprocess import scale_features
    import pickle
    
    # Load model and scaler
    config = load_config()
    model, metadata = load_model()
    
    with open(os.path.join(config['paths']['models'], 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load and evaluate on batch 7
    batch7 = load_batch_data(7)
    feature_cols = [col for col in batch7.columns if col.startswith('feature_')]
    batch7[feature_cols] = scaler.transform(batch7[feature_cols])
    
    metrics = evaluate_model(model, batch7, batch_id=7)
    print_evaluation_results(metrics, batch_id=7)
    
    print("✅ Model evaluation module working correctly!")
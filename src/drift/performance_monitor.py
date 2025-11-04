"""
Performance Monitoring Module
Tracks model performance degradation over time (concept drift)
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
from datetime import datetime


def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def monitor_performance_degradation(baseline_metrics: dict,
                                    current_metrics: dict,
                                    batch_id: int) -> dict:
    """
    Monitor performance degradation compared to baseline.
    
    Args:
        baseline_metrics: Baseline evaluation metrics (e.g., from validation set)
        current_metrics: Current batch evaluation metrics
        batch_id: Current batch ID
    
    Returns:
        Dictionary with degradation analysis
    """
    config = load_config()
    thresholds = config['drift']['performance']
    
    print("\n" + "="*60)
    print(f"📉 PERFORMANCE DEGRADATION MONITORING - BATCH {batch_id}")
    print("="*60)
    
    # Calculate metric changes
    changes = {
        'accuracy': current_metrics['accuracy'] - baseline_metrics['accuracy'],
        'f1_macro': current_metrics['f1_macro'] - baseline_metrics['f1_macro'],
        'precision_macro': current_metrics['precision_macro'] - baseline_metrics['precision_macro'],
        'recall_macro': current_metrics['recall_macro'] - baseline_metrics['recall_macro'],
        'confidence_mean': current_metrics['confidence']['mean'] - baseline_metrics['confidence']['mean']
    }
    
    # Detect degradation
    degradation_detected = (
        abs(changes['accuracy']) > thresholds['accuracy_drop_threshold'] or
        abs(changes['f1_macro']) > thresholds['f1_drop_threshold']
    )
    
    # Per-class degradation
    per_class_degradation = {}
    for cls in baseline_metrics['per_class'].keys():
        if cls in current_metrics['per_class']:
            baseline_f1 = baseline_metrics['per_class'][cls]['f1']
            current_f1 = current_metrics['per_class'][cls]['f1']
            
            per_class_degradation[cls] = {
                'class_name': current_metrics['per_class'][cls]['class_name'],
                'baseline_f1': baseline_f1,
                'current_f1': current_f1,
                'f1_change': current_f1 - baseline_f1,
                'degradation_detected': abs(current_f1 - baseline_f1) > thresholds['f1_drop_threshold']
            }
    
    monitoring_result = {
        'batch_id': batch_id,
        'timestamp': datetime.now().isoformat(),
        'baseline_accuracy': baseline_metrics['accuracy'],
        'current_accuracy': current_metrics['accuracy'],
        'metric_changes': changes,
        'degradation_detected': degradation_detected,
        'per_class_degradation': per_class_degradation,
        'thresholds': thresholds
    }
    
    # Print results
    print(f"\n📊 Performance Changes:")
    print(f"   Accuracy:       {baseline_metrics['accuracy']:.4f} → {current_metrics['accuracy']:.4f} "
          f"({changes['accuracy']:+.4f})")
    print(f"   F1-Score:       {baseline_metrics['f1_macro']:.4f} → {current_metrics['f1_macro']:.4f} "
          f"({changes['f1_macro']:+.4f})")
    print(f"   Precision:      {baseline_metrics['precision_macro']:.4f} → {current_metrics['precision_macro']:.4f} "
          f"({changes['precision_macro']:+.4f})")
    print(f"   Recall:         {baseline_metrics['recall_macro']:.4f} → {current_metrics['recall_macro']:.4f} "
          f"({changes['recall_macro']:+.4f})")
    print(f"   Confidence:     {baseline_metrics['confidence']['mean']:.4f} → {current_metrics['confidence']['mean']:.4f} "
          f"({changes['confidence_mean']:+.4f})")
    
    # Alert on degradation
    print(f"\n⚠️ Degradation Analysis:")
    if degradation_detected:
        print(f"   🔴 PERFORMANCE DEGRADATION DETECTED!")
        
        if abs(changes['accuracy']) > thresholds['accuracy_drop_threshold']:
            print(f"      - Accuracy change ({abs(changes['accuracy'])*100:.2f}%) exceeds threshold "
                  f"({thresholds['accuracy_drop_threshold']*100:.1f}%)")
        
        if abs(changes['f1_macro']) > thresholds['f1_drop_threshold']:
            print(f"      - F1-Score change ({abs(changes['f1_macro'])*100:.2f}%) exceeds threshold "
                  f"({thresholds['f1_drop_threshold']*100:.1f}%)")
        
        # Check per-class degradation
        degraded_classes = [cls for cls, deg in per_class_degradation.items() if deg['degradation_detected']]
        if degraded_classes:
            print(f"\n   📌 Classes with Significant Degradation:")
            for cls in degraded_classes:
                deg = per_class_degradation[cls]
                print(f"      {cls} ({deg['class_name']}): {deg['baseline_f1']:.4f} → {deg['current_f1']:.4f} "
                      f"({deg['f1_change']:+.4f})")
    else:
        print(f"   ✅ No significant performance degradation detected")
    
    print("\n" + "="*60 + "\n")
    
    return monitoring_result


def analyze_confidence_drift(baseline_confidence: dict,
                             current_confidence: dict,
                             batch_id: int) -> dict:
    """
    Analyze changes in prediction confidence distribution.
    
    Args:
        baseline_confidence: Baseline confidence statistics
        current_confidence: Current confidence statistics
        batch_id: Current batch ID
    
    Returns:
        Dictionary with confidence drift analysis
    """
    print(f"\n🔮 Analyzing Prediction Confidence Drift - Batch {batch_id}")
    
    confidence_changes = {
        'mean_change': current_confidence['mean'] - baseline_confidence['mean'],
        'std_change': current_confidence['std'] - baseline_confidence['std'],
        'median_change': current_confidence['median'] - baseline_confidence['median']
    }
    
    # Check if confidence is dropping
    config = load_config()
    min_threshold = config['drift']['performance']['min_confidence_threshold']
    
    confidence_warning = current_confidence['mean'] < min_threshold
    
    confidence_analysis = {
        'batch_id': batch_id,
        'baseline_mean': baseline_confidence['mean'],
        'current_mean': current_confidence['mean'],
        'changes': confidence_changes,
        'low_confidence_warning': confidence_warning,
        'min_threshold': min_threshold
    }
    
    print(f"   Mean Confidence: {baseline_confidence['mean']:.4f} → {current_confidence['mean']:.4f} "
          f"({confidence_changes['mean_change']:+.4f})")
    print(f"   Std Confidence:  {baseline_confidence['std']:.4f} → {current_confidence['std']:.4f} "
          f"({confidence_changes['std_change']:+.4f})")
    
    if confidence_warning:
        print(f"   ⚠️ WARNING: Mean confidence ({current_confidence['mean']:.4f}) below threshold ({min_threshold})")
    else:
        print(f"   ✅ Confidence levels acceptable")
    
    return confidence_analysis


def track_misclassification_patterns(baseline_cm: list,
                                     current_cm: list,
                                     batch_id: int) -> dict:
    """
    Analyze changes in misclassification patterns.
    
    Args:
        baseline_cm: Baseline confusion matrix
        current_cm: Current confusion matrix
        batch_id: Current batch ID
    
    Returns:
        Dictionary with misclassification pattern analysis
    """
    config = load_config()
    class_names = config['data']['classes']
    
    baseline_cm = np.array(baseline_cm)
    current_cm = np.array(current_cm)
    
    print(f"\n🎯 Analyzing Misclassification Patterns - Batch {batch_id}")
    
    # Normalize confusion matrices
    baseline_cm_norm = baseline_cm.astype('float') / baseline_cm.sum(axis=1)[:, np.newaxis]
    current_cm_norm = current_cm.astype('float') / current_cm.sum(axis=1)[:, np.newaxis]
    
    # Find significant changes in confusion patterns
    cm_diff = current_cm_norm - baseline_cm_norm
    
    # Find top misclassification changes
    np.fill_diagonal(cm_diff, 0)  # Ignore diagonal (correct classifications)
    
    top_changes = []
    for i in range(cm_diff.shape[0]):
        for j in range(cm_diff.shape[1]):
            if abs(cm_diff[i, j]) > 0.05:  # 5% change threshold
                top_changes.append({
                    'true_class': i + 1,
                    'true_class_name': class_names[i + 1],
                    'predicted_class': j + 1,
                    'predicted_class_name': class_names[j + 1],
                    'change': float(cm_diff[i, j])
                })
    
    top_changes.sort(key=lambda x: abs(x['change']), reverse=True)
    
    pattern_analysis = {
        'batch_id': batch_id,
        'significant_changes': top_changes[:10],  # Top 10 changes
        'total_significant_changes': len(top_changes)
    }
    
    if top_changes:
        print(f"   Found {len(top_changes)} significant changes in misclassification patterns:")
        for i, change in enumerate(top_changes[:5], 1):
            print(f"      {i}. {change['true_class_name']} → {change['predicted_class_name']}: "
                  f"{change['change']:+.3f}")
    else:
        print(f"   ✅ No significant changes in misclassification patterns")
    
    return pattern_analysis


def comprehensive_performance_monitoring(baseline_metrics: dict,
                                         current_metrics: dict,
                                         batch_id: int) -> dict:
    """
    Perform comprehensive performance monitoring.
    
    Args:
        baseline_metrics: Baseline evaluation metrics
        current_metrics: Current evaluation metrics
        batch_id: Current batch ID
    
    Returns:
        Dictionary with comprehensive monitoring results
    """
    print("\n" + "="*60)
    print(f"📊 COMPREHENSIVE PERFORMANCE MONITORING - BATCH {batch_id}")
    print("="*60)
    
    # Performance degradation analysis
    degradation = monitor_performance_degradation(baseline_metrics, current_metrics, batch_id)
    
    # Confidence drift analysis
    confidence_drift = analyze_confidence_drift(
        baseline_metrics['confidence'],
        current_metrics['confidence'],
        batch_id
    )
    
    # Misclassification pattern analysis
    misclassification_patterns = track_misclassification_patterns(
        baseline_metrics['confusion_matrix'],
        current_metrics['confusion_matrix'],
        batch_id
    )
    
    # Overall assessment
    monitoring_result = {
        'batch_id': batch_id,
        'timestamp': datetime.now().isoformat(),
        'degradation_analysis': degradation,
        'confidence_analysis': confidence_drift,
        'misclassification_patterns': misclassification_patterns,
        'overall_degradation_detected': (
            degradation['degradation_detected'] or
            confidence_drift['low_confidence_warning']
        )
    }
    
    print("\n" + "="*60)
    print(f"🎯 OVERALL ASSESSMENT:")
    if monitoring_result['overall_degradation_detected']:
        print(f"   🔴 CONCEPT DRIFT DETECTED - Model performance degraded!")
        print(f"   📌 Recommendation: Consider retraining the model")
    else:
        print(f"   ✅ Model performance stable - No concept drift detected")
    print("="*60 + "\n")
    
    return monitoring_result


def save_performance_report(monitoring_result: dict, report_dir: str = None):
    """
    Save performance monitoring report.
    
    Args:
        monitoring_result: Monitoring result dictionary
        report_dir: Directory to save reports
    """
    config = load_config()
    
    if report_dir is None:
        report_dir = config['paths']['drift_reports']
    
    os.makedirs(report_dir, exist_ok=True)
    
    batch_id = monitoring_result['batch_id']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"performance_report_batch{batch_id}_{timestamp}.json")
    
    with open(report_path, 'w') as f:
        json.dump(monitoring_result, f, indent=2)
    
    print(f"💾 Saved performance report to: {report_path}")
    
    return report_path


if __name__ == "__main__":
    print("🧪 Testing Performance Monitoring Module\n")
    
    from src.models.train import load_model
    from src.models.evaluate import evaluate_model
    from src.data.data_loader import load_multiple_batches, load_batch_data
    import pickle
    
    # Load config, model, and scaler
    config = load_config()
    model, metadata = load_model()
    
    with open(os.path.join(config['paths']['models'], 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Get baseline metrics (from training data)
    train_data = load_multiple_batches(config['data']['train_batches'])
    feature_cols = [col for col in train_data.columns if col.startswith('feature_')]
    train_data[feature_cols] = scaler.transform(train_data[feature_cols])
    baseline_metrics = evaluate_model(model, train_data)
    
    # Get current metrics (batch 7)
    batch7 = load_batch_data(7)
    batch7[feature_cols] = scaler.transform(batch7[feature_cols])
    current_metrics = evaluate_model(model, batch7, batch_id=7)
    
    # Perform monitoring
    monitoring = comprehensive_performance_monitoring(baseline_metrics, current_metrics, batch_id=7)
    
    print("\n✅ Performance monitoring module working correctly!")
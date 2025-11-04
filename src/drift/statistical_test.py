"""
Statistical Drift Detection Module
Implements KS test, PSI, Wasserstein distance for feature drift detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
import yaml
import json
from datetime import datetime
import os


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures the shift in distribution between two datasets.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change (drift detected)
    
    Args:
        reference: Reference distribution (e.g., training data)
        current: Current distribution (e.g., test data)
        bins: Number of bins for discretization
    
    Returns:
        PSI value
    """
    # Create bins based on reference data
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    
    if len(breakpoints) < 2:
        return 0.0  # Cannot calculate PSI with less than 2 unique values
    
    # Bin the data
    reference_binned = np.digitize(reference, breakpoints[:-1], right=False)
    current_binned = np.digitize(current, breakpoints[:-1], right=False)
    
    # Calculate proportions
    reference_counts = np.bincount(reference_binned, minlength=len(breakpoints))
    current_counts = np.bincount(current_binned, minlength=len(breakpoints))
    
    reference_props = reference_counts / len(reference)
    current_props = current_counts / len(current)
    
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    reference_props = np.where(reference_props == 0, epsilon, reference_props)
    current_props = np.where(current_props == 0, epsilon, current_props)
    
    # Calculate PSI
    psi = np.sum((current_props - reference_props) * np.log(current_props / reference_props))
    
    return float(psi)


def kolmogorov_smirnov_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Perform Kolmogorov-Smirnov test for distribution comparison.
    
    Args:
        reference: Reference distribution
        current: Current distribution
    
    Returns:
        Dictionary with test statistic and p-value
    """
    statistic, p_value = stats.ks_2samp(reference, current)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'drift_detected': p_value < 0.05  # Reject null hypothesis at 0.05 level
    }


def calculate_wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Calculate Wasserstein distance (Earth Mover's Distance).
    
    Args:
        reference: Reference distribution
        current: Current distribution
    
    Returns:
        Wasserstein distance
    """
    return float(wasserstein_distance(reference, current))


def detect_feature_drift(reference_df: pd.DataFrame, 
                        current_df: pd.DataFrame,
                        feature_importances: pd.DataFrame = None,
                        top_n_features: int = None) -> dict:
    """
    Detect drift in feature distributions.
    
    Args:
        reference_df: Reference DataFrame (training data)
        current_df: Current DataFrame (test data)
        feature_importances: DataFrame with feature importances
        top_n_features: Number of top features to monitor (None for all)
    
    Returns:
        Dictionary with drift detection results
    """
    config = load_config()
    
    feature_cols = [col for col in reference_df.columns if col.startswith('feature_')]
    
    # If feature importances provided, focus on top features
    if feature_importances is not None and top_n_features is not None:
        top_features = feature_importances.head(top_n_features)['feature'].tolist()
        feature_cols = [col for col in feature_cols if col in top_features]
    
    print(f"\n🔍 Detecting drift in {len(feature_cols)} features...")
    
    drift_results = {
        'n_features_tested': len(feature_cols),
        'feature_drift': {}
    }
    
    thresholds = config['drift']['statistical']
    
    psi_drifts = []
    ks_drifts = []
    wasserstein_drifts = []
    
    for feature in feature_cols:
        reference_values = reference_df[feature].values
        current_values = current_df[feature].values
        
        # Calculate drift metrics
        psi = calculate_psi(reference_values, current_values)
        ks_result = kolmogorov_smirnov_test(reference_values, current_values)
        wasserstein = calculate_wasserstein_distance(reference_values, current_values)
        
        # Determine if drift detected
        psi_drift = psi >= thresholds['psi_threshold']
        ks_drift = ks_result['drift_detected']
        wasserstein_drift = wasserstein >= thresholds['wasserstein_threshold']
        
        if psi_drift:
            psi_drifts.append(feature)
        if ks_drift:
            ks_drifts.append(feature)
        if wasserstein_drift:
            wasserstein_drifts.append(feature)
        
        drift_results['feature_drift'][feature] = {
            'psi': psi,
            'psi_drift': psi_drift,
            'ks_statistic': ks_result['statistic'],
            'ks_p_value': ks_result['p_value'],
            'ks_drift': ks_drift,
            'wasserstein_distance': wasserstein,
            'wasserstein_drift': wasserstein_drift,
            'any_drift': psi_drift or ks_drift or wasserstein_drift
        }
    
    # Summary statistics
    drift_results['summary'] = {
        'psi_drifts': len(psi_drifts),
        'ks_drifts': len(ks_drifts),
        'wasserstein_drifts': len(wasserstein_drifts),
        'total_features_with_drift': len(set(psi_drifts + ks_drifts + wasserstein_drifts)),
        'features_with_psi_drift': psi_drifts,
        'features_with_ks_drift': ks_drifts,
        'features_with_wasserstein_drift': wasserstein_drifts
    }
    
    return drift_results


def print_drift_summary(drift_results: dict, batch_id: int = None):
    """
    Print drift detection summary.
    
    Args:
        drift_results: Dictionary with drift detection results
        batch_id: Current batch ID
    """
    print("\n" + "="*60)
    if batch_id:
        print(f"🔍 STATISTICAL DRIFT DETECTION - BATCH {batch_id}")
    else:
        print("🔍 STATISTICAL DRIFT DETECTION SUMMARY")
    print("="*60)
    
    summary = drift_results['summary']
    
    print(f"\n📊 Drift Detection Summary:")
    print(f"   Features Tested: {drift_results['n_features_tested']}")
    print(f"   Total Features with Drift: {summary['total_features_with_drift']}")
    print(f"   - PSI Drift:          {summary['psi_drifts']}")
    print(f"   - KS Test Drift:      {summary['ks_drifts']}")
    print(f"   - Wasserstein Drift:  {summary['wasserstein_drifts']}")
    
    if summary['total_features_with_drift'] > 0:
        print(f"\n⚠️ DRIFT DETECTED!")
        
        # Show top drifted features by PSI
        feature_drift = drift_results['feature_drift']
        drifted_features = [(f, d['psi']) for f, d in feature_drift.items() if d['any_drift']]
        drifted_features.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n   Top Drifted Features (by PSI):")
        for i, (feature, psi) in enumerate(drifted_features[:10], 1):
            ks_p = feature_drift[feature]['ks_p_value']
            wass = feature_drift[feature]['wasserstein_distance']
            print(f"      {i:2d}. {feature}: PSI={psi:.4f}, KS_p={ks_p:.4f}, Wass={wass:.4f}")
    else:
        print(f"\n✅ No significant drift detected")
    
    print("\n" + "="*60 + "\n")


def detect_label_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Detect drift in label distribution.
    
    Args:
        reference_df: Reference DataFrame
        current_df: Current DataFrame
    
    Returns:
        Dictionary with label drift results
    """
    print(f"\n🏷️ Detecting label distribution drift...")
    
    # Get label distributions
    reference_dist = reference_df['class'].value_counts(normalize=True).sort_index()
    current_dist = current_df['class'].value_counts(normalize=True).sort_index()
    
    # Align distributions
    all_classes = sorted(set(reference_dist.index) | set(current_dist.index))
    reference_props = np.array([reference_dist.get(c, 0) for c in all_classes])
    current_props = np.array([current_dist.get(c, 0) for c in all_classes])
    
    # Create contingency table for chi-square test
    reference_counts = reference_df['class'].value_counts().reindex(all_classes, fill_value=0)
    current_counts = current_df['class'].value_counts().reindex(all_classes, fill_value=0)
    
    # Create 2xN contingency table
    contingency_table = np.array([reference_counts.values, current_counts.values])
    
    # Perform chi-square test on contingency table
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Calculate KL divergence
    epsilon = 1e-10
    reference_props_safe = np.where(reference_props == 0, epsilon, reference_props)
    current_props_safe = np.where(current_props == 0, epsilon, current_props)
    
    kl_divergence = np.sum(current_props_safe * np.log(current_props_safe / reference_props_safe))
    
    # Calculate Population Stability Index (PSI) for labels
    psi = np.sum((current_props_safe - reference_props_safe) * 
                 np.log(current_props_safe / reference_props_safe))
    
    label_drift = {
        'reference_distribution': reference_dist.to_dict(),
        'current_distribution': current_dist.to_dict(),
        'chi2_statistic': float(chi2_stat),
        'chi2_p_value': float(p_value),
        'kl_divergence': float(kl_divergence),
        'psi': float(psi),
        'drift_detected': p_value < 0.05 or psi >= 0.1
    }
    
    print(f"   Chi-square p-value: {p_value:.4f}")
    print(f"   KL Divergence: {kl_divergence:.4f}")
    print(f"   PSI: {psi:.4f}")
    print(f"   Drift detected: {'Yes' if label_drift['drift_detected'] else 'No'}")
    
    return label_drift


def comprehensive_drift_analysis(reference_df: pd.DataFrame,
                                 current_df: pd.DataFrame,
                                 batch_id: int,
                                 feature_importances: pd.DataFrame = None) -> dict:
    """
    Perform comprehensive drift analysis.
    
    Args:
        reference_df: Reference DataFrame (training data)
        current_df: Current DataFrame (test data)
        batch_id: Current batch ID
        feature_importances: Feature importances from trained model
    
    Returns:
        Dictionary with comprehensive drift analysis
    """
    config = load_config()
    
    print("\n" + "="*60)
    print(f"🔬 COMPREHENSIVE DRIFT ANALYSIS - BATCH {batch_id}")
    print("="*60)
    
    # Feature drift detection
    feature_drift = detect_feature_drift(
        reference_df, 
        current_df,
        feature_importances,
        top_n_features=config['drift']['feature_drift']['top_n_features']
    )
    
    # Label drift detection
    label_drift = detect_label_drift(reference_df, current_df)
    
    # Overall assessment
    overall_drift = {
        'batch_id': batch_id,
        'timestamp': datetime.now().isoformat(),
        'feature_drift': feature_drift,
        'label_drift': label_drift,
        'overall_drift_detected': (
            feature_drift['summary']['total_features_with_drift'] > 0 or
            label_drift['drift_detected']
        )
    }
    
    # Print summaries
    print_drift_summary(feature_drift, batch_id)
    
    return overall_drift


def save_drift_report(drift_analysis: dict, report_dir: str = None):
    """
    Save drift analysis report.
    
    Args:
        drift_analysis: Drift analysis dictionary
        report_dir: Directory to save reports
    """
    config = load_config()
    
    if report_dir is None:
        report_dir = config['paths']['drift_reports']
    
    os.makedirs(report_dir, exist_ok=True)
    
    batch_id = drift_analysis['batch_id']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"drift_report_batch{batch_id}_{timestamp}.json")

    # Convert numpy types to Python native types for JSON serialization
    drift_analysis_serializable = convert_numpy_types(drift_analysis)

    with open(report_path, 'w') as f:
        json.dump(drift_analysis_serializable, f, indent=2)
    
    print(f"💾 Saved drift report to: {report_path}")
    
    return report_path


if __name__ == "__main__":
    print("🧪 Testing Statistical Drift Detection Module\n")
    
    from src.data.data_loader import load_multiple_batches, load_batch_data
    from src.data.preprocess import scale_features
    import pickle
    
    # Load config and scaler
    config = load_config()
    with open(os.path.join(config['paths']['models'], 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load training data (reference)
    train_batches = load_multiple_batches(config['data']['train_batches'])
    feature_cols = [col for col in train_batches.columns if col.startswith('feature_')]
    train_batches[feature_cols] = scaler.transform(train_batches[feature_cols])
    
    # Load test batch
    batch7 = load_batch_data(7)
    batch7[feature_cols] = scaler.transform(batch7[feature_cols])
    
    # Perform drift analysis
    drift_analysis = comprehensive_drift_analysis(train_batches, batch7, batch_id=7)
    
    print("\n✅ Statistical drift detection module working correctly!")
"""
Airflow DAG: Monitoring Pipeline for Gas Sensor Drift Detection
Batch-by-batch monitoring: Load → Preprocess → Predict → Detect Drift (Statistical + Performance)
Monitors batches 7, 8, 9, 10 sequentially to observe drift progression
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os
import json

# Add project root to path
import pathlib
# In Docker, files are mounted at /opt/airflow
# In local, files are at ~/airflow
airflow_home = os.getenv('AIRFLOW_HOME', str(pathlib.Path.home() / 'airflow'))
sys.path.insert(0, airflow_home)

# Import project modules
from src.data.data_loader import load_batch_data
from src.models.train import load_model
from src.models.evaluate import evaluate_model, print_evaluation_results, save_evaluation_report, compare_metrics, print_comparison
from src.drift.statistical_test import comprehensive_drift_analysis, save_drift_report
from src.drift.performance_monitor import comprehensive_performance_monitoring, save_performance_report
import pandas as pd
import pickle
import yaml


# Load configuration
def load_config():
    # Use AIRFLOW_HOME to find config file
    config_path = os.path.join(airflow_home, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Function to process a single batch
def monitor_batch(batch_id: int, **context):
    """
    Complete monitoring workflow for a single batch.
    
    Args:
        batch_id: Batch number to monitor (7, 8, 9, or 10)
    """
    print("\n" + "="*80)
    print(f"🔍 MONITORING BATCH {batch_id}")
    print("="*80 + "\n")
    
    config = load_config()
    
    # STEP 1: Load batch data
    print(f"📥 Step 1: Loading Batch {batch_id} data...")
    batch_df = load_batch_data(batch_id)
    
    # STEP 2: Load model and scaler
    print(f"\n🤖 Step 2: Loading trained model and scaler...")
    model, model_metadata = load_model()
    
    scaler_path = os.path.join(config['paths']['models'], 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # STEP 3: Scale features
    print(f"\n⚙️ Step 3: Scaling features...")
    feature_cols = [col for col in batch_df.columns if col.startswith('feature_')]
    batch_df[feature_cols] = scaler.transform(batch_df[feature_cols])
    
    # STEP 4: Load baseline metrics
    print(f"\n📊 Step 4: Loading baseline metrics...")
    baseline_metrics_path = os.path.join(config['paths']['drift_reports'], 'baseline_metrics.json')
    with open(baseline_metrics_path, 'r') as f:
        baseline_metrics = json.load(f)
    
    # STEP 5: Load reference data for statistical drift detection
    print(f"\n📚 Step 5: Loading reference (training) data...")
    train_scaled_path = os.path.join(config['paths']['processed_data'], 'train_scaled.csv')
    train_df = pd.read_csv(train_scaled_path)
    
    # STEP 6: Evaluate model performance on current batch
    print(f"\n🎯 Step 6: Evaluating model on Batch {batch_id}...")
    batch_metrics = evaluate_model(model, batch_df, batch_id=batch_id)
    print_evaluation_results(batch_metrics, batch_id=batch_id)
    
    # Save batch evaluation
    eval_report_path = os.path.join(
        config['paths']['drift_reports'], 
        f'batch{batch_id}_evaluation.json'
    )
    save_evaluation_report(batch_metrics, eval_report_path)
    
    # STEP 7: Performance comparison
    print(f"\n📈 Step 7: Comparing performance with baseline...")
    comparison = compare_metrics(baseline_metrics, batch_metrics)
    print_comparison(comparison, "Training (Baseline)", batch_id)
    
    # STEP 8: Statistical drift detection
    print(f"\n🔬 Step 8: Detecting statistical drift...")
    
    # Get feature importances for focused analysis
    feature_importance_df = None
    if model_metadata and 'top_features' in model_metadata:
        feature_importance_df = pd.DataFrame(model_metadata['top_features'])
    
    drift_analysis = comprehensive_drift_analysis(
        train_df,
        batch_df,
        batch_id,
        feature_importance_df
    )
    
    # Save drift report
    drift_report_path = save_drift_report(drift_analysis)
    
    # STEP 9: Performance monitoring (concept drift)
    print(f"\n📉 Step 9: Monitoring performance degradation...")
    performance_monitoring = comprehensive_performance_monitoring(
        baseline_metrics,
        batch_metrics,
        batch_id
    )
    
    # Save performance report
    perf_report_path = save_performance_report(performance_monitoring)
    
    # STEP 10: Overall drift assessment
    print("\n" + "="*80)
    print(f"🎯 OVERALL DRIFT ASSESSMENT - BATCH {batch_id}")
    print("="*80)
    
    statistical_drift = drift_analysis['overall_drift_detected']
    performance_drift = performance_monitoring['overall_degradation_detected']
    
    print(f"\n📊 Drift Detection Summary:")
    print(f"   Statistical Drift (Feature Distribution): {'🔴 DETECTED' if statistical_drift else '✅ NOT DETECTED'}")
    print(f"   Performance Drift (Model Degradation):   {'🔴 DETECTED' if performance_drift else '✅ NOT DETECTED'}")
    
    if statistical_drift or performance_drift:
        print(f"\n⚠️ DRIFT DETECTED IN BATCH {batch_id}!")
        print(f"\n   📌 Recommendations:")
        if statistical_drift:
            print(f"      - Feature distributions have shifted significantly")
            print(f"      - Input data characteristics are changing")
        if performance_drift:
            print(f"      - Model performance has degraded")
            print(f"      - Relationship between features and labels may have changed")
        print(f"      - Consider retraining the model with recent data")
        print(f"      - Review data collection process for anomalies")
    else:
        print(f"\n✅ NO SIGNIFICANT DRIFT DETECTED IN BATCH {batch_id}")
        print(f"   Model continues to perform well on this batch")
    
    print("\n" + "="*80 + "\n")
    
    # STEP 11: Push results to XCom for downstream tasks
    context['ti'].xcom_push(key=f'batch{batch_id}_accuracy', value=batch_metrics['accuracy'])
    context['ti'].xcom_push(key=f'batch{batch_id}_f1', value=batch_metrics['f1_macro'])
    context['ti'].xcom_push(key=f'batch{batch_id}_statistical_drift', value=statistical_drift)
    context['ti'].xcom_push(key=f'batch{batch_id}_performance_drift', value=performance_drift)
    context['ti'].xcom_push(key=f'batch{batch_id}_drift_detected', value=statistical_drift or performance_drift)
    
    return {
        'batch_id': batch_id,
        'accuracy': batch_metrics['accuracy'],
        'f1_macro': batch_metrics['f1_macro'],
        'statistical_drift': statistical_drift,
        'performance_drift': performance_drift,
        'drift_detected': statistical_drift or performance_drift
    }


# Individual batch monitoring tasks
def monitor_batch_7(**context):
    """Monitor Batch 7"""
    return monitor_batch(7, **context)


def monitor_batch_8(**context):
    """Monitor Batch 8"""
    return monitor_batch(8, **context)


def monitor_batch_9(**context):
    """Monitor Batch 9"""
    return monitor_batch(9, **context)


def monitor_batch_10(**context):
    """Monitor Batch 10"""
    return monitor_batch(10, **context)


# Final summary task
def generate_monitoring_summary(**context):
    """Generate comprehensive summary of all monitored batches"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MONITORING SUMMARY - ALL BATCHES")
    print("="*80 + "\n")
    
    ti = context['ti']
    config = load_config()
    
    # Collect results from all batches
    results = []
    for batch_id in [7, 8, 9, 10]:
        result = {
            'batch_id': batch_id,
            'accuracy': ti.xcom_pull(task_ids=f'monitor_batch_{batch_id}', key=f'batch{batch_id}_accuracy'),
            'f1_macro': ti.xcom_pull(task_ids=f'monitor_batch_{batch_id}', key=f'batch{batch_id}_f1'),
            'statistical_drift': ti.xcom_pull(task_ids=f'monitor_batch_{batch_id}', key=f'batch{batch_id}_statistical_drift'),
            'performance_drift': ti.xcom_pull(task_ids=f'monitor_batch_{batch_id}', key=f'batch{batch_id}_performance_drift'),
            'drift_detected': ti.xcom_pull(task_ids=f'monitor_batch_{batch_id}', key=f'batch{batch_id}_drift_detected'),
        }
        results.append(result)
    
    # Print summary table
    print("📈 Batch-by-Batch Performance:")
    print(f"   {'Batch':<8} {'Accuracy':<12} {'F1-Score':<12} {'Stat Drift':<12} {'Perf Drift':<12} {'Overall':<12}")
    print(f"   {'-'*76}")
    
    for r in results:
        stat_icon = "🔴 YES" if r['statistical_drift'] else "✅ NO"
        perf_icon = "🔴 YES" if r['performance_drift'] else "✅ NO"
        overall_icon = "🔴 DRIFT" if r['drift_detected'] else "✅ STABLE"
        
        print(f"   Batch {r['batch_id']:<2}  {r['accuracy']:<12.4f} {r['f1_macro']:<12.4f} {stat_icon:<12} {perf_icon:<12} {overall_icon:<12}")
    
    # Overall trends
    print(f"\n📊 Drift Progression Over Time:")
    batches_with_drift = [r for r in results if r['drift_detected']]
    batches_with_stat_drift = [r for r in results if r['statistical_drift']]
    batches_with_perf_drift = [r for r in results if r['performance_drift']]
    
    print(f"   Total Batches Monitored:        {len(results)}")
    print(f"   Batches with Drift:             {len(batches_with_drift)} ({len(batches_with_drift)/len(results)*100:.1f}%)")
    print(f"   Batches with Statistical Drift: {len(batches_with_stat_drift)}")
    print(f"   Batches with Performance Drift: {len(batches_with_perf_drift)}")
    
    # Accuracy trend
    accuracies = [r['accuracy'] for r in results]
    print(f"\n📉 Performance Trends:")
    print(f"   Accuracy Range: {min(accuracies):.4f} - {max(accuracies):.4f}")
    print(f"   Accuracy Std:   {pd.Series(accuracies).std():.4f}")
    
    if accuracies[0] > accuracies[-1]:
        print(f"   ⚠️ Degrading trend: Accuracy decreased from Batch 7 to Batch 10")
    else:
        print(f"   ✅ Stable or improving trend")
    
    # Final recommendations
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    if len(batches_with_drift) >= 2:
        print(f"   🔴 CRITICAL: Drift detected in multiple batches ({len(batches_with_drift)}/4)")
        print(f"   📌 Action Required:")
        print(f"      1. Investigate root cause of drift (sensor calibration, environmental changes)")
        print(f"      2. Retrain model with recent data (include Batches 1-10)")
        print(f"      3. Update monitoring thresholds if drift is expected")
        print(f"      4. Consider online learning or model adaptation strategies")
    elif len(batches_with_drift) == 1:
        print(f"   ⚠️ WARNING: Drift detected in 1 batch")
        print(f"   📌 Recommendations:")
        print(f"      1. Continue monitoring subsequent batches")
        print(f"      2. Prepare for potential retraining if drift persists")
    else:
        print(f"   ✅ EXCELLENT: No drift detected across all monitored batches")
        print(f"   📌 Recommendations:")
        print(f"      1. Continue regular monitoring")
        print(f"      2. Current model performs well on deployment data")
    
    print("\n" + "="*80)
    print("✅ MONITORING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    # Save comprehensive summary
    summary_path = os.path.join(config['paths']['drift_reports'], 'monitoring_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'batches_with_drift': len(batches_with_drift),
            'recommendations': 'Retrain model' if len(batches_with_drift) >= 2 else 'Continue monitoring'
        }, f, indent=2)
    
    print(f"💾 Comprehensive summary saved to: {summary_path}\n")


# Define DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    '02_monitoring_pipeline',
    default_args=default_args,
    description='Batch-by-batch drift monitoring pipeline for Gas Sensor data',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'monitoring', 'drift-detection', 'gas-sensor'],
) as dag:
    
    # Define monitoring tasks for each batch
    task_monitor_batch7 = PythonOperator(
        task_id='monitor_batch_7',
        python_callable=monitor_batch_7,
        provide_context=True,
    )
    
    task_monitor_batch8 = PythonOperator(
        task_id='monitor_batch_8',
        python_callable=monitor_batch_8,
        provide_context=True,
    )
    
    task_monitor_batch9 = PythonOperator(
        task_id='monitor_batch_9',
        python_callable=monitor_batch_9,
        provide_context=True,
    )
    
    task_monitor_batch10 = PythonOperator(
        task_id='monitor_batch_10',
        python_callable=monitor_batch_10,
        provide_context=True,
    )
    
    task_summary = PythonOperator(
        task_id='generate_summary',
        python_callable=generate_monitoring_summary,
        provide_context=True,
    )
    
    # Define task dependencies - sequential monitoring to observe drift progression
    task_monitor_batch7 >> task_monitor_batch8 >> task_monitor_batch9 >> task_monitor_batch10 >> task_summary
"""
Airflow DAG: Training Pipeline for Gas Sensor Drift Detection
End-to-end pipeline: Data Loading → Preprocessing → Training → Evaluation
"""

from airflow import DAG 
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
import pathlib
# In Docker, files are mounted at /opt/airflow
# In local, files are at ~/airflow
airflow_home = os.getenv('AIRFLOW_HOME', str(pathlib.Path.home() / 'airflow'))
sys.path.insert(0, airflow_home)

# Import project modules
from src.data.data_loader import load_multiple_batches
from src.data.preprocess import preprocess_pipeline
from src.models.train import train_pipeline
from src.models.evaluate import evaluate_model, print_evaluation_results, save_evaluation_report
import yaml


# Load configuration
def load_config():
    # Use AIRFLOW_HOME to find config file
    config_path = os.path.join(airflow_home, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Task 1: Load Training Data
def load_training_data(**context):
    """Load batches 1-6 for training"""
    print("="*60)
    print("STEP 1: LOADING TRAINING DATA")
    print("="*60)
    
    config = load_config()
    train_batch_ids = config['data']['train_batches']
    
    print(f"\n📚 Loading training batches: {train_batch_ids}")
    train_df = load_multiple_batches(train_batch_ids)
    
    # Save to processed directory
    processed_dir = config['paths']['processed_data']
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, 'train_raw.csv'), index=False)
    
    print(f"\n✅ Training data loaded successfully!")
    print(f"   Shape: {train_df.shape}")
    print(f"   Saved to: {processed_dir}/train_raw.csv")
    
    # Push data info to XCom
    context['ti'].xcom_push(key='train_shape', value=train_df.shape)
    context['ti'].xcom_push(key='train_samples', value=len(train_df))


# Task 2: Load Test Data
def load_test_data(**context):
    """Load batches 7-10 for testing"""
    print("="*60)
    print("STEP 2: LOADING TEST DATA")
    print("="*60)
    
    config = load_config()
    test_batch_ids = config['data']['test_batches']
    
    print(f"\n🧪 Loading test batches: {test_batch_ids}")
    test_df = load_multiple_batches(test_batch_ids)
    
    # Save to processed directory
    processed_dir = config['paths']['processed_data']
    os.makedirs(processed_dir, exist_ok=True)
    test_df.to_csv(os.path.join(processed_dir, 'test_raw.csv'), index=False)
    
    print(f"\n✅ Test data loaded successfully!")
    print(f"   Shape: {test_df.shape}")
    print(f"   Saved to: {processed_dir}/test_raw.csv")
    
    # Push data info to XCom
    context['ti'].xcom_push(key='test_shape', value=test_df.shape)
    context['ti'].xcom_push(key='test_samples', value=len(test_df))


# Task 3: Preprocess Data
def preprocess_data(**context):
    """Preprocess and scale features"""
    print("="*60)
    print("STEP 3: PREPROCESSING DATA")
    print("="*60)
    
    # Run preprocessing pipeline
    train_scaled, test_scaled, scaler = preprocess_pipeline(save_intermediate=True)
    
    print(f"\n✅ Preprocessing completed successfully!")
    
    # Push preprocessing info to XCom
    context['ti'].xcom_push(key='preprocessing_complete', value=True)


# Task 4: Train Model
def train_model(**context):
    """Train Random Forest classifier"""
    print("="*60)
    print("STEP 4: TRAINING MODEL")
    print("="*60)
    
    import pandas as pd
    from src.models.train import train_pipeline
    
    config = load_config()
    processed_dir = config['paths']['processed_data']
    
    # Load preprocessed training data
    train_scaled = pd.read_csv(os.path.join(processed_dir, 'train_scaled.csv'))
    
    # Train model with cross-validation
    model, metadata = train_pipeline(
        train_scaled, 
        perform_cv=True, 
        save_model_flag=True
    )
    
    print(f"\n✅ Model training completed successfully!")
    print(f"   Training Accuracy: {metadata['training_accuracy']:.4f}")
    if metadata['cv_results']:
        print(f"   CV Mean Accuracy: {metadata['cv_results']['mean_accuracy']:.4f} ± "
              f"{metadata['cv_results']['std_accuracy']:.4f}")
    
    # Push training info to XCom
    context['ti'].xcom_push(key='training_accuracy', value=metadata['training_accuracy'])
    context['ti'].xcom_push(key='cv_mean_accuracy', value=metadata['cv_results']['mean_accuracy'] if metadata['cv_results'] else None)
    context['ti'].xcom_push(key='model_path', value=metadata.get('model_path'))


# Task 5: Evaluate on Training Data (Baseline)
def evaluate_on_training_data(**context):
    """Evaluate model on training data to establish baseline"""
    print("="*60)
    print("STEP 5: BASELINE EVALUATION (Training Data)")
    print("="*60)
    
    import pandas as pd
    from src.models.train import load_model
    from src.models.evaluate import evaluate_model, print_evaluation_results, save_evaluation_report
    
    config = load_config()
    processed_dir = config['paths']['processed_data']
    
    # Load model and data
    model, _ = load_model()
    train_scaled = pd.read_csv(os.path.join(processed_dir, 'train_scaled.csv'))
    
    # Evaluate
    baseline_metrics = evaluate_model(model, train_scaled)
    print_evaluation_results(baseline_metrics)
    
    # Save baseline metrics
    report_path = save_evaluation_report(
        baseline_metrics,
        os.path.join(config['paths']['drift_reports'], 'baseline_metrics.json')
    )
    
    print(f"\n✅ Baseline evaluation completed!")
    
    # Push metrics to XCom
    context['ti'].xcom_push(key='baseline_accuracy', value=baseline_metrics['accuracy'])
    context['ti'].xcom_push(key='baseline_f1', value=baseline_metrics['f1_macro'])


# Task 6: Evaluate on Test Data
def evaluate_on_test_data(**context):
    """Evaluate model on test data (batches 7-10)"""
    print("="*60)
    print("STEP 6: EVALUATION ON TEST DATA")
    print("="*60)
    
    import pandas as pd
    from src.models.train import load_model
    from src.models.evaluate import evaluate_model, print_evaluation_results, save_evaluation_report
    
    config = load_config()
    processed_dir = config['paths']['processed_data']
    
    # Load model and data
    model, _ = load_model()
    test_scaled = pd.read_csv(os.path.join(processed_dir, 'test_scaled.csv'))
    
    # Evaluate
    test_metrics = evaluate_model(model, test_scaled)
    print_evaluation_results(test_metrics)
    
    # Save test metrics
    report_path = save_evaluation_report(
        test_metrics,
        os.path.join(config['paths']['drift_reports'], 'test_metrics_all.json')
    )
    
    print(f"\n✅ Test evaluation completed!")
    
    # Push metrics to XCom
    context['ti'].xcom_push(key='test_accuracy', value=test_metrics['accuracy'])
    context['ti'].xcom_push(key='test_f1', value=test_metrics['f1_macro'])


# Task 7: Generate Training Summary
def generate_training_summary(**context):
    """Generate final summary of the training pipeline"""
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETE - SUMMARY")
    print("="*60)
    
    ti = context['ti']
    
    # Pull metrics from XCom
    train_samples = ti.xcom_pull(task_ids='load_training_data', key='train_samples')
    test_samples = ti.xcom_pull(task_ids='load_test_data', key='test_samples')
    training_accuracy = ti.xcom_pull(task_ids='train_model', key='training_accuracy')
    cv_accuracy = ti.xcom_pull(task_ids='train_model', key='cv_mean_accuracy')
    baseline_accuracy = ti.xcom_pull(task_ids='evaluate_on_training', key='baseline_accuracy')
    test_accuracy = ti.xcom_pull(task_ids='evaluate_on_test', key='test_accuracy')
    model_path = ti.xcom_pull(task_ids='train_model', key='model_path')
    
    print(f"\n📊 Data Summary:")
    print(f"   Training Samples: {train_samples}")
    print(f"   Test Samples:     {test_samples}")
    
    print(f"\n🌲 Model Performance:")
    print(f"   Training Accuracy:       {training_accuracy:.4f}")
    if cv_accuracy:
        print(f"   CV Accuracy:             {cv_accuracy:.4f}")
    print(f"   Baseline Accuracy:       {baseline_accuracy:.4f}")
    print(f"   Test Accuracy (All):     {test_accuracy:.4f}")
    
    print(f"\n💾 Artifacts:")
    print(f"   Model Path: {model_path}")
    
    print(f"\n✅ Training pipeline executed successfully!")
    print(f"📌 Next step: Run the monitoring pipeline to detect drift on individual batches")
    print("="*60 + "\n")


# Define DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    '01_training_pipeline',
    default_args=default_args,
    description='End-to-end training pipeline for Gas Sensor Drift Detection',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'training', 'gas-sensor', 'drift-detection'],
) as dag:
    
    # Define tasks
    task_load_train = PythonOperator(
        task_id='load_training_data',
        python_callable=load_training_data,
        provide_context=True,
    )
    
    task_load_test = PythonOperator(
        task_id='load_test_data',
        python_callable=load_test_data,
        provide_context=True,
    )
    
    task_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        provide_context=True,
    )
    
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )
    
    task_eval_baseline = PythonOperator(
        task_id='evaluate_on_training',
        python_callable=evaluate_on_training_data,
        provide_context=True,
    )
    
    task_eval_test = PythonOperator(
        task_id='evaluate_on_test',
        python_callable=evaluate_on_test_data,
        provide_context=True,
    )
    
    task_summary = PythonOperator(
        task_id='generate_summary',
        python_callable=generate_training_summary,
        provide_context=True,
    )
    
    # Define task dependencies
    # Load data in parallel
    [task_load_train, task_load_test] >> task_preprocess
    
    # Train after preprocessing
    task_preprocess >> task_train
    
    # Evaluate after training
    task_train >> [task_eval_baseline, task_eval_test]
    
    # Generate summary after all evaluations
    [task_eval_baseline, task_eval_test] >> task_summary
# Gas Sensor Drift Detection - MLOps Pipeline with Apache Airflow

## Project Overview

This project implements an end-to-end MLOps pipeline for detecting drift in gas sensor data using Apache Airflow. The system trains a Random Forest classifier on historical gas sensor readings and monitors for data drift and performance degradation over time.

### Key Features

- **Automated Training Pipeline**: Loads, preprocesses, trains, and evaluates a machine learning model
- **Drift Detection Pipeline**: Monitors model performance and detects statistical drift across multiple batches
- **Statistical Analysis**: Uses KS test, PSI (Population Stability Index), and Wasserstein distance
- **Dockerized Environment**: Full Docker Compose setup for easy deployment
- **Airflow Orchestration**: Two DAGs managing training and monitoring workflows

---

## Project Structure

```
airflow_lab1/
├── dags/
│   ├── 01_training_pipeline.py       # Training workflow DAG
│   └── 02_monitoring_pipeline.py     # Drift monitoring DAG
├── src/
│   ├── data/
│   │   ├── data_loader.py            # Data loading utilities
│   │   └── preprocess.py             # Preprocessing pipeline
│   ├── models/
│   │   ├── train.py                  # Model training
│   │   └── evaluate.py               # Model evaluation
│   └── drift/
│       ├── statistical_test.py       # Statistical drift detection
│       └── performance_monitor.py    # Performance monitoring
├── config/
│   └── config.yaml                   # Configuration file
├── data/
│   ├── raw/                          # Raw sensor data (batches 1-10)
│   ├── processed/                    # Processed datasets
│   └── drift_reports/                # Drift analysis reports
├── models/                           # Trained model artifacts
├── docker-compose.yaml               # Docker orchestration
├── Dockerfile                        # Custom Airflow image
├── .env                              # Environment variables
├── start-airflow.ps1                 # Windows startup script
├── stop-airflow.ps1                  # Windows shutdown script
└── README.md                         # This file
```

---

## Prerequisites

- **Docker Desktop** (Windows/Mac) or Docker Engine (Linux)
- **Docker Compose** v2.0+
- **PowerShell** (for Windows users)
- At least **4GB RAM** and **10GB disk space**

---

## Setup Instructions

### 1. Clone the Repository

```bash
cd c:\Users\suraj\Hariharan\Assignments\Term3\MLOps\mlops_labs\airflow_lab1
```

### 2. Environment Configuration

The `.env` file is already configured with:
```env
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin
```

### 3. Start Airflow

#### Windows (PowerShell)
```powershell
.\start-airflow.ps1
```

#### Linux/Mac
```bash
# Build custom image
docker-compose build

# Initialize Airflow database
docker-compose up airflow-init

# Start services
docker-compose up -d
```

### 4. Access Airflow Web UI

1. Open browser: http://localhost:8080
2. Login credentials:
   - **Username**: `admin`
   - **Password**: `admin`

### 5. Stop Airflow

#### Windows (PowerShell)
```powershell
.\stop-airflow.ps1
```

#### Linux/Mac
```bash
docker-compose down
```

To remove all data including database:
```bash
docker-compose down -v
```

---

## Pipeline Workflows

### 1. Training Pipeline (`01_training_pipeline`)

**Purpose**: Train the initial model on historical data (batches 1-6) and establish baseline metrics.

**Workflow Steps**:
```
Load Training Data (Batches 1-6)
          |
Load Test Data (Batches 7-10)
          |
    Preprocess Data
          |
     Train Model
          |
   ┌──────┴──────┐
   │             │
Evaluate on    Evaluate on
Training Data  Test Data
   │             │
   └──────┬──────┘
          |
  Generate Summary
```

**Tasks**:
1. **load_training_data**: Loads batches 1-6 for training
2. **load_test_data**: Loads batches 7-10 for testing
3. **preprocess_data**: Scales features using StandardScaler
4. **train_model**: Trains Random Forest with cross-validation
5. **evaluate_on_training**: Establishes baseline metrics
6. **evaluate_on_test**: Tests on unseen data
7. **generate_summary**: Creates performance report

**Outputs**:
- Trained model: `models/random_forest_model.pkl`
- Scaler: `models/scaler.pkl`
- Metrics: `data/drift_reports/baseline_metrics.json`

### 2. Monitoring Pipeline (`02_monitoring_pipeline`)

**Purpose**: Monitor individual test batches for drift and performance degradation.

**Workflow Steps** (per batch):
```
Load Batch Data
      |
  Load Model
      |
  Scale Data
      |
Evaluate Performance
      |
  Detect Drift
      |
Monitor Performance
      |
Save Reports
      |
Generate Summary
```

**Dynamic Tasks**: Creates parallel tasks for batches 7, 8, 9, and 10
- `monitor_batch_7`
- `monitor_batch_8`
- `monitor_batch_9`
- `monitor_batch_10`

**Drift Detection Methods**:
- **KS Test**: Kolmogorov-Smirnov test for distribution changes
- **PSI**: Population Stability Index
- **Wasserstein Distance**: Earth Mover's Distance

**Outputs** (per batch):
- Drift reports: `data/drift_reports/drift_report_batch{X}_{timestamp}.json`
- Performance reports: `data/drift_reports/performance_report_batch{X}_{timestamp}.json`

---

## Airflow UI Visualization

### DAG Graph View

#### Training Pipeline
1. Navigate to **DAGs** tab
2. Click on `01_training_pipeline`
3. Click **Graph** button

You'll see:
- **Parallel data loading** (training and test data)
- **Sequential preprocessing and training**
- **Parallel evaluation** (baseline and test)
- **Final summary generation**

![Training Pipeline Structure](docs/training_pipeline_graph.png)

#### Monitoring Pipeline
1. Navigate to **DAGs** tab
2. Click on `02_monitoring_pipeline`
3. Click **Graph** button

You'll see:
- **Four parallel monitoring tasks** (one per batch)
- Each task runs the complete drift detection workflow

![Monitoring Pipeline Structure](docs/monitoring_pipeline_graph.png)

### Triggering DAGs

#### Manual Trigger (Recommended)
1. Go to **DAGs** page
2. Click the **play button** (▶) on the right side of the DAG name
3. Confirm trigger

#### Sequence for First Run
1. **First**: Run `01_training_pipeline` to train the model
2. **Wait**: Until all tasks show green (success)
3. **Then**: Run `02_monitoring_pipeline` to detect drift

### Viewing Task Logs

1. Click on any **task** in the Graph view
2. Click **Log** button
3. View detailed execution logs with:
   - Data shapes and statistics
   - Model performance metrics
   - Drift detection results
   - Error messages (if any)

### Monitoring Execution

**Grid View**: Shows historical runs
- Green: Success
- Red: Failed
- Yellow: Running
- Grey: Not started

**Calendar View**: Visual timeline of DAG runs

**Code View**: View the DAG Python code directly in UI

---

## Understanding the Results

### Training Pipeline Metrics

Located in task logs of `generate_summary`:

```
📊 Data Summary:
   Training Samples: ~XXXXX
   Test Samples:     ~XXXXX

🌲 Model Performance:
   Training Accuracy:       0.XXXX
   CV Accuracy:             0.XXXX
   Baseline Accuracy:       0.XXXX
   Test Accuracy (All):     0.XXXX
```

### Drift Detection Results

Located in `data/drift_reports/drift_report_batch{X}_{timestamp}.json`:

```json
{
  "batch_id": 7,
  "timestamp": "2025-11-03T...",
  "overall_drift_detected": true/false,
  "drift_severity": "low/medium/high",
  "features_with_drift": [...],
  "ks_test_results": {...},
  "psi_results": {...},
  "wasserstein_results": {...}
}
```

**Drift Thresholds**:
- **KS Test p-value**: < 0.05 indicates drift
- **PSI**: > 0.1 indicates drift
- **Wasserstein**: Higher values indicate more drift

### Performance Reports

Located in `data/drift_reports/performance_report_batch{X}_{timestamp}.json`:

```json
{
  "batch_id": 7,
  "metrics": {
    "accuracy": 0.XXXX,
    "f1_macro": 0.XXXX,
    ...
  },
  "performance_degradation_detected": true/false,
  "accuracy_drop": 0.XXX
}
```

---

## Configuration

### Modify Training/Test Split

Edit `config/config.yaml`:

```yaml
data:
  train_batches: [1, 2, 3, 4, 5, 6]
  test_batches: [7, 8, 9, 10]
```

### Adjust Drift Thresholds

Edit `config/config.yaml`:

```yaml
drift_thresholds:
  ks_test_pvalue: 0.05
  psi_threshold: 0.1
  wasserstein_threshold: 0.5
```

### Model Hyperparameters

Edit `config/config.yaml`:

```yaml
model:
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5
  random_state: 42
```

---

## Troubleshooting

### DAGs Not Showing Up

**Issue**: DAGs not visible in Airflow UI

**Solutions**:
1. Check logs: `docker-compose logs airflow-scheduler`
2. Verify DAG files have no syntax errors
3. Refresh the UI (Ctrl+F5)
4. Check file permissions

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Already fixed in DAG files with:
```python
airflow_home = os.getenv('AIRFLOW_HOME', str(pathlib.Path.home() / 'airflow'))
sys.path.insert(0, airflow_home)
```

### Task Failures

**Issue**: Task shows red (failed)

**Solution**:
1. Click on task → **Log**
2. Read error message
3. Common issues:
   - Missing data files in `data/raw/`
   - Configuration errors in `config/config.yaml`
   - Model not trained (run training pipeline first)

### Docker Issues

**Issue**: Containers won't start

**Solutions**:
```powershell
# Rebuild image
docker-compose build --no-cache

# Remove old containers
docker-compose down -v

# Check Docker status
docker ps -a
```

### Performance Issues

**Issue**: Slow execution

**Solutions**:
- Increase Docker memory allocation (Settings → Resources)
- Reduce `n_estimators` in config
- Use fewer batches for testing

---

## Best Practices

### Development Workflow

1. **Make changes** to DAG files locally
2. **Save** - changes auto-sync to Docker (volumes mounted)
3. **Refresh** Airflow UI
4. **Test** by triggering DAG

### Production Considerations

1. **Schedule**: Set `schedule_interval` in DAG definition
   ```python
   schedule_interval='@daily'  # Run daily
   ```

2. **Alerts**: Configure email notifications
   ```python
   default_args = {
       'email': ['your-email@example.com'],
       'email_on_failure': True,
       'email_on_retry': True,
   }
   ```

3. **Retries**: Already configured with 2 retries and 5-minute delay

4. **Monitoring**: Use XCom to pass metrics between tasks

---

## Advanced Features

### Viewing XCom Variables

XCom stores inter-task communication data:

1. **Admin** → **XComs**
2. Filter by DAG ID and Task ID
3. View metrics passed between tasks

### Task Dependencies

Defined using bit-shift operators:

```python
# Sequential
task_a >> task_b >> task_c

# Parallel then merge
[task_a, task_b] >> task_c

# Complex
task_a >> [task_b, task_c] >> task_d
```

### Dynamic Task Generation

The monitoring pipeline uses dynamic task creation:

```python
for batch_id in [7, 8, 9, 10]:
    PythonOperator(
        task_id=f'monitor_batch_{batch_id}',
        python_callable=monitor_batch,
        op_kwargs={'batch_id': batch_id},
    )
```

---

## Data Flow Diagram

```
Raw Data (Batches 1-10)
         |
         v
  Training Pipeline
         |
    ├─ Load Data
    ├─ Preprocess
    ├─ Train Model ──> Saved Model
    └─ Evaluate    ──> Baseline Metrics
         |
         v
  Monitoring Pipeline
         |
    ├─ Load Batch
    ├─ Scale with Saved Scaler
    ├─ Predict with Saved Model
    ├─ Detect Drift ──> Drift Reports
    └─ Monitor Performance ──> Performance Reports
```

---

## Technologies Used

- **Apache Airflow 2.10.4**: Workflow orchestration
- **Python 3.11**: Programming language
- **Docker & Docker Compose**: Containerization
- **PostgreSQL 13**: Airflow metadata database
- **scikit-learn**: Machine learning
- **pandas**: Data manipulation
- **scipy**: Statistical tests
- **PyYAML**: Configuration management

---

## Project Team

- **Owner**: mlops_team
- **Course**: MLOps - Term 3
- **Assignment**: Airflow Lab 1

---

## License

This project is for educational purposes as part of the MLOps course curriculum.

---

## Next Steps

1. **Run Training Pipeline**: Execute `01_training_pipeline`
2. **Review Metrics**: Check logs and generated reports
3. **Run Monitoring**: Execute `02_monitoring_pipeline`
4. **Analyze Drift**: Review drift reports in `data/drift_reports/`
5. **Experiment**: Modify thresholds and retrain

---

## Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Review Airflow logs: `docker-compose logs airflow-scheduler`
3. Check task logs in Airflow UI

---

**Happy MLOps! 🚀**

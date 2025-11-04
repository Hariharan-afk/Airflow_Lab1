FROM apache/airflow:2.10.4-python3.11

# Switch to airflow user for package installation
USER airflow

# Install required Python packages
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    pyyaml \
    numpy

# Switch back to root for any additional setup
USER root

# Create necessary directories
RUN mkdir -p /opt/airflow/data/drift_reports /opt/airflow/data/processed

# Set permissions
RUN chown -R airflow:root /opt/airflow/data

# Switch back to airflow user
USER airflow

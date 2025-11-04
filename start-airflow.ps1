# Airflow Docker Startup Script for Windows
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Apache Airflow with Docker" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Docker is running" -ForegroundColor Green
    } else {
        throw "Docker not running"
    }
} catch {
    Write-Host "[ERROR] Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Yellow
$dirs = @("logs", "plugins", "data/drift_reports", "data/processed")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "[OK] Created $dir" -ForegroundColor Green
    }
}

# Initialize Airflow
Write-Host ""
Write-Host "Initializing Airflow (this may take a few minutes on first run)..." -ForegroundColor Yellow
docker-compose up airflow-init

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Airflow initialized successfully" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Airflow initialization failed" -ForegroundColor Red
    exit 1
}

# Start Airflow services
Write-Host ""
Write-Host "Starting Airflow services..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Airflow UI will be available at:" -ForegroundColor Green
Write-Host "  http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "Login credentials:" -ForegroundColor Green
Write-Host "  Username: admin" -ForegroundColor White
Write-Host "  Password: admin" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

docker-compose up

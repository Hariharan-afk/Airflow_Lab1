# Stop Airflow Docker containers
Write-Host "Stopping Airflow services..." -ForegroundColor Yellow
docker-compose down

Write-Host "[OK] Airflow services stopped" -ForegroundColor Green
Write-Host ""
Write-Host "To remove all data (including database), run:" -ForegroundColor Yellow
Write-Host "  docker-compose down -v" -ForegroundColor White

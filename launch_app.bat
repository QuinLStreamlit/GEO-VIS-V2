@echo off
REM One-click launcher for Windows
REM Double-click this file to launch the Geotechnical Data Analysis Tool

echo Starting Geotechnical Data Analysis Tool...
echo Working directory: %cd%

REM Navigate to the script directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python first.
    echo Download from: https://python.org/downloads
    pause
    exit /b 1
)

echo Python found. Checking Streamlit...

REM Check if streamlit is installed
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing...
    pip install streamlit
    if errorlevel 1 (
        echo ERROR: Failed to install Streamlit
        pause
        exit /b 1
    )
)

REM Check required packages
echo Checking dependencies...
python -c "import streamlit, pandas, numpy, matplotlib, plotly, openpyxl, pyproj" >nul 2>&1
if errorlevel 1 (
    echo Installing missing packages...
    pip install streamlit pandas numpy matplotlib plotly openpyxl scipy watchdog pyproj
    if errorlevel 1 (
        echo WARNING: Some packages may not have installed correctly
    )
)

REM Clean up any existing streamlit processes on port 8501
echo Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8501" ^| find "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Launch the application
echo Launching application...
echo Your browser will open automatically at: http://localhost:8501
echo To stop the application, close this window or press Ctrl+C
echo.

REM Try to open browser automatically
start http://localhost:8501

REM Run streamlit with system Python (Windows compatible) - disable auto browser
python -m streamlit run main_app.py --server.port=8501 --browser.gatherUsageStats=false --server.headless=true

echo Application closed. Thank you for using the Geotechnical Data Analysis Tool!
pause
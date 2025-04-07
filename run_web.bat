@echo off
echo Starting Galatea Web Interface...
echo.

REM Check if Python is installed and in PATH
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH. Please install Python and try again.
    pause
    exit /b 1
)

REM Check if required directories exist, create if not
if not exist templates mkdir templates
if not exist static\css mkdir static\css
if not exist static\js mkdir static\js

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Flask not installed. Installing now...
    pip install flask
)

REM Run the application
echo Starting Flask server...
echo The web interface will be available at http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
python app.py

pause

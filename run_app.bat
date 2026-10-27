@echo off
title AI Image Caption Generator
cd /d "%~dp0"
echo =========================================================
echo    AI Image Caption Generator with Voice Narration
echo =========================================================
echo.
echo Launching Streamlit web application...
echo If your default browser does not open automatically,
echo navigate to: http://localhost:8501
echo.
py -3.13 -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Streamlit exited with error code %ERRORLEVEL%.
    pause
)


@echo off
title Push to GitHub
cd /d "%~dp0"
echo =========================================================
echo    Pushing Image Caption Generator to GitHub
echo =========================================================
echo.
echo Connecting to https://github.com/angelinsherin/image-caption-generator-
echo.
git push -u origin main
echo.
echo =========================================================
if %ERRORLEVEL% EQU 0 (
    echo Successfully pushed to GitHub!
) else (
    echo Push encountered an error. Please review the output above.
)
echo =========================================================
pause

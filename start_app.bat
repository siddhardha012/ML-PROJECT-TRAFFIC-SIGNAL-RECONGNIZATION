@echo off
title Traffic Signal Recognition System
echo ===================================================
echo   Traffic Signal Recognition System - Deep Learning
echo ===================================================
echo Starting Flask Web Server...
echo Opening browser at http://127.0.0.1:5000
timeout /t 2 /nobreak > nul
start http://127.0.0.1:5000
python app.py
pause

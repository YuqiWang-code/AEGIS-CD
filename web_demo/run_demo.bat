@echo off
setlocal
cd /d "%~dp0.."
set PY=C:\Python314\python.exe
if not exist "%PY%" set PY=python

echo ==========================================================
echo   AEGIS-CD  Change Detection Demo
echo   URL: http://127.0.0.1:5000
echo   The browser will open automatically when ready.
echo   Close this window to stop the server.
echo ==========================================================
echo.

"%PY%" web_demo\app.py
pause

@echo off
cd /d "%~dp0"
echo ============================================
echo   DermaScreen server
echo   Open in browser:  http://127.0.0.1:5000
echo   (Close this window to stop the server)
echo ============================================
venv\Scripts\python.exe app.py
pause

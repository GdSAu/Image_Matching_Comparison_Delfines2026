@echo off
setlocal

cd /d "%~dp0"
set PYTHONUNBUFFERED=1
"%~dp0vision_env\Scripts\python.exe" -u "%~dp0gradio_interfaz.py"

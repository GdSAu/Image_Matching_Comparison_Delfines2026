$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$env:PYTHONUNBUFFERED = "1"
& "$PSScriptRoot\vision_env\Scripts\python.exe" -u "$PSScriptRoot\gradio_interfaz.py"

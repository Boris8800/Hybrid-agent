@echo off
rem check_embedding_service.bat - Windows launcher for the PowerShell health-check.
rem Exit code: 0 = healthy, 1 = unhealthy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_embedding_service.ps1" %*
exit /b %ERRORLEVEL%

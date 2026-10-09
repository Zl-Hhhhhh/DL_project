@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 -m object_collage.cli ui
pause

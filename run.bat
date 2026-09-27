@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  if errorlevel 1 goto fail
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 goto fail
python main.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo.
echo Не удалось установить зависимости или запустить приложение.
echo Рекомендуется Python 3.10 или 3.11 на Windows 10/11.
pause
exit /b 1


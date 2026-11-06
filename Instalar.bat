@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto erro
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto erro
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto erro
echo Instalacao concluida. Abra Iniciar_Interface.bat.
pause
exit /b 0
:erro
echo A instalacao falhou. Confira o erro acima e a conexao com a internet.
pause
exit /b 1

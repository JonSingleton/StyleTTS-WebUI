@echo off

@REM set PYTHON=
@REM set GIT=
@REM set VENV_DIR=venv
set COMMANDLINE_ARGS=share=True
@REM --no-half

call venv\Scripts\activate
.\venv\Scripts\python.exe webui.py

cmd /k
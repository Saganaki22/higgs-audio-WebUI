@echo off
REM Batch script to launch Higgs Audio WebUI

REM Activate conda environment
CALL conda activate higgs_audio_env
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate conda environment 'higgs_audio_env'.
    echo Make sure Anaconda/Miniconda is installed and the environment exists.
    pause
    exit /b 1
)

REM Run the Gradio app
python higgs_audio_gradio.py

REM Keep the window open after execution
echo.
echo [INFO] The app has exited. If you see errors above, please review them.
pause
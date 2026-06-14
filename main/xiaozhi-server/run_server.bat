@echo off
::set TEMP=D:\Files\xiaozhi-esp32-server\main\xiaozhi-server\temp_for_xiaozhi_server
::set TMP=D:\Files\xiaozhi-esp32-server\main\xiaozhi-server\temp_for_xiaozhi_server

::if not exist "%TEMP%" mkdir "%TEMP%"

echo TEMP=%TEMP%
echo TMP=%TMP%
echo.

call conda activate xiaozhi-esp32-server

cd /d D:\Files\xiaozhi-esp32-server\main\xiaozhi-server

python app.py

pause
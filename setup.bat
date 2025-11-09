@echo off
echo ========================================
echo      ValBot CLI Setup Script
echo ========================================
echo.

:: Define allowed Python versions: >= 3.12 and < 3.14
set MIN_PY_MAJOR=3
set MIN_PY_MINOR=12
set MAX_PY_MAJOR=3
set MAX_PY_MINOR=13
chcp 65001 >nul

call :check_python_version
if %errorlevel% neq 0 (
    echo Setup aborted due to Python version requirement not met.
    pause
    exit /b 1
)

call :check_env_file

echo ✅  Python %PY_VER% found! Proceeding with setup...
echo.

:: Create virtual environment
echo Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

"%PY_EXE%" %PY_ARGS% -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅  Virtual environment created successfully!
echo.

:: Prompt for proxy before pip operations
call :prompt_proxy

:: Upgrade virtual environment's pip
echo Upgrading pip...
if "%USE_PROXY%"=="1" (
    venv\Scripts\python.exe -m pip install --proxy "%PROXY_URL%" --upgrade pip --no-cache-dir
) else (
    venv\Scripts\python.exe -m pip install --upgrade pip --no-cache-dir
)

:: Install requirements
echo Installing requirements...
if "%USE_PROXY%"=="1" (
    venv\Scripts\pip.exe install --proxy "%PROXY_URL%" -r requirements.txt --disable-pip-version-check --upgrade --no-cache-dir
) else (
    venv\Scripts\pip.exe install -r requirements.txt --disable-pip-version-check --upgrade --no-cache-dir
)
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo Please check your requirements.txt file and internet connection
    pause
    exit /b 1
)

call :build_valbot_exe

echo ========================================
echo        Setup Complete!
echo ========================================
echo.
echo IMPORTANT: You need to configure your VALBOT_CLI_KEY before running the bot.
echo Set VALBOT_CLI_KEY in the .env file (created in this directory).
echo.
echo ========================================
echo     How to start the ValBot CLI:
echo ========================================
echo.
echo 1. Run the bot with a message:
echo    .\valbot.bat "Your message here"
echo.
echo 2. Run with context files:
echo    .\valbot.bat -m "Your message" -c file1.txt file2.txt
echo.
echo 3. Run with an agent:
echo    .\valbot.bat -a agent_name -p param1=value1 param2=value2
echo.
echo 4. Use a custom config file:
echo    .\valbot.bat --config your_config.json "Your message"
echo.
echo Available command line options:
echo   positional_message    Initial message (positional argument)
echo   -m, --message        Initial message to send to the AI
echo   -c, --context        File(s) to load context from
echo   -a, --agent          Agent flow to invoke at startup
echo   -p, --params         Parameters for agent (key=value format)
echo   --config             Path to custom configuration file
echo.
echo Examples:
echo   .\valbot.bat "Hello, how can you help me?"
echo   .\valbot.bat -m "Analyze this code" -c myfile.py
echo   .\valbot.bat -a file_edit_agent -p file=example.py action=review
echo.
echo ========================================
echo.
pause
echo.
echo NOTE: If you started this setup by double-clicking the file, the window will close after you press any key.
echo To start using ValBot CLI, open a Command Prompt or PowerShell window in this directory.
echo.
echo To open a terminal here:
echo   - In File Explorer, right-click inside this folder and select 'Open in Terminal' or 'Open Command Window Here'.
echo   - Or, in File Explorer, press CTRL+L and type 'cmd' or 'powershell' in the address bar and press Enter.
echo.
echo Then, run the commands listed above to use ValBot CLI.
echo.
goto :eof

:prompt_proxy
REM Ask the user if they want to use a proxy for pip operations
set "USE_PROXY=0"
set "PROXY_URL=http://proxy-chain.intel.com:911"

echo Are you behind a corporate proxy/firewall and want to use it for pip installs? [Y/n]
set "ANS="
set /p ANS=
if not defined ANS set ANS=Y

if /I "%ANS%"=="Y" (
    echo Enter proxy URL [%PROXY_URL%]:
    set "INPUT="
    set /p INPUT=
    if defined INPUT (
        set "PROXY_URL=%INPUT%"
    )
    REM Set common proxy environment variables for completeness
    set HTTP_PROXY=%PROXY_URL%
    set HTTPS_PROXY=%PROXY_URL%
    set http_proxy=%PROXY_URL%
    set https_proxy=%PROXY_URL%
    set "USE_PROXY=1"
    echo Using proxy: %PROXY_URL% for pip installs
) else (
    echo Proceeding without proxy.
)
goto :eof

:build_valbot_exe
REM Build executable with PyInstaller and copy to user's bin directory
echo.
echo Building standalone executable with PyInstaller...

REM Get current version and repo path from git and save to build_info.json
echo Getting build information...
for /f "delims=" %%i in ('git describe --tags 2^>nul') do set VERSION=%%i
if "%VERSION%"=="" (
    for /f "delims=" %%i in ('git rev-parse --short HEAD 2^>nul') do set VERSION=%%i
)
if "%VERSION%"=="" set VERSION=unknown
set REPO_PATH=%CD%
echo Version: %VERSION%
echo Repo path: %REPO_PATH%
(
    echo {
    echo   "version": "%VERSION%",
    echo   "repo_path": "%REPO_PATH:\=\\%"
    echo }
) > build_info.json

set "ADD_DATA="
if exist "user_config.json" set ADD_DATA=%ADD_DATA% --add-data "user_config.json;."
if exist "default_config.json" set ADD_DATA=%ADD_DATA% --add-data "default_config.json;."
if exist "valbot_config.json" set ADD_DATA=%ADD_DATA% --add-data "valbot_config.json;."
set ADD_DATA=%ADD_DATA% --add-data ".env;."
set ADD_DATA=%ADD_DATA% --add-data "agent_plugins;agent_plugins"
set ADD_DATA=%ADD_DATA% --add-data "build_info.json;."

venv\Scripts\python.exe -m PyInstaller --onefile --name valbot %ADD_DATA% --collect-all readchar --copy-metadata pydantic-ai-slim --copy-metadata pydantic-ai --hidden-import pydantic_ai app.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to build executable with PyInstaller.
    pause
    exit /b 1
)

set "USERBIN=%USERPROFILE%\bin"
if not exist "%USERBIN%" (
    mkdir "%USERBIN%"
)
copy /Y dist\valbot.exe "%USERBIN%\valbot.exe"
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy valbot.exe to %USERBIN%
    pause
    exit /b 1
)
echo ✅  valbot.exe has been placed in %USERBIN%
echo.
echo Add %USERBIN% to your PATH if you want to run 'valbot' from anywhere.
echo.
goto :eof

:check_python_version
REM Find a compatible Python interpreter: >= %MIN_PY_MAJOR%.%MIN_PY_MINOR% and < %MAX_PY_MAJOR%.%MAX_PY_MINOR%+1 (i.e., less than 3.14)
set "PY_EXE="
set "PY_ARGS="

REM Prefer the Windows Python launcher (py) for specific minors, highest first
for /L %%m in (%MAX_PY_MINOR%,-1,%MIN_PY_MINOR%) do (
    py -3.%%m -V >nul 2>&1
    if not errorlevel 1 (
        set "PY_EXE=py"
        set "PY_ARGS=-3.%%m"
        goto :found_py
    )
)

REM Try named executables on PATH next
for /L %%m in (%MAX_PY_MINOR%,-1,%MIN_PY_MINOR%) do (
    where python3.%%m.exe >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('where python3.%%m.exe') do (
            set "PY_EXE=%%P"
            set "PY_ARGS="
            goto :found_py
        )
    )
)

REM Fallback to default 'python' and validate its version is in range
python --version >tmp_pyver.txt 2>&1
if %errorlevel% neq 0 (
    echo ERROR: No compatible Python interpreter found on PATH.
    echo Please install Python between %MIN_PY_MAJOR%.%MIN_PY_MINOR% and %MAX_PY_MAJOR%.%MAX_PY_MINOR% ^(less than 3.14^).
    echo Recommended: Install Python 3.13.x or 3.12.x from https://python.org/downloads/windows/
    echo If you have the Python launcher 'py', you can use it to manage multiple versions.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in (tmp_pyver.txt) do set PY_VER=%%v
del tmp_pyver.txt
for /f "tokens=1-3 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
    set PY_PATCH=%%c
)
if %PY_MAJOR% LSS %MIN_PY_MAJOR% goto :py_out_of_range
if %PY_MAJOR%==%MIN_PY_MAJOR% if %PY_MINOR% LSS %MIN_PY_MINOR% goto :py_out_of_range
if %PY_MAJOR% GTR %MAX_PY_MAJOR% goto :py_out_of_range
if %PY_MAJOR%==%MAX_PY_MAJOR% if %PY_MINOR% GTR %MAX_PY_MINOR% goto :py_out_of_range
set "PY_EXE=python"
set "PY_ARGS="
goto :ver_ready

:found_py
"%PY_EXE%" %PY_ARGS% --version >tmp_pyver.txt 2>&1
for /f "tokens=2 delims= " %%v in (tmp_pyver.txt) do set PY_VER=%%v
del tmp_pyver.txt

:ver_ready
goto :eof

:py_out_of_range
echo ERROR: Found Python %PY_VER%, but a version between %MIN_PY_MAJOR%.%MIN_PY_MINOR% and %MAX_PY_MAJOR%.%MAX_PY_MINOR% ^(less than 3.14^) is required.
echo Please install Python 3.13.x or 3.12.x from https://www.python.org/downloads/release/python-3138/
pause
exit /b 1

:check_env_file
REM Check for .env file and create with VALBOT_CLI_KEY if missing
if exist ".env" (
    echo ✅  .env file already exists. Ensure VALBOT_CLI_KEY is set.
) else (
    echo Creating .env file...
    echo You need a VALBOT_CLI_KEY to use ValBot CLI. You can obtain this key by following the instructions at:
    echo https://github.com/intel-innersource/applications.ai.valbot-cli?tab=readme-ov-file#setup
    echo Once you have the key, you can enter it now.
    echo.
    set "VALBOT_CLI_KEY_INPUT="
    set /p VALBOT_CLI_KEY_INPUT=Enter your VALBOT_CLI_KEY ^(press Enter to use placeholder^):
    if defined VALBOT_CLI_KEY_INPUT (
        > .env call echo VALBOT_CLI_KEY=%%VALBOT_CLI_KEY_INPUT%%
        echo ✅  .env file created with provided VALBOT_CLI_KEY.
    ) else (
        > .env echo VALBOT_CLI_KEY=your_valbot_cli_key_here
        echo ✅  .env file created with placeholder VALBOT_CLI_KEY. You can update it later.
    )
)
goto :eof

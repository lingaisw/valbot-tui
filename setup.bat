@echo off
setlocal enabledelayedexpansion
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

:: Create or use virtual environment
call :choose_or_create_venv

:: Prompt for proxy before pip operations
call :prompt_proxy

:: Upgrade virtual environment's pip
echo Upgrading pip...
if "%USE_PROXY%"=="1" (
    "%VENV_PATH%\Scripts\python.exe" -m pip install --proxy "%PROXY_URL%" --upgrade pip --no-cache-dir
) else (
    "%VENV_PATH%\Scripts\python.exe" -m pip install --upgrade pip --no-cache-dir
)

:: Install requirements
echo Installing requirements...
if "%USE_PROXY%"=="1" (
    "%VENV_PATH%\Scripts\pip.exe" install --proxy "%PROXY_URL%" -r requirements.txt --disable-pip-version-check --upgrade --no-cache-dir
) else (
    "%VENV_PATH%\Scripts\pip.exe" install -r requirements.txt --disable-pip-version-check --upgrade --no-cache-dir
)
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo Please check your requirements.txt file and internet connection
    pause
    exit /b 1
)

call :create_tui_launcher
call :build_valbot_exe

echo ========================================
echo        Setup Complete!
echo ========================================
echo.
echo IMPORTANT: You need to configure your VALBOT_CLI_KEY before running the bot.
echo Set VALBOT_CLI_KEY in the .env file (created in this directory).
echo.
echo ========================================
echo     How to start ValBot:
echo ========================================
echo.
echo 1. Run the TUI (Terminal User Interface) - Recommended:
echo    .\valbot_tui.bat
echo.
echo 2. Run the CLI with a message:
echo    .\valbot.bat "Your message here"
echo.
echo 3. Run with context files:
echo    .\valbot.bat -m "Your message" -c file1.txt file2.txt
echo.
echo 4. Run with an agent:
echo    .\valbot.bat -a agent_name -p param1=value1 param2=value2
echo.
echo 5. Use a custom config file:
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
echo   .\valbot_tui.bat
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

:create_tui_launcher
REM Create valbot_tui.bat launcher script
echo.
echo Creating ValBot TUI launcher script...
set "TUI_LAUNCHER=%CD%\valbot_tui.bat"

(
echo @echo off
echo REM ValBot TUI Launcher for Windows
echo REM This script launches the Terminal User Interface version of ValBot
echo.
echo setlocal enabledelayedexpansion
echo.
echo REM Check if Python is available
echo python --version ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo Error: Python is not installed or not in PATH
echo     echo Please install Python 3.11+ and try again
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Get the directory where this script is located
echo set SCRIPT_DIR=%%~dp0
echo.
echo REM Don't change to script directory - stay in caller's directory
echo REM cd /d "%%SCRIPT_DIR%%"
echo.
echo REM List of paths to check for virtual environment
echo REM You can manually add additional paths to this list ^(space-separated^)
echo set "VENV_PATHS=%VENV_PATH% %%SCRIPT_DIR%%venv %%SCRIPT_DIR%%valbot-venv %%SCRIPT_DIR%%.venv"
echo.
echo REM Check for virtual environment in the listed paths
echo set VENV_FOUND=0
echo set FINAL_VENV_PATH=
echo.
echo for %%%%P in ^(%%VENV_PATHS%%^) do ^(
echo     if exist "%%%%P\Scripts\activate.bat" ^(
echo         set "FINAL_VENV_PATH=%%%%P"
echo         set VENV_FOUND=1
echo         goto :venv_found
echo     ^)
echo ^)
echo.
echo :venv_found
echo if %%VENV_FOUND%%==1 ^(
echo     echo Activating virtual environment at %%FINAL_VENV_PATH%%...
echo     call "%%FINAL_VENV_PATH%%\Scripts\activate.bat"
echo ^) else ^(
echo     echo Warning: Virtual environment not found
echo     echo Running with system Python...
echo ^)
echo.
echo REM Launch the TUI
echo echo Starting ValBot TUI...
echo python "%%SCRIPT_DIR%%valbot_tui_launcher.py" %%*
echo.
echo REM Deactivate virtual environment if it was activated
echo if defined VIRTUAL_ENV ^(
echo     deactivate
echo ^)
echo.
echo endlocal
) > "%TUI_LAUNCHER%"

echo ✅  Created valbot_tui.bat launcher
echo.
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

"%VENV_PATH%\Scripts\python.exe" -m PyInstaller --onefile --name valbot %ADD_DATA% --collect-all readchar --copy-metadata pydantic-ai-slim --copy-metadata pydantic-ai --hidden-import pydantic_ai app.py
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
    echo ✅  .env file already exists.
    REM Try to extract existing key
    for /f "usebackq tokens=2 delims==" %%k in (".env") do (
        set "EXISTING_KEY=%%k"
        goto :check_existing_key
    )
    :check_existing_key
    if defined EXISTING_KEY (
        if not "%EXISTING_KEY%"=="your_valbot_cli_key_here" (
            echo Found existing VALBOT_CLI_KEY in .env file.
            set /p "USE_EXISTING=Use the existing key from .env file? [Y/n]: "
            if /I "!USE_EXISTING!"=="n" (
                echo You can enter a new key below.
                goto :prompt_new_key
            ) else (
                echo ✅  Using existing key from .env file.
                goto :eof
            )
        )
    )
    echo Ensure VALBOT_CLI_KEY is set.
) else (
    :prompt_new_key
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

:choose_or_create_venv
REM Create a new virtual environment or use an existing one
echo.
echo Create a new virtual environment or use an existing one. ^(Default: create new^)
set /p "CREATE_NEW=Create a new virtual environment now? [Y/n]: "
if /I "%CREATE_NEW%"=="n" (
    goto :use_existing_venv
)

:prompt_venv_path
set "DEFAULT_VENV=%CD%\venv"
set /p "VENV_INPUT=Enter a path where we should create the venv [%DEFAULT_VENV%]: "
if not defined VENV_INPUT set "VENV_INPUT=%DEFAULT_VENV%"
set "VENV_PATH=%VENV_INPUT%"

REM Check if path exists and is not a directory
if exist "%VENV_PATH%" (
    if not exist "%VENV_PATH%\*" (
        echo ERROR: Path exists and is not a directory: %VENV_PATH%
        goto :prompt_venv_path
    )
)

REM Check if directory exists and is not empty
if exist "%VENV_PATH%\*" (
    dir /b "%VENV_PATH%" | findstr "^" >nul
    if not errorlevel 1 (
        echo Directory '%VENV_PATH%' exists and is not empty.
        :ask_action
        set /p "ACTION=How would you like to proceed? [U]se as is, [D]elete and create new, or [C]hoose a different path [u/d/c]: "
        if /I "%ACTION%"=="u" (
            REM Check if activate script exists
            if exist "%VENV_PATH%\Scripts\activate.bat" (
                echo ✅  Using existing venv: %VENV_PATH%
                goto :eof
            ) else (
                echo WARNING: No activate script found at '%VENV_PATH%\Scripts\activate.bat'. This may not be a Python virtual environment.
                set /p "CONTINUE_ANYWAY=Continue anyway and use this directory as is? [y/N]: "
                if /I "!CONTINUE_ANYWAY!"=="y" (
                    echo ✅  Using existing directory as venv: %VENV_PATH%
                    goto :eof
                ) else (
                    goto :ask_action
                )
            )
        ) else if /I "%ACTION%"=="d" (
            set /p "CONFIRM_DELETE=Really delete '%VENV_PATH%' and create a new venv here? This will remove ALL contents. [y/N]: "
            if /I "!CONFIRM_DELETE!"=="y" (
                echo Deleting existing directory...
                rmdir /s /q "%VENV_PATH%"
                goto :create_new_venv
            ) else (
                goto :ask_action
            )
        ) else if /I "%ACTION%"=="c" (
            goto :prompt_venv_path
        ) else (
            echo Please enter 'u' ^(use^), 'd' ^(delete and create new^), or 'c' ^(choose a different path^).
            goto :ask_action
        )
    )
)

:create_new_venv
echo Creating virtual environment at: %VENV_PATH%
"%PY_EXE%" %PY_ARGS% -m venv "%VENV_PATH%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✅  Virtual environment created successfully!
echo.
goto :eof

:use_existing_venv
set "DEFAULT_VENV=%CD%\venv"
:prompt_existing_venv
set /p "VENV_INPUT=Enter existing venv path [%DEFAULT_VENV%]: "
if not defined VENV_INPUT set "VENV_INPUT=%DEFAULT_VENV%"
set "VENV_PATH=%VENV_INPUT%"

if exist "%VENV_PATH%\Scripts\activate.bat" (
    echo ✅  Using existing venv: %VENV_PATH%
    echo.
    goto :eof
) else (
    echo ERROR: No activate script found at '%VENV_PATH%\Scripts\activate.bat'. Please provide a valid venv path.
    goto :prompt_existing_venv
)
goto :eof

@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Cleaning up build directory and rebuilding
echo ============================================

:: MinGWのパスを最優先に設定
set PATH=C:\Program Files\mingw64\bin;%PATH%

:: プロジェクトルートディレクトリを設定
set PROJECT_DIR=C:\Users\Mizuki\University\Reserch\reservoir\reservoir_simulation\reservoir_motor

cd %PROJECT_DIR%

:: ビルドディレクトリのパス
set BUILD_DIR=%PROJECT_DIR%\build

:: ビルドディレクトリが存在する場合は削除
if exist %BUILD_DIR% (
    echo Removing build directory...
    rmdir /s /q %BUILD_DIR%
)

:: 新しいビルドディレクトリを作成
echo Creating new build directory...
mkdir %BUILD_DIR%

:: ビルドディレクトリに移動
cd %BUILD_DIR%

:: CMakeの実行（ソースディレクトリを明示的に指定）
echo Running CMake configuration...
cmake %PROJECT_DIR% -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

:: ビルドの実行
if %errorlevel% neq 0 (
    echo CMake configuration failed. Exiting...
    exit /b 1
)

echo Building project...
mingw32-make

if %errorlevel% neq 0 (
    echo Build failed. Exiting...
    exit /b 1
)

echo ============================================
echo Build completed successfully!
echo ============================================

cd %BUILD_DIR%

pause

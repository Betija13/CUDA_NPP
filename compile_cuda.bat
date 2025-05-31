@echo off
echo Setting up Visual Studio environment for CUDA...

:: Add Visual Studio compiler to PATH
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
set "PATH=%VS_PATH%;%PATH%"

:: Add Windows SDK to PATH (if needed)
set "WIN_SDK=C:\Program Files (x86)\Windows Kits\10\bin"
if exist "%WIN_SDK%" set "PATH=%WIN_SDK%;%PATH%"

:: Verify cl.exe is in PATH
where cl.exe
if %ERRORLEVEL% neq 0 (
    echo ERROR: cl.exe not found in PATH even after setting environment variables.
    echo Please make sure Visual Studio with C++ development tools is properly installed.
    exit /b 1
)

:: Compile the CUDA program
echo Compiling CUDA program...
nvcc -o main.exe main.cu kernels\grayscale.cu kernels\flip.cu kernels\gaussian_blur.cu kernels\sobel.cu kernels\threshold.cu kernels\threshold_r.cu kernels\rotation.cu kernels\brightness.cu kernels\contrast.cu kernels\g_corr.cu kernels\lut.cu kernels\statistics.cu include\image_utils.c -Iinclude -lcudart -lnppc -lnppial -lnppicc -lnppig -lnppif -lnppitc -lnppidei -lnppist

if %ERRORLEVEL% equ 0 (
    echo Compilation successful! Running the program...
    if "%2"=="grayscale" (
        main.exe %1 grayscale
    ) else if "%2"=="flip_horizontal" (
        main.exe %1 flip_horizontal
    ) else if "%2"=="flip_vertical" (
        main.exe %1 flip_vertical
    ) else if "%2"=="flip_both" (
        main.exe %1 flip_both
    ) else if "%2"=="sobel" (
        main.exe %1 sobel
    ) else if "%2"=="g_blur" (
        main.exe %1 g_blur %3
    ) else if "%2"=="threshold" (
        main.exe %1 threshold %3
    ) else if "%2"=="threshold_r" (
        main.exe %1 threshold_r %3
    ) else if "%2"=="rotation" (
        main.exe %1 rotation %3
    ) else if "%2"=="brightness" (
        main.exe %1 brightness %3
    ) else if "%2"=="contrast" (
        main.exe %1 contrast %3
    ) else if "%2"=="lut" (
        main.exe %1 lut %3
    ) else if "%2"=="g_corr" (
        main.exe %1 g_corr
    ) else if "%2"=="statistics" (
        main.exe %1 statistics
    ) else if "%2"=="all" (
        echo ===GOING THROUGH ALL OPTIONS===
        main.exe %1 grayscale
        main.exe %1 flip_horizontal
        main.exe %1 flip_vertical
        main.exe %1 flip_both
        main.exe %1 sobel
        main.exe %1 g_blur 33
        main.exe %1 g_blur 77
        main.exe %1 g_blur 1515
        main.exe %1 threshold
        main.exe %1 threshold_r
        main.exe %1 rotation
        main.exe %1 brightness
        main.exe %1 contrast
        main.exe %1 lut
        main.exe %1 g_corr
        echo ===STATISTICS FOR GRAYSCALE===
        main.exe test_images/output_grayscale.png statistics
        echo ===STATISTICS FOR ORIGINAL===
        main.exe %1 statistics
        
    ) else (
        echo Unknown or missing argument. Usage: compile_cuda.bat [inputImage] [grayscale|flip_horizontal|flip_vertical|flip_both|g_blur|sobel|threshold|threshold_r|rotation|brightness|contrast|g_corr|lut|statistics] [number]
    )
) else (
    echo Compilation failed with error code %ERRORLEVEL%
)

pause

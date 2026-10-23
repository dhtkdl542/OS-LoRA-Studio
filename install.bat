@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [OS LoRA Studio] 설치를 시작합니다.
echo.

rem ---- Python 3.12 (python.org 설치본: tkinter 포함, py 런처 사용)
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python 3.12를 찾지 못했습니다.
    echo        https://www.python.org/downloads/release/python-3120/ 에서 Windows installer 64-bit를 받아
    echo        설치할 때 "py launcher"와 "tcl/tk" 항목을 켜 두고 설치한 뒤 다시 실행하세요.
    goto :error
)

rem ---- NVIDIA GPU / 드라이버
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [오류] NVIDIA GPU 드라이버를 찾지 못했습니다. NVIDIA GPU와 최신 드라이버가 필요합니다.
    echo        https://www.nvidia.com/Download/index.aspx
    goto :error
)
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name^,memory.total^,driver_version --format^=csv^,noheader') do echo GPU: %%g
echo   (CUDA 12.8 빌드를 씁니다. 드라이버 570 이상이 필요하고, VRAM 16GB 기준으로 맞춰져 있어요)
echo.

echo [1/3] 가상환경 생성 (Python 3.12)
if not exist venv (
    py -3.12 -m venv venv || goto :error
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [2/3] PyTorch (CUDA 12.8) 설치
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 || goto :error

echo [3/3] 나머지 패키지 설치
pip install -r requirements.txt || goto :error

python -c "import torch; assert torch.cuda.is_available(), 'CUDA를 쓸 수 없습니다'; print('CUDA 확인:', torch.cuda.get_device_name(0))" || goto :error

echo.
echo 설치 완료! run.bat 으로 실행하세요.
echo 처음 실행하면 학습 설정 ^> 모델 파일에서 MOSS 모델 파일 4개의 경로를 지정하세요.
echo 파일이 없다면 각 칸 옆의 ⬇ 버튼으로 받을 수 있어요 (README.md 참고).
pause
exit /b 0

:error
echo.
echo 설치 중 오류가 발생했습니다. 위 메시지를 확인하세요.
pause
exit /b 1

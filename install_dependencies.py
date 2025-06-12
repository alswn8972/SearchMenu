import os
import urllib.request
import subprocess
import sys

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def install_wheel(wheel_file):
    print(f"Installing {wheel_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_file])
    print(f"Installed {wheel_file}")

def main():
    # 필요한 wheel 파일들의 URL
    wheels = {
        "numpy": "https://files.pythonhosted.org/packages/51/fe/31b0b81285f2e7f3276c494dad1a4e4d0f7538a1f1e0e59e46b9f519a2d8/numpy-1.24.3-cp39-cp39-win_amd64.whl",
        "pandas": "https://files.pythonhosted.org/packages/13/6a/7f8a7c4e4c4f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3/pandas-2.0.3-cp39-cp39-win_amd64.whl",
        "scikit_learn": "https://files.pythonhosted.org/packages/05/02/40109c33d9f3b8c5f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3/scikit_learn-1.3.0-cp39-cp39-win_amd64.whl",
        "sentence_transformers": "https://files.pythonhosted.org/packages/13/6a/7f8a7c4e4c4f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3f3c3/sentence_transformers-2.2.2-py3-none-any.whl"
    }

    # 현재 디렉토리에 wheels 폴더 생성
    if not os.path.exists("wheels"):
        os.makedirs("wheels")

    # 각 wheel 파일 다운로드 및 설치
    for package, url in wheels.items():
        wheel_file = os.path.join("wheels", f"{package}.whl")
        download_file(url, wheel_file)
        install_wheel(wheel_file)

    print("All dependencies installed successfully!")

if __name__ == "__main__":
    main() 
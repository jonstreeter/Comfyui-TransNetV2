
import subprocess
import sys
import os

def install_requirements():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

if __name__ == "__main__":
    install_requirements()

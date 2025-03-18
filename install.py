import os
import subprocess
import sys
import torch

def install_requirements():
    """ Installs necessary Python packages """
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

def convert_tf_to_pytorch():
    """ Converts TensorFlow weights to PyTorch if necessary """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, "inference", "transnetv2-weights")
    tf_model_path = os.path.join(weights_dir, "saved_model.pb")
    pytorch_weights_path = os.path.join(weights_dir, "transnetv2-pytorch-weights.pth")

    if os.path.exists(pytorch_weights_path):
        print("✅ PyTorch weights already exist. Skipping conversion.")
        return
    
    if not os.path.exists(tf_model_path):
        print(f"❌ TensorFlow weights not found at {tf_model_path}. Cannot convert.")
        return

    print("🔄 Converting TensorFlow weights to PyTorch...")

    convert_script = os.path.join(current_dir, "inference-pytorch", "convert_weights.py")

    if not os.path.exists(convert_script):
        print(f"❌ Conversion script not found at {convert_script}. Make sure it exists.")
        return

    try:
        subprocess.check_call([sys.executable, convert_script, "--tf_weights", weights_dir])
        print("✅ Conversion successful. PyTorch weights saved.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")

if __name__ == "__main__":
    print("🔧 Installing TransNetV2 for ComfyUI...")
    install_requirements()
    convert_tf_to_pytorch()
    print("✅ Installation complete.")

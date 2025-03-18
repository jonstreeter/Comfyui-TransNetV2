import os
import sys
import torch
import importlib.util
from moviepy.editor import VideoFileClip

# Explicitly load the correct PyTorch TransNetV2 module
current_dir = os.path.dirname(os.path.abspath(__file__))
transnet_module_path = os.path.join(current_dir, "inference-pytorch", "transnetv2_pytorch.py")

if not os.path.exists(transnet_module_path):
    raise FileNotFoundError(f"Could not find {transnet_module_path}, ensure it exists!")

spec = importlib.util.spec_from_file_location("transnetv2_pytorch", transnet_module_path)
transnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transnet_module)

TransNetV2 = transnet_module.TransNetV2

class TransNetV2Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "output_type": (["timecodes", "split_videos"],),
                "output_folder": ("STRING", {"default": "output/transnet_scenes", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_video"
    CATEGORY = "video"

    def process_video(self, video_path, threshold, output_type, output_folder):
        if not os.path.isfile(video_path):
            return (f"Error: Video not found at {video_path}",)

        os.makedirs(output_folder, exist_ok=True)

        # Ensure weights path is correctly set to lowercase folder name
        weights_path = os.path.join(current_dir, "inference", "transnetv2-weights")

        # Initialize PyTorch TransNetV2 model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransNetV2(weights_path=weights_path).to(device)

        # Predict shot boundaries
        predictions, fps = model.predict_video(video_path)
        scenes = model.predictions_to_scenes(predictions, threshold=threshold)

        if output_type == "timecodes":
            timecodes = "Detected Scenes:\n"
            for i, (start, end) in enumerate(scenes):
                start_sec = start / fps
                end_sec = end / fps
                timecodes += f"Scene {i+1}: {start_sec:.2f}s to {end_sec:.2f}s\n"
            return (timecodes,)

        elif output_type == "split_videos":
            clip = VideoFileClip(video_path)
            for i, (start, end) in enumerate(scenes):
                start_sec = start / fps
                end_sec = end / fps
                subclip = clip.subclip(start_sec, end_sec)
                output_file = os.path.join(output_folder, f"scene_{i+1:03d}.mp4")
                subclip.write_videofile(output_file, codec="libx264")
            return (f"Video successfully split into scenes at {output_folder}",)

        return ("Unexpected Error",)

NODE_CLASS_MAPPINGS = {
    "TransNetV2Node": TransNetV2Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransNetV2Node": "TransNetV2 Video Processor"
}

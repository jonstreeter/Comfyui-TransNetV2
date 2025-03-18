import os
import sys

# Ensure TransNetV2 can be imported from root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from transnetv2 import TransNetV2
from transnetv2_utils import draw_video_with_predictions, scenes_from_predictions
import tensorflow as tf

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

        # Initialize TransNetV2 model
        model = TransNetV2()

        # Predict shot boundaries
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
        scenes = scenes_from_predictions(single_frame_predictions, threshold=threshold)

        if output_type == "timecodes":
            timecodes = "Detected Scenes:\n"
            for i, (start, end) in enumerate(scenes):
                start_sec = start / model.frame_rate
                end_sec = end / model.frame_rate
                timecodes += f"Scene {i+1}: {start_sec:.2f}s to {end_sec:.2f}s\n"
            return (timecodes,)

        elif output_type == "split_videos":
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            for i, (start, end) in enumerate(scenes):
                start_sec = start / model.frame_rate
                end_sec = end / model.frame_rate
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

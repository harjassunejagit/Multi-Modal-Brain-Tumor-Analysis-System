import sys
import os

# Ensure the parent folder is in sys.path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from brain_tumor_ai.train import train_model
    from brain_tumor_ai.infer import run_inference
    from brain_tumor_ai.utils import load_config
except ModuleNotFoundError:
    # fallback if running from root directly
    from train import train_model
    from infer import run_inference
    from utils import load_config

def main():
    config_path = os.path.join(current_dir, "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(root_dir, "brain_tumor_ai", "config.yaml")

    config = load_config(config_path)
    mode = config.get("mode", "train")  # default to train

    if mode.lower() == "train":
        print("[INFO] Starting training...")
        train_model(config)
    elif mode.lower() == "inference":
        print("[INFO] Running inference...")
        run_inference(config)
    else:
        print(f"[ERROR] Unknown mode '{mode}' in config.yaml")

if __name__ == "__main__":
    main()
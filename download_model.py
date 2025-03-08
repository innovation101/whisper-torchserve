import whisper
import os
import torch
import sys

# Set model parameters
MODEL_SIZE = "base"  # Options: "tiny", "base", "small", "medium", "large"

# Define model directory - always use a local path for safety
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper-models")

def main():
    print(f"Downloading whisper {MODEL_SIZE} model...")
    
    # We need to access the global MODEL_DIR variable
    global MODEL_DIR
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Created directory: {MODEL_DIR}")
    except PermissionError:
        print(f"Permission denied when creating {MODEL_DIR}")
        print("Using current directory instead")
        MODEL_DIR = os.path.join(os.getcwd(), "whisper-models")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
    # Check CUDA availability for logging
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}, using {device}")
    
    # Download the model
    print(f"Starting model download to {MODEL_DIR}...")
    model = whisper.load_model(MODEL_SIZE, device="cpu")
    
    # Save model to disk
    model_path = os.path.join(MODEL_DIR, f"whisper_{MODEL_SIZE}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
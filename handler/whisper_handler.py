import os
import torch
import torchaudio
import whisper
import numpy as np
import soundfile as sf
import logging
from io import BytesIO
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class WhisperHandler(BaseHandler):
    def __init__(self):
        super(WhisperHandler, self).__init__()
        self.model = None
        self.initialized = False
        self.model_size = "base"

    def initialize(self, context):
        """
        Initialize model with explicit GPU configuration
        """
        self.manifest = context.manifest
        properties = context.system_properties
        
        # Log detailed information about environment
        logger.info(f"Python version: {properties.get('python_version')}")
        logger.info(f"Torch version: {torch.__version__}")
        
        # Check for CUDA
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            logger.warning("CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
            
        logger.info(f"Using device: {self.device}")
        
        try:
            # Define paths to check for pre-downloaded model
            model_paths = [
                # Container path
                f"/home/model-server/whisper-models/whisper_{self.model_size}.pt",
                # Volume mount path
                os.path.join(os.getcwd(), f"whisper-models/whisper_{self.model_size}.pt"),
                # Model store path
                os.path.join(properties.get("model_dir", ""), f"whisper_{self.model_size}.pt"),
            ]
            
            # First try to load from saved model file
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    logger.info(f"Found saved model file at {model_path} ({os.path.getsize(model_path)/1024/1024:.2f} MB)")
                    try:
                        # Create empty model with the right architecture
                        self.model = whisper.load_model(self.model_size, download_root="/tmp", in_memory=True, device="cpu")
                        # Load saved weights
                        state_dict = torch.load(model_path, map_location=self.device)
                        self.model.load_state_dict(state_dict)
                        self.model = self.model.to(self.device)
                        logger.info(f"Successfully loaded model from {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_path}: {e}")
            
            # If no saved model was found or loaded successfully, download it
            if not model_loaded:
                logger.warning("No valid saved model found. Downloading model (this may take a while)...")
                self.model = whisper.load_model(self.model_size, device=self.device)
                
                # Save the model for future use
                save_path = "/home/model-server/whisper-models/whisper_base.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                logger.info(f"Saving model to {save_path}")
                torch.save(self.model.state_dict(), save_path)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Verify model device
            for name, param in list(self.model.named_parameters())[:1]:
                logger.info(f"Model parameter {name} is on device: {param.device}")
            
            self.initialized = True
            logger.info("Model initialization complete")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e


    def preprocess(self, data):
        """
        Preprocess audio data
        """
        audio_data = data[0].get("audio")
        
        # For long audio files, consider chunking
        MAX_DURATION = 300  # 5 minutes per chunk
        
        try:
            # Convert to numpy array
            if isinstance(audio_data, (bytes, bytearray)):
                # Handle bytes data (from API requests)
                audio_bytes = BytesIO(audio_data)
                y, sr = sf.read(audio_bytes)
                if y.ndim > 1:
                    y = y.mean(axis=1)  # Convert stereo to mono
                
                # Check duration
                duration = len(y) / sr
                logger.info(f"Audio duration: {duration} seconds")
                
                # For very long audio, log a warning
                if duration > 600:  # > 10 minutes
                    logger.warning(f"Very long audio ({duration:.1f}s), processing may take significant time")
                
                # Convert to expected format
                audio_np = y
            else:
                # Handle numpy arrays or other formats
                audio_np = audio_data
                
            return audio_np
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise e
    def inference(self, data):
        """Run inference with the model"""
        logger.info(f"Running inference on {self.device}")
        
        try:
            # Prepare audio for Whisper model
            logger.info("Processing audio for whisper model")
            audio = whisper.pad_or_trim(data)
            logger.info("Computing log mel spectrogram")
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            logger.info("Running whisper model")
            with torch.no_grad():
                options = whisper.DecodingOptions(fp16=torch.cuda.is_available())
                result = whisper.decode(self.model, mel, options)
            
            logger.info("Inference complete")
            return result.text
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e

    def postprocess(self, inference_output):
        """Return the transcription results"""
        logger.info(f"Transcription result: {inference_output[:30]}...")
        return [{"transcription": inference_output}]
# Whisper TorchServe

This project provides a Dockerized deployment of the Whisper model using TorchServe. It includes all necessary components to build, serve, and interact with the Whisper model for audio transcription.

## Project Structure

- **Dockerfile**: Contains instructions to build the Docker image for the Whisper model.
- **model-store/**: Directory to hold model artifacts that will be served by TorchServe.
- **config/**: Contains configuration files for TorchServe.
  - **config.properties**: Specifies configuration settings for model management and logging options.
- **handler/**: Contains the custom handler for the Whisper model.
  - **whisper_handler.py**: Defines the `WhisperHandler` class for preprocessing, inference, and postprocessing of audio data.
- **requirements.txt**: Lists Python dependencies required for the project.
- **model-archiver/**: Contains scripts for creating model archives.
  - **create_model_archive.sh**: Shell script to package the model and handler for TorchServe.
- **serve.sh**: Shell script to start the TorchServe server with the specified model and configuration.
- **README.md**: Documentation for the project.

## Getting Started

### Prerequisites

- Docker installed on your machine.
- Basic knowledge of Docker and TorchServe.

### Building the Docker Image

To build the Docker image for the Whisper model, navigate to the project directory and run:

```bash
docker build -t whisper-torchserve .
```

### Running the Server

After building the image, you can run the TorchServe server using the provided `serve.sh` script:

```bash
./serve.sh
```

### Using the Whisper Model

Once the server is running, you can send audio files to the model for transcription. Refer to the API documentation for details on how to interact with the model.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
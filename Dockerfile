FROM pytorch/torchserve:latest

# Install dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install openai-whisper torchaudio soundfile numpy nvgpu

# Create directories with proper permissions
RUN mkdir -p /home/model-server/logs && \
    mkdir -p /home/model-server/model-store && \
    mkdir -p /home/model-server/whisper-models && \
    chmod 777 /home/model-server/logs && \
    chown -R model-server:model-server /home/model-server/logs && \
    chown -R model-server:model-server /home/model-server/whisper-models

# Copy handler
COPY handler/whisper_handler.py /home/model-server/model-archiver/handler/

# Copy model-archiver files with proper permissions
COPY --chmod=0755 model-archiver/create_model_archive.sh /home/model-server/model-archiver/
COPY model-archiver/requirements.txt /home/model-server/model-archiver/

# Copy configuration
COPY config/config.properties /home/model-server/config/
COPY config/model-config.yaml /home/model-server/config/

# Download model during build
COPY --chmod=0755 download_model.py /home/model-server/
RUN python /home/model-server/download_model.py

# Copy the start script with proper permissions
COPY --chmod=0755 serve.sh /home/model-server/serve.sh

# Run as model-server user
USER model-server
CMD ["/home/model-server/serve.sh"]
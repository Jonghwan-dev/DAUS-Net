# 1) Base image (must remain unchanged by competition rules)
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# 2) Set working directory inside the container
WORKDIR /app

# 3) Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy project source files
COPY model.py .
COPY networks/ /app/networks/
COPY datasets/ /app/datasets/
COPY data/ /app/data/
COPY train_checkpoint/ /app/train_checkpoint/
COPY trainer.py .
COPY train.py .
COPY test.py .
COPY pred.py .
COPY util/utils.py .
COPY README.md .

# 5) Copy dataset samples and model weights into the image (for self-contained runs)
#    Note: This makes the image large; alternatively, you can mount these at runtime.
COPY ./data/Val /app/input/
COPY ./data/private_val_for_participants.json /app/input.json
COPY ./train_checkpoint/best_model.pth /app/weights/best_model.pth

# 6) Set environment variables (so code does not require path edits)
#    These define the default in-container paths used by model.py
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output
ENV JSON_PATH=/app/input.json
ENV CKPT=/app/weights/best_model.pth

# 7) Entrypoint and default command
ENTRYPOINT ["python"]
CMD ["model.py"]
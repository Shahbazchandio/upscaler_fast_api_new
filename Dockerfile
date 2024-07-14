FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the RealESRGAN model
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Copy the application code
COPY main.py .

# Copy the templates directory
COPY templates templates/

# Copy the static directory if you have one (for CSS, JS, etc.)
# COPY static static/

# Copy any other necessary files
# COPY other_necessary_files .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
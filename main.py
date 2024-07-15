import io
import base64
import os
import asyncio
import logging
from typing import List
from functools import lru_cache
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from concurrent.futures import ThreadPoolExecutor
import cv2
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Imports for the chatbot feature
from transformers import AutoModelForCausalLM, AutoTokenizer

# Imports for BLIP image captioning
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a ThreadPoolExecutor for background tasks
executor = ThreadPoolExecutor(max_workers=2)  # Adjust based on your CPU

# Load the language model and tokenizer
@lru_cache(maxsize=None)
def get_language_model():
    model_name = "openai-community/gpt2-large"
    logger.info(f"Loading model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Add padding token to tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

@lru_cache(maxsize=None)
def get_models():
    model_4x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_2x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    model_path_4x = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    model_path_2x = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

    upsampler_4x = RealESRGANer(scale=4, model_path=model_path_4x, model=model_4x, tile=256, tile_pad=10, pre_pad=0, half=False)
    upsampler_2x = RealESRGANer(scale=2, model_path=model_path_2x, model=model_2x, tile=256, tile_pad=10, pre_pad=0, half=False)

    gfpgan_model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    gfpgan = GFPGANer(model_path=gfpgan_model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=upsampler_2x)

    return upsampler_2x, upsampler_4x, gfpgan

# BLIP Image Captioning model
@lru_cache(maxsize=None)
def get_image_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model


MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

async def validate_image(file: UploadFile):
    content = await file.read()
    size = len(content)
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail=f"Image size should not exceed {MAX_IMAGE_SIZE/1024/1024} MB")
    
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        if img.format not in ['JPEG', 'PNG']:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    return content

def process_image(content: bytes):
    image = Image.open(io.BytesIO(content))
    return np.array(image)

def enhance_image(img, scale_factor, mode, color_correct=False, sharpen=False):
    upsampler_2x, upsampler_4x, gfpgan = get_models()
    
    if color_correct:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge((l,a,b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    if mode == 'upscale':
        if scale_factor == 2:
            output, _ = upsampler_2x.enhance(img, outscale=2,  tile=128)
        elif scale_factor == 3:
            output, _ = upsampler_4x.enhance(img, outscale=3,  tile=128)
        else:  # scale_factor == 4
            output, _ = upsampler_4x.enhance(img, outscale=4,  tile=128)
    elif mode == 'face_enhance':
        # Apply GFPGAN for face enhancement
        _, _, output = gfpgan.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        
        # Apply additional upscaling if needed
        if scale_factor > 2:
            if scale_factor == 3:
                output, _ = upsampler_4x.enhance(output, outscale=1.5)
            else:  # scale_factor == 4
                output, _ = upsampler_4x.enhance(output, outscale=2)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        output = cv2.filter2D(output, -1, kernel)
    
    return output

def generate_caption(image):
    processor, model = get_image_captioning_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

async def process_queue_item():
    while True:
        item = await app.state.queue.get()
        try:
            img, scale_factor, mode = item['img'], item['scale_factor'], item['mode']
            output = await asyncio.get_event_loop().run_in_executor(executor, enhance_image, img, scale_factor, mode)
            item['future'].set_result(output)
        except Exception as e:
            item['future'].set_exception(e)
        finally:
            app.state.queue.task_done()

@app.on_event("startup")
async def startup_event():
    app.state.queue = asyncio.Queue(maxsize=5)
    app.state.worker = asyncio.create_task(process_queue_item())

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.queue.join()
    app.state.worker.cancel()
    await app.state.worker

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/enhance/")
async def enhance_image_api(file: UploadFile = File(...), scale_factor: int = Form(...),  mode: str = Form(...), color_correct: bool = Form(False), sharpen: bool = Form(False)):
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}, scale_factor: {scale_factor}, mode: {mode}")
    
    try:
        content = await validate_image(file)
        img = process_image(content)
        
        future = asyncio.Future()
        await app.state.queue.put({'img': img, 'scale_factor': scale_factor, 'mode': mode, 'color_correct': color_correct,'sharpen': sharpen,'future': future})
        output = await future

        output_img = Image.fromarray(output)

        buffered_original = io.BytesIO()
        Image.open(io.BytesIO(content)).save(buffered_original, format="JPEG")
        original_image_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        buffered_enhanced = io.BytesIO()
        output_img.save(buffered_enhanced, format="JPEG", quality=100)
        enhanced_image_base64 = base64.b64encode(buffered_enhanced.getvalue()).decode()
        
        return JSONResponse({
            "original": original_image_base64,
            "enhanced": enhanced_image_base64
        })
    
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def generate_response(prompt, max_length=100):
    tokenizer, model = get_language_model()
    device = next(model.parameters()).device
    
    # Format the prompt
    full_prompt = f"Human: {prompt}\nAI:"
    logger.info(f"Full prompt: {full_prompt}")
    
    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    logger.info(f"Encoded input shape: {inputs.shape}")
    
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=inputs.shape[1] + max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    logger.info(f"Raw model response: {response}")
    
    # Basic post-processing
    response = response.strip()
    if len(response) < 2 or response.lower().startswith("human:"):
        response = "I'm sorry, I don't have a good response for that. Could you please try asking something else?"
    
    logger.info(f"Final response: {response}")
    return response

@app.post("/chat/")
async def chat(message: str = Form(...)):
    try:
        logger.info(f"Received message: {message}")
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, generate_response, message
        )
        
        logger.info(f"Sending response: {response}")
        return JSONResponse({"response": response})
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating chat response: {str(e)}")
# Route for image captioning
@app.post("/caption/")
async def caption_image(file: UploadFile = File(...)):
    try:
        content = await validate_image(file)
        img = Image.open(io.BytesIO(content)).convert('RGB')
        
        caption = await asyncio.get_event_loop().run_in_executor(executor, generate_caption, img)
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "image": image_base64,
            "caption": caption
        })
    
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
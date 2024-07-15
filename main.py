import io
import base64
import os
import asyncio
import logging
import gc
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

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@lru_cache(maxsize=None)
def get_models():
    model_4x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_2x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    model_path_4x = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    model_path_2x = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

    upsampler_4x = RealESRGANer(scale=4, model_path=model_path_4x, model=model_4x, tile=256, tile_pad=10, pre_pad=0, half=False)
    upsampler_2x = RealESRGANer(scale=2, model_path=model_path_2x, model=model_2x, tile=256, tile_pad=10, pre_pad=0, half=False)

    return upsampler_2x, upsampler_4x

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

async def validate_image(file: UploadFile):
    content = await file.read()
    size = len(content)
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail=f"Image size should not exceed {MAX_IMAGE_SIZE/1024/1024} MB")
    return content

def process_image(content: bytes):
    image = Image.open(io.BytesIO(content))
    max_dimension = 1024
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension))
    return np.array(image)

async def async_enhance(img, scale_factor):
    upsampler_2x, upsampler_4x = get_models()
    loop = asyncio.get_event_loop()
    if scale_factor == 2:
        return await loop.run_in_executor(None, upsampler_2x.enhance, img, 2)
    elif scale_factor == 3:
        return await loop.run_in_executor(None, upsampler_4x.enhance, img, 3)
    else:  # scale_factor == 4
        return await loop.run_in_executor(None, upsampler_4x.enhance, img, 4)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/enhance/")
async def enhance_image(background_tasks: BackgroundTasks, file: UploadFile = File(...), scale_factor: int = Form(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")
    
    if scale_factor not in [2, 3, 4]:
        raise HTTPException(status_code=400, detail="Scale factor must be 2, 3, or 4.")
    
    try:
        content = await validate_image(file)
        img = process_image(content)
        
        output, _ = await async_enhance(img, scale_factor)

        output_img = Image.fromarray(output.astype(np.uint8))

        buffered_original = io.BytesIO()
        Image.open(io.BytesIO(content)).save(buffered_original, format="JPEG")
        original_image_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        buffered_enhanced = io.BytesIO()
        output_img.save(buffered_enhanced, format="JPEG", quality=95)
        enhanced_image_base64 = base64.b64encode(buffered_enhanced.getvalue()).decode()
        
        # Schedule cleanup
        background_tasks.add_task(cleanup, [img, output, output_img, buffered_original, buffered_enhanced])
        
        return JSONResponse({
            "original": original_image_base64,
            "enhanced": enhanced_image_base64
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing image")

def cleanup(objects_to_delete):
    for obj in objects_to_delete:
        del obj
    gc.collect()

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
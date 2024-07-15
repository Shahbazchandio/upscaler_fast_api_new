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

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a ThreadPoolExecutor for background tasks
executor = ThreadPoolExecutor(max_workers=2)  # Adjust based on your CPU

@lru_cache(maxsize=None)
def get_models():
    model_4x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_2x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    model_path_4x = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    model_path_2x = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

    upsampler_4x = RealESRGANer(scale=4, model_path=model_path_4x, model=model_4x, tile=256, tile_pad=10, pre_pad=0, half=False)
    upsampler_2x = RealESRGANer(scale=2, model_path=model_path_2x, model=model_2x, tile=256, tile_pad=10, pre_pad=0, half=False)

    gfpgan_model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    gfpgan = GFPGANer(model_path=gfpgan_model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

    return upsampler_2x, upsampler_4x, gfpgan

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

def enhance_image(img, scale_factor, mode):
    upsampler_2x, upsampler_4x, gfpgan = get_models()
    
    if mode == 'upscale':
        if scale_factor == 2:
            output, _ = upsampler_2x.enhance(img, outscale=2)
        elif scale_factor == 3:
            output, _ = upsampler_4x.enhance(img, outscale=3)
        else:  # scale_factor == 4
            output, _ = upsampler_4x.enhance(img, outscale=4)
    elif mode == 'face_enhance':
        _, _, output = gfpgan.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return output

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
    app.state.queue = asyncio.Queue(maxsize=10)
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
async def enhance_image_api(file: UploadFile = File(...), scale_factor: int = Form(...), mode: str = Form(...)):
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}, scale_factor: {scale_factor}, mode: {mode}")
    
    try:
        content = await validate_image(file)
        img = process_image(content)
        
        future = asyncio.Future()
        await app.state.queue.put({'img': img, 'scale_factor': scale_factor, 'mode': mode, 'future': future})
        output = await future

        output_img = Image.fromarray(output)

        buffered_original = io.BytesIO()
        Image.open(io.BytesIO(content)).save(buffered_original, format="JPEG")
        original_image_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        buffered_enhanced = io.BytesIO()
        output_img.save(buffered_enhanced, format="JPEG", quality=95)
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
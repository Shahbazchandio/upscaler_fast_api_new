import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = FastAPI()

templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
# Initialize Real-ESRGAN models
model_4x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_2x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

model_path_4x = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
model_path_2x = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

upsampler_4x = RealESRGANer(scale=4, model_path=model_path_4x, model=model_4x, tile=0, tile_pad=10, pre_pad=0, half=False)
upsampler_2x = RealESRGANer(scale=2, model_path=model_path_2x, model=model_2x, tile=0, tile_pad=10, pre_pad=0, half=False)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...), scale_factor: int = Form(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")
    
    if scale_factor not in [2, 3, 4]:
        raise HTTPException(status_code=400, detail="Scale factor must be 2, 3, or 4.")
    
    try:
        image_data = await file.read()
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        if scale_factor == 2:
            output, _ = upsampler_2x.enhance(img, outscale=2)
        elif scale_factor == 3:
            output, _ = upsampler_4x.enhance(img, outscale=3)
        else:  # scale_factor == 4
            output, _ = upsampler_4x.enhance(img, outscale=4)

        if img_mode == 'RGBA':
            output_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA))
        else:
            output_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

        buffered = io.BytesIO()
        Image.open(io.BytesIO(image_data)).save(buffered, format="PNG")
        original_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG")
        enhanced_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "original": original_image_base64,
            "enhanced": enhanced_image_base64
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
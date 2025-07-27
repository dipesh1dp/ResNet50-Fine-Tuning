from fastapi import FastAPI, UploadFile, File 
from PIL import Image 
import io 
from app.preprocessing import preprocess_image 
from app.inference import predict 

app = FastAPI(title='Flowe classifier API') 

@app.post("/predict/") 
async def predict_flower(file: UploadFile = File(...)): 
    contents = await file.read() 
    image = Image.open(io.BytesIO(contents)).convert("RGB") 
    processed = preprocess_image(image) 
    result = predict(processed) 
    return result 
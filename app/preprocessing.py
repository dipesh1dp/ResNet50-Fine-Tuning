from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))                                # Resize
    image = np.array(image).astype(np.float32) / 255 
    image = (image - [0.485,0.456, 0.406]) / [0.229, 0.224, 0.225]  # Normaliz0e
    image = np.transpose(image, (2, 0, 1))     # Channels first 
    image = np.expand_dims(image, axis=0)      # Add batch dim 
    return image.astype(np.float32) 
    
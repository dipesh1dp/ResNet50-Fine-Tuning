import onnxruntime as ort 
import numpy as np 
import json 
import os

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/resnet50_flower.onnx"))
# load image mapping 
with open("app/utils/class_mapping.json") as f:
    idx_to_label = json.load(f)

# Prefer GPU execution 
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] 
ort_session = ort.InferenceSession(model_path)

def predict(image_array: np.ndarray): 
    inputs = {ort_session.get_inputs()[0].name: image_array.astype(np.float32)} 
    outputs = ort_session.run(None, inputs)[0] 
    probs = softmax(outputs[0]) 
    top_idx = np.argmax(probs) 
    return {
        "class_id": int(top_idx), 
        "label": idx_to_label.get(str(top_idx), "Unknown"), 
        "confidence": float(probs[top_idx])
    }

def softmax(x): 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0)


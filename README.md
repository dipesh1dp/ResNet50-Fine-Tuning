<p align="center">
  <img src="frontend\assets\header.png" alt="App Header" width="200"/>
</p>

## 102 Category Flower Classification using ResNet50

This project implements a transfer learning pipeline for classifying flowers into 102 categories using a pretrained ResNet-50 model. The model is fine-tuned on the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), with hyperparameter tuning done using **Optuna** and support for **ONNX export**.

---

### 📌 Project Highlights

* ResNet50 transfer learning (final FC layer fine-tuned)
* Optuna-based hyperparameter tuning with:

  * Learning rate
  * Optimizer (`Adam`, `SGD`)
  * Batch size
  * Weight decay
* Early stopping + Optuna pruning
* Model export to PyTorch `.pth` and ONNX `.onnx` format

---

## Demo 

<p align="left">
  <img src="frontend\assets\input.png" alt="App Header" width="600"/>
</p>

<p align="left">
  <img src="frontend\assets\output.png" alt="App Header" width="600"/>
</p>


### 📁 Dataset

* [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
* Dataset contains:

  * 8,189 flower images
  * 102 classes
* Labels and splits are loaded from `.mat` files

---

### 🧪 Training Workflow

1. Load pretrained ResNet-50 from `torchvision.models`.
2. Replace the final FC layer to output 102 classes.
3. Freeze all layers except FC.
4. Train with `CrossEntropyLoss`, tuned optimizer.
5. Use Optuna to search best hyperparameters.
6. Track `val_f1` as the objective metric.
7. Apply early stopping and Optuna pruning.
8. Save the best model.

---

### ⚙️ Hyperparameters Tuned via Optuna

* `learning_rate`: Log-uniform between `1e-5` and `1e-2`
* `batch_size`: `[8, 16, 32]`
* `optimizer`: `['Adam', 'SGD']`
* `weight_decay`: `1e-5` to `1e-2`
* `momentum`: for SGD, between `0.85` and `0.99`

---

## 📊 Results & Performance

### Best Hyperparameters:

  ```json
  {
    "optimizer": "Adam",
    "lr": 0.005495396524417391,
    "batch_size": 8,
    "weight_decay": 0.0006037709271197784
    }
  ```
### Performance: 
| Metric        | Validation   | Test     |
| ------------- | -----------  | -------  |
| **Loss**      | `0.7556`     | `0.8808` |
| **Accuracy**  | `83.50%`     | `78.54%` |
| **Precision** | `67.51%`     | `59.96%` |
| **Recall**    | `61.43%`     | `53.95%` |
| **F1 Score**  | `63.63%`     | `56.13%` |



---
## How to run? 

1. Clone the Repository
```bash
git clone https://github.com/dipesh1dp/toxic-comment-app.git
cd bert-toxic-comment-classifier
```
2. Create and Activate a Virtual Environment (Optional but Recommended)
On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install all dependencies
```bash
pip install -r requirements.txt
```
4. Run the FastAPI server 
```bash
uvicorn app.main:app --reload 
```
Your backend will now be running at: http://127.0.0.1:8000. You can also see the docs at: http://127.0.0.1:8000/docs

5. Run the Streamlit frontend
```bash
streamlit run frontend/streamlit_ui.py
```

---

### 📎 File Structure

```
├── app/
|   ├── __init__.py
|   ├── inference.py
|   ├── main.py
|   ├── preprocessing.py
|   └── utils/
|       └── class_mapping.json
|
├── model/
|   ├── best_resnet50_flower.pth
|   └── resnet50_flower.onnx
|
├── frontend/
|   ├── streamlit_ui.py
|   └── assets/
|       ├── header.png
|       ├── input.png 
|       └── output.png
|
├── model-training.ipynb            # Training Notebook (ran on Kaggle)
└── README.md
      
```

---

### 🙌 Credits

* [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/)
* TorchVision for pretrained models
* Optuna for efficient hyperparameter tuning

---

### 📌 License

MIT License

---

Learning Project by [Dipesh Pandit](https://www.linkedin.com/in/dipesh1dp/).

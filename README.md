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

### 💾 Saving & Exporting the Model

* **Save best PyTorch model:**

  ```python
  torch.save(best_model_wts, "best_resnet50_flower.pth")
  ```

* **Load for inference:**

  ```python
  model.load_state_dict(torch.load("best_resnet50_flower.pth"))
  ```

* **Export to ONNX:**

  ```python
  torch.onnx.export(model, dummy_input, "resnet50_flower.onnx", ...)
  ```

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
## 🚀 Requirements

* Python 3.8+
* PyTorch
* torchvision
* torchmetrics
* Optuna
* scipy
* tqdm
* onnx
* 

Install via:

```bash
pip install torch torchvision torchmetrics optuna scipy tqdm onnx 
```
---

### 📎 File Structure

```
├── app/
|   ├── __init__.py
|   ├── inference.py
|   ├── main.py
|   ├── preprocessing.py
|   └── utils
|       └── class_mapping.json
|
├── model/
|   ├── best_resnet50_flower.pth
|   └── resnet50_flower.onnx
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

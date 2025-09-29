🖼️ CNN Image Classification with TensorFlow & PyTorch

This project demonstrates how to build a Convolutional Neural Network (CNN) from scratch for image classification using CIFAR-10 dataset.
Two implementations are provided: one using TensorFlow/Keras and one using PyTorch.

📌 Features

Custom CNN architecture with Conv2D, BatchNorm, Dropout, Pooling layers

Data augmentation (flip, rotation, zoom, crop)

Callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau) in TensorFlow

Explicit training loop in PyTorch with LR scheduler

Achieves ~70–80% accuracy on CIFAR-10 (higher possible with transfer learning)

Model saving & loading support (.keras / .h5 for TensorFlow, .pth for PyTorch)

⚙️ Requirements

Install dependencies:

pip install tensorflow torch torchvision

🚀 Usage
🔹 TensorFlow / Keras

Run training:

python tf_cifar10_cnn.py


Save model (preferred format):

model.save("tf_cifar10_final.keras")


Load model:

from tensorflow import keras
model = keras.models.load_model("tf_cifar10_final.keras")

🔹 PyTorch

Run training:

python torch_cifar10_cnn.py


Save best model:

torch.save(model.state_dict(), "torch_cifar10_best.pth")


Load model:

from model import SimpleCNN
import torch

model = SimpleCNN()
model.load_state_dict(torch.load("torch_cifar10_best.pth"))
model.eval()

📊 Results
Framework	Dataset	Accuracy
TensorFlow	CIFAR-10	~75%
PyTorch	CIFAR-10	~75%

(Accuracy depends on training epochs, augmentations, and hardware.)

📂 Project Structure
.
├── tf_cifar10_cnn.py        # TensorFlow implementation
├── torch_cifar10_cnn.py     # PyTorch implementation
├── checkpoints/             # Saved models (PyTorch)
├── data/                    # CIFAR-10 dataset (auto-downloaded)
└── README.md                # Project documentation

🔮 Future Improvements

Use transfer learning (ResNet, EfficientNet, ViT)

Add CutMix, MixUp, or Cutout for augmentation

Deploy trained model with Streamlit or FastAPI

# OUTPUT

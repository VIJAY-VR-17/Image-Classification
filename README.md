# CIFAR-10 Image Classification using CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset.

## 📁 Dataset

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is divided into 50,000 training images and 10,000 test images.

**Class Names:**

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

## 📌 Project Workflow

### 1. Install TensorFlow (if not already installed)

```bash
pip install tensorflow
```

### 2. Import Libraries

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

### 3. Load and Preprocess Data

* Load training and testing sets from CIFAR-10.
* Normalize the pixel values between 0 and 1.

### 4. Visualize Dataset

Plot a sample of 25 training images with their corresponding class labels.

### 5. Build CNN Model

Model architecture:

* Conv2D (32 filters, 3x3)
* MaxPooling2D
* Conv2D (64 filters, 3x3)
* MaxPooling2D
* Conv2D (64 filters, 3x3)
* Flatten
* Dense (64 units)
* Dense (10 units - output layer)

### 6. Compile the Model

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### 7. Train the Model

```python
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
```

### 8. Evaluate Model Performance

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
```

### 9. Plot Accuracy and Loss

Graph training and validation accuracy and loss across epochs.

## 📊 Results

Final Test Accuracy: **68.6%** after 20 epochs (10 + 10).

Validation accuracy initially improves but then degrades due to overfitting.

## 📈 Visualization Sample

The project includes visualizations for:

* Input images and labels
* Training/Validation accuracy over time
* Training/Validation loss over time

## 🛠️ Improvements

* Add dropout or batch normalization to reduce overfitting
* Data augmentation to increase variability
* Use pretrained models like ResNet for transfer learning

## 💡 Requirements

* Python 3.7+
* TensorFlow 2.18+
* matplotlib

## ✅ License

This project is provided for educational purposes.

---

Developed using TensorFlow & Keras

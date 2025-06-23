# 🐶🐱 Dog vs Cat Classification using Neural Networks

This project demonstrates a basic binary image classification task using neural networks — distinguishing between **dogs** and **cats**. The project is beginner-friendly and provides insight into essential deep learning techniques such as image preprocessing, model training, and evaluation using Python and TensorFlow/Keras.

---

## 🧠 Project Overview

- **Objective:** Train a neural network to classify input images as either a dog or a cat.
- **Dataset:** A collection of labeled dog and cat images. The project includes manual testing with real-world examples.
- **Frameworks Used:** TensorFlow, Keras, NumPy, Matplotlib, PIL

---

## 🖼️ Sample Images

Here are two example images used for classification testing:

### Example 1: Dog Image  
![Dog Image](https://raw.githubusercontent.com/nithun-rajan/Dog-vs-Cat-classification-using-Neural-Networks/main/4.jpg)

### Example 2: Cat Image  
![Cat Image](https://raw.githubusercontent.com/nithun-rajan/Dog-vs-Cat-classification-using-Neural-Networks/main/8.jpg)

---

## 🧪 How It Works

1. **Image Preprocessing:**
   - Resize images to uniform size (e.g., 64x64)
   - Normalize pixel values to improve convergence

2. **Model Architecture:**
   - Input Layer: Accepts image arrays
   - Hidden Layers: Dense layers with activation functions (ReLU)
   - Output Layer: Sigmoid function for binary classification (Dog or Cat)

3. **Training:**
   - Binary Crossentropy loss function
   - Optimizer: Adam
   - Evaluation metrics: Accuracy

---

## 📁 File Structure

Dog-vs-Cat-classification-using-Neural-Networks/
├── main.py # Main training and prediction script
├── 4.jpg # Dog image for test/demo
├── 8.jpg # Cat image for test/demo
├── README.md # Project overview and documentation


---

📊 Results & Evaluation

Accuracy: ~97% on test set
The model correctly classifies custom images, including the two examples above.
📌 Key Learnings

Binary image classification using neural networks
Image preprocessing techniques
Training and evaluating deep learning models using Python
Working with real-world image data for testing
📸 Visualization & Prediction Output

The script includes visualization using matplotlib to display prediction results alongside the input image. The output also shows the predicted class (Dog/Cat) for manual test images.

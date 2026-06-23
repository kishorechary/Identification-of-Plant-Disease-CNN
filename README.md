# PlantDiseaseDetection

It is a plant disease detection project...
cnn.py is used to train and test the model
ui.py is used for finding the disease in the given leaf by the user

PACKAGE REQUIREMENTS
  numpy (pip install numpy)
  tqdm (pip install tqdm)
  TensorFlow
  openCv
  matplotlib


# 🌿 Plant Disease Detection using CNN

## 📌 Project Overview

Plant Disease Detection using Convolutional Neural Networks (CNN) is a deep learning-based application that identifies whether a plant leaf is healthy or unhealthy from an input image. The system uses image processing and CNN classification techniques to assist in the early detection of plant diseases.

## 🎯 Objectives

* Detect plant leaf health status from images.
* Classify leaves as Healthy or Unhealthy.
* Provide a simple graphical user interface (GUI) for users.
* Demonstrate the application of Deep Learning in Agriculture.

## 🛠️ Technologies Used

* Python 3.7
* TensorFlow 1.15
* TFLearn
* OpenCV
* NumPy
* Pillow
* Matplotlib
* Tkinter

## 📂 Project Structure

Identification-of-Plant-Disease-CNN

├── ui.py

├── healthyvsunhealthy-0.001-2conv-basic.model

├── healthyvsunhealthy-0.001-2conv-basic.model.meta

├── healthyvsunhealthy-0.001-2conv-basic.model.index

├── requirements.txt

└── README.md

## ⚙️ Installation

### Create Environment

```bash
conda create -n plantcnn python=3.7
conda activate plantcnn
```

### Install Dependencies

```bash
pip install tensorflow==1.15
pip install tflearn==0.5.0
pip install numpy==1.21.6
pip install opencv-python==4.5.5.64
pip install matplotlib==3.5.3
pip install pillow==9.5.0
pip install tqdm==4.67.1
```

## ▶️ Execution Process

Navigate to the project directory:

```bash
cd Identification-of-Plant-Disease-CNN
```

Run the application:

```bash
python ui.py
```

### Steps to Use

1. Launch the application.
2. Click **Select Image**.
3. Choose a plant leaf image.
4. Click **Analyze**.
5. View the prediction result.

## 📊 Output

The system predicts one of the following:

* Healthy Leaf 🌿
* Unhealthy Leaf 🍂

## 🚀 Features

* CNN-based image classification.
* User-friendly GUI.
* Fast prediction results.
* Deep learning implementation using TensorFlow and TFLearn.

## 👨‍💻 Author

**Kammari Kishore**
,**Ansh N Parmar**.
**Sharan Teja**

* LinkedIn: https://www.linkedin.com/in/kishore-kammari-25b409320
* GitHub: https://github.com/kishorechary

## 📄 License

This project is developed for academic and learning purposes.

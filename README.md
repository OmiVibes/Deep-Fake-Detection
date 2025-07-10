# 🧠 DeepFake Detection

A deep learning project to detect real vs. fake images using MobileNetV2 and a Flask-based web interface.

---

## 🚀 Project Overview

With the rise of deepfake technology, identifying manipulated media has become critical.  
This project uses **MobileNetV2** (via transfer learning) to classify images as **real** or **fake**, and deploys a web interface using **Flask** for real-time detection.

---

## ⚙️ Tech Stack

- **Frontend**: Flask (Image upload & results display)  
- **Backend**: Python, TensorFlow/Keras (Model training and inference)  
- **Model**: MobileNetV2 (pre-trained, fine-tuned)  
- **Preprocessing**: Image resizing, normalization, augmentation  
- **Dataset**: Real and Fake Face Dataset (from Kaggle)  

---

## 📁 Dataset Info

- Two classes: `Real` and `Fake`
- Preprocessing includes:
  - Resize to 224×224
  - Normalize pixel values
  - Augment images with flipping, rotation, etc.

⚠️ **Note**: Dataset is not included fully in GitHub due to size  


Place the folders in your project root:

/training_real

/training_fake


---

## 📊 Model Performance

- **Accuracy**: 77.5%
- **Precision/Recall** improved significantly over baseline
- Real-time detection through web app
- Lightweight and deployable on low-resource systems

---

## 🛠 How to Run the Project

Follow these steps to run the project locally:

1. **Clone the repository**
   - Open your terminal and run:
     ```bash
     git clone https://github.com/OmiVibes/Deep-Fake-Detection.git
     cd Deep-Fake-Detection
     ```

2. **Install the dependencies**
   - Run the following command to install the required Python libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download and add the dataset**
   - Download the image dataset from [Google Drive](https://your-dataset-link.com).
   - Create the following folders inside the project directory and place the images accordingly:
     ```
     /training_real
     /training_fake
     ```

4. **Run the Flask application**
   - Start the Flask server using:
     ```bash
     python app.py
     ```

5. **Access the application**
   - Open your browser and visit:
     ```
     http://localhost:5000/
     ```
   - Upload an image to get an instant real or fake prediction.

---

## 👨‍💻 Author

**Om Shinde**  
📧 [omshinde1819@gmail.com](mailto:omshinde1819@gmail.com)  
🌐 [GitHub – OmiVibes](https://github.com/OmiVibes)

---

## Authors

- [@octokatherine](https://www.github.com/octokatherine)


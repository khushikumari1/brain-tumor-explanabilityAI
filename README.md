# 🧠 Brain Tumor Detection System

A comprehensive deep learning system for detecting and classifying brain tumors from MRI scans using Convolutional Neural Networks (CNN). The system can identify four different conditions: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

## 🌟 Features

- **Multi-class Classification**: Detects 4 different brain conditions
- **Interactive Web Interface**: User-friendly Streamlit application
- **Grad-CAM Visualization**: Explainable AI with heatmap overlays
- **PDF Report Generation**: Professional medical reports with QR codes
- **Real-time Predictions**: Fast inference with confidence scores
- **Model Training Pipeline**: Complete training and evaluation setup

## 📊 Dataset

The system is trained on a comprehensive brain MRI dataset with the following structure:

```
dataset/
├── Training/
│   ├── glioma/        (1,321 images)
│   ├── meningioma/    (1,339 images)
│   ├── notumor/       (1,595 images)
│   └── pituitary/     (1,457 images)
└── Testing/
    ├── glioma/        (300 images)
    ├── meningioma/    (306 images)
    ├── notumor/       (405 images)
    └── pituitary/     (300 images)
```

**Total Dataset**: ~7,000+ brain MRI images across 4 classes

## 🏗️ Model Architecture

The CNN model consists of:

- **Input Layer**: 150x150x3 RGB images
- **Convolutional Layers**:
  - Conv2D(32) → MaxPooling2D
  - Conv2D(64) → MaxPooling2D
  - Conv2D(128) → MaxPooling2D
- **Dense Layers**:
  - Dense(128) with ReLU activation
  - Dropout(0.5) for regularization
  - Dense(4) with Softmax activation (4 classes)

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone "https://github.com/khushikumari1/brain-tumor-explanabilityAI"
   cd brain-tumor-explanabilityAI
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Download the dataset in /dataset

Link of dataset -> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Train the Model

```bash
python train_brain_tumor_model.py
```

NOTE - .h5 file will be created here

### Predict the Model

```bash
python predict_brain_tumor.py
```

Specify the path of the MRI image for detection of brain tumor

### Running the Application

1. **Launch the Streamlit app**

   ```bash
   streamlit run brain_tumor_app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload an MRI image** and click "Analyze MRI" to get predictions

## 📁 Project Structure

```
brain-tumor/
├── brain_tumor_app.py          # Main Streamlit application
├── train_brain_tumor_model.py  # Model training script
├── predict_brain_tumor.py      # Standalone prediction script
├── brain_tumor_model.h5        # Trained model weights
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── dataset/                    # Training and testing data
│   ├── Training/
│   └── Testing/
└── venv/                       # Virtual environment
```

## 🔧 Usage

### Web Application

1. **Upload**: Select a brain MRI image (JPG, JPEG, PNG)
2. **Analyze**: Click "Analyze MRI" to process the image
3. **View Results**:
   - **Home**: See prediction with confidence scores
   - **Grad-CAM**: Visualize model attention with heatmaps
   - **Download Report**: Generate PDF medical reports

### Command Line Prediction

```bash
python predict_brain_tumor.py
# Enter the full path to your MRI image when prompted
```

### Training New Models

```bash
python train_brain_tumor_model.py
```

## 🎯 Model Performance

The trained model achieves high accuracy on the test dataset with the following capabilities:

- **Classification**: 4-class brain tumor detection
- **Confidence Scoring**: Probability distribution across all classes
- **Visualization**: Grad-CAM heatmaps for model interpretability
- **Report Generation**: Professional PDF reports for medical documentation

## 🔍 Grad-CAM Visualization

The application includes Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations of the model's decision-making process:

- **Heatmap Overlay**: Shows which regions the model focuses on
- **Adjustable Intensity**: Control overlay transparency
- **Medical Interpretability**: Helps understand model predictions

## 📄 PDF Report Features

Generated reports include:

- **Cover Page**: Professional header with timestamp
- **Image Analysis**: Original MRI and Grad-CAM overlay side-by-side
- **Prediction Summary**: Class prediction with confidence scores
- **Confidence Breakdown**: Detailed probability distribution
- **Medical Notes**: Disclaimers and recommendations
- **QR Code**: Unique report identifier

## 🛠️ Dependencies

- **TensorFlow**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Image processing
- **PIL/Pillow**: Image manipulation
- **Plotly**: Interactive visualizations
- **ReportLab**: PDF generation
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization

## ⚠️ Important Disclaimers

- **Medical Disclaimer**: This tool is for research and educational purposes only
- **Not for Clinical Use**: Always consult qualified medical professionals for actual diagnoses
- **Model Limitations**: AI models may have biases and limitations
- **Data Privacy**: Ensure patient data privacy when using this system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Brain MRI dataset contributors
- TensorFlow and Keras communities
- Streamlit for the excellent web framework
- Medical imaging research community

## 📞 Support

For questions, issues, or contributions, please:

1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Remember**: This is a research tool. Always consult medical professionals for clinical decisions! 🏥


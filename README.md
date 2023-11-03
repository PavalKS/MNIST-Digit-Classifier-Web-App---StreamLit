# MNIST-Digit-Classifier-Web-App---StreamLit
This repository contains a Streamlit web application that serves as an MNIST digit classifier using a trained neural network. Users can upload images for classification, view training statistics, and understand the working of the neural network.

This is a Streamlit web application that serves as an MNIST digit classifier using a trained neural network. The application allows users to either upload images for classification or draw digits on a canvas. It also provides insights into the working of the neural network and training statistics.

## Features

- Upload a handwritten digit image for classification.
- Draw digits on a canvas and classify them.
- Visualize training statistics, including epoch, training loss, validation accuracy, validation loss, and duration.
- Learn how the neural network works and is trained.

## Installation

### 1. Clone the repository:

   ```
   git clone https://github.com/PavalKS/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

### 2. Install the required Python packages:

  ```
  pip install -r requirements.txt
  ```

### 3. Run the Streamlit app:
  ```
  streamlit run app.py
  ```

Open your web browser and navigate to the provided URL to use the application.

## Usage:
- Upload Image as input
- The model will predict the digit and display the result.
- Explore training statistics and learn how the neural network works.

## Technologies Used
- Python
- Streamlit
- PyTorch
- scikit-learn
- Matplotlib

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


## Code Split for Efficiency
If you want to improve efficiency or have separate stages for model training and model usage in your Streamlit app, consider splitting the code into two parts:
1. **Model Training**: Create a dedicated Python script for training your neural network model. Save the trained model to a file (e.g., `mnist_classifier.pth`).
2. **Streamlit App**: Develop another Python script that loads the pretrained model and integrates it into the Streamlit app for digit classification.
By splitting the code, you can save time when running the Streamlit app as you won't need to train the model every time you use it. This separation also facilitates a more efficient workflow for model development and web application deployment.


## Here's what it looks like:
<img width="456" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/513964bf-ecd2-447a-9214-5cb2b3427cb4">
<img width="423" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/848bc5a4-3d0a-4267-994b-e732a40c28df">
<img width="432" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/7ef10fd3-662e-4460-80e5-3dc76b6d30ad">
<img width="417" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/68581524-cf76-46c7-b528-d60ac4ffe351">
<img width="276" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/19ce2bf2-aa88-489a-a12d-1160f24b08b1">
<img width="411" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/9e424121-7676-4305-b4ce-1add5b894fcc">
<img width="413" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/1196fd2f-caae-47e4-a0f4-9aacdbc1a5da">
<img width="412" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/26a41b22-d935-4050-8aff-1ec79b76531b">
<img width="412" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/425d49ad-0a34-4af1-a835-54950cb548ec">
<img width="292" alt="image" src="https://github.com/PavalKS/MNIST-Digit-Classifier-Web-App---StreamLit/assets/74084308/3e401b15-049a-4cc2-a14a-2016c773c28c">

import streamlit as st
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from skimage import color
import pandas as pd
import io

# Load MNIST dataset
st.title("MNIST Digit Classifier")
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)

# Preprocess data
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define functions to display images
def plot_example(X, y):
    """Plot a grid of images and their labels."""
    plt.figure(figsize=(15, 15))
    for i in range(10):
        for j in range(10):
            index = i * 10 + j
            plt.subplot(10, 10, index + 1)
            plt.imshow(X[index].reshape(28, 28))
            plt.xticks([])
            plt.yticks([])
            plt.title(y[index], fontsize=8)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    st.pyplot(plt)

# Display a selection of training images and their labels
st.write("Sample of Training Images and Labels")
plot_example(X_train, y_train)

# Build and train a neural network
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim / 8)
output_dim = len(np.unique(mnist.target))

class ClassifierModule(nn.Module):
    def __init__(self, input_dim=mnist_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X


net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=20,
    lr=0.1,
    device=device,
)

with st.spinner("Training the model..."):
    net.fit(X_train, y_train)

# Display training statistics in a table
st.subheader("Training Statistics")
# Explain the terms in the training statistics
st.subheader("Explanation of Training Statistics")
st.markdown("1. **Epoch:** The number of times the model has been trained on the entire training dataset.")
st.markdown("2. **Train Loss:** The training loss at the end of each epoch, indicating how well the model is fitting the training data.")
st.markdown("3. **Valid Acc:** Validation accuracy, the accuracy of the model on the validation dataset at the end of each epoch.")
st.markdown("4. **Valid Loss:** Validation loss, the loss on the validation dataset, measuring how well the model generalizes.")
st.markdown("5. **Duration:** The time taken to complete each epoch.")
st.markdown("`  **Epoch**  |  **Train loss**  |  **Valid Acc**  |  **Valid Loss**  |  **Duration**  `")
# Extract the relevant training statistics
train_stats = net.history[:, ['epoch', 'train_loss', 'valid_acc', 'valid_loss', 'dur']]
st.table(train_stats)

# Make predictions and calculate accuracy
y_pred = net.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize misclassified images
error_mask = y_pred != y_test
st.write("Misclassified Images")
plot_example(X_test[error_mask], y_pred[error_mask])

# Preprocess and predict
st.subheader("Upload a Handwritten Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Make predictions on the preprocessed image here and display the result
    prediction = net.predict(image)
    st.subheader("Predicted Output")
    st.write(f"The model predicts that the digit is: {prediction[0]}")

# Explanation of the working of the neural network
st.subheader("How the Neural Network Works")
st.image('neuralnetwork.png', caption='Neural Network Architecture', use_column_width=True)
st.write("The neural network is trained on the MNIST dataset, which consists of 28x28 pixel images of handwritten digits (0-9).")
st.write("The neural network uses one hidden layer with 98 neurons and an output layer with 10 neurons, each representing a digit.")
st.write("The images are preprocessed to ensure consistency with the training data.")
st.write("The model is trained using the training data, and accuracy is measured on the test data.")

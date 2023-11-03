import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import io
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Load MNIST dataset
st.title("MNIST Digit Classifier")
st.write("Authored by Paval KS - pavalsudhakar@gmail.com")
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)

# Explanation of the working of the neural network
st.subheader("How the Neural Network Works")
st.image('neuralnetwork.png', caption='Neural Network Architecture', use_column_width=True)
st.write("Neural networks are a type of computer program that's inspired by the way our brains work. They're used for a wide range of tasks, including things like recognizing handwritten digits in the MNIST dataset, which is a common example in the field of machine learning.")
st.write("The neural network is trained on the MNIST dataset, which consists of 28x28 pixel images of handwritten digits (0-9).")
st.write("The neural network uses one hidden layer with 98 neurons and an output layer with 10 neurons, each representing a digit.")
st.write("The images are preprocessed to ensure consistency with the training data.")
st.write("The model is trained using the training data, and accuracy is measured on the test data.")

st.subheader("Here's a simplified explanation of how a neural network works:")
st.image('neuralnetwork1.png', caption='A Simple Neural Network Architecture', use_column_width=True)
st.write("1. **Input Layer:** The process begins with an input layer that receives the data. In the case of the MNIST dataset, each handwritten digit is represented as an image of 28x28 pixels, which is flattened into a 1D array of 784 numbers. This array is the input to the neural network.")
st.write("2. **Hidden Layers:** IThere is a hidden layer with 98 neurons. These neurons perform calculations on the input data. Each neuron takes in some of the input numbers, processes them using a set of weights and biases, and produces an output. This is where the magic happens – the network learns to adjust the weights and biases during training to make accurate predictions.")
st.write("3. **Output Layer:** After processing through the hidden layer, the data is passed to the output layer, which consists of 10 neurons in this case (since there are 10 possible digits, 0 through 9). Each output neuron corresponds to a specific digit, and the network's goal is to activate the correct output neuron for a given input image.")
st.write("4. **Training:** To make the neural network good at recognizing digits, it needs to be trained using a dataset of labeled examples. During training, the network compares its predictions to the correct answers and adjusts its internal parameters (weights and biases) to minimize the error between the predicted and actual values. This is done using optimization algorithms.")
st.write("5. **Prediction:** Once the neural network has been trained, it can take new, unlabeled images of handwritten digits and make predictions about which digit each image represents. It does this by forward-propagating the input data through the network, and the output neuron with the highest activation value indicates the predicted digit.")
st.write("So, in a nutshell, a neural network is a program that can learn to recognize patterns in data, such as handwritten digits in the MNIST dataset, by adjusting its internal parameters through training to make accurate predictions.")


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
        super(ClassifierModule, self).__init__()  # Corrected the superclass initialization
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

def load_model():
    net = NeuralNetClassifier(
        module=ClassifierModule,  # Corrected 'module' parameter
        max_epochs=20,
        lr=0.1,
        device=device,
    )
    net.fit(X_train, y_train)
    return net

net = load_model()
# Display training statistics in a table
st.subheader("Training Statistics")
# Explain the terms in the training statistics
st.subheader("Explanation of Training Statistics")
st.markdown("1. *Epoch:* The number of times the model has been trained on the entire training dataset.")
st.markdown("2. *Train Loss:* The training loss at the end of each epoch, indicating how well the model is fitting the training data.")
st.markdown("3. *Valid Acc:* Validation accuracy, the accuracy of the model on the validation dataset at the end of each epoch.")
st.markdown("4. *Valid Loss:* Validation loss, the loss on the validation dataset, measuring how well the model generalizes.")
st.markdown("5. *Duration:* The time taken to complete each epoch.")
# Extract the relevant training statistics
train_stats = net.history[:, ['epoch', 'train_loss', 'valid_acc', 'valid_loss', 'dur']]
st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Epoch&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Train loss&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Valid Acc&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Valid Loss&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Duration*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;")
st.table(train_stats)

# Plot the training statistics
st.subheader("Training Statistics Plot")
fig, ax = plt.subplots(figsize=(10, 6))
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ax.plot(x, net.history[:, ['train_loss']], label='Train Loss')
ax.plot(x, net.history[:, ['valid_acc']], label='Valid Acc')
ax.plot(x, net.history[:, ['valid_loss']], label='Valid Loss')
ax.plot(x, net.history[:, ['dur']], label='Duration')
ax.set_xlabel('Epoch')
ax.set_title('Training Statistics')
ax.legend()
st.pyplot(fig)

# Make predictions and calculate accuracy
y_pred = net.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy * 100:.2f}%")
st.write("Accuracy is the number of correctly predicted data points out of all the data points.")

# Display the confusion matrix
st.subheader("Confusion Matrix")
st.markdown("The confusion matrix is a visualization of the model's performance on the test data. It shows how many data points were correctly predicted and how many were misclassified.")
st.markdown("Each row of the matrix represents the actual class (true label), and each column represents the predicted class by the model.")
st.markdown("Here's what the confusion matrix elements mean:")
st.markdown("- True Positives (TP): The number of data points that were correctly predicted as positive.")
st.markdown("- True Negatives (TN): The number of data points that were correctly predicted as negative.")
st.markdown("- False Positives (FP): The number of data points that were incorrectly predicted as positive (false alarms).")
st.markdown("- False Negatives (FN): The number of data points that were incorrectly predicted as negative (misses).")
st.markdown("The diagonal elements (top-left to bottom-right) represent correct predictions (TP and TN), while off-diagonal elements represent misclassifications (FP and FN).")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)

# Visualize misclassified images
error_mask = y_pred != y_test
st.write("Misclassified Images")
plot_example(X_test[error_mask], y_pred[error_mask])

# Preprocess and predict
st.subheader("Upload an Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST size
    image = transforms.ToTensor()(image)
    image = 1 - image  # Invert black and white
    image = image.view(1, -1)  # Reshape the image to (1, 784)
    
    # Ensure the input data matches the model's architecture
    prediction = net.predict(image)
    st.subheader("Predicted Output")
    st.write(f"The model predicts that the digit is: {prediction[0]}")

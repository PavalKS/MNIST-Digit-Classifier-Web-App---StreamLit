import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import io
import seaborn as sns
from matplotlib.animation import FuncAnimation

st.title("MNIST Digit Classifier")
st.write("Authored by PAVAL KS - pavalsudhakar@gmail.com")
st.divider()

# Explanation of the working of the neural network
st.subheader("How the Neural Network Works")
st.image('neuralnetwork.png', caption='Neural Network Architecture', use_column_width=True)
st.write("Neural networks are a type of computer program that's inspired by the way our brains work. They're used for a wide range of tasks, including things like recognizing handwritten digits in the MNIST dataset, which is a common example in the field of machine learning.")
st.write("The neural network is trained on the MNIST dataset, which consists of 28x28 pixel images of handwritten digits (0-9).")
st.write("The neural network uses one hidden layer with 98 neurons and an output layer with 10 neurons, each representing a digit.")
st.write("The images are preprocessed to ensure consistency with the training data.")
st.write("The model is trained using the training data, and accuracy is measured on the test data.")
st.divider()

st.subheader("Here's a simplified explanation of how a neural network works:")
st.image('neuralnetwork1.png', caption='A Simple Neural Network Architecture', use_column_width=True)
st.write("1. **Input Layer:** The process begins with an input layer that receives the data. In the case of the MNIST dataset, each handwritten digit is represented as an image of 28x28 pixels, which is flattened into a 1D array of 784 numbers. This array is the input to the neural network.")
st.write("2. **Hidden Layers:** IThere is a hidden layer with 98 neurons. These neurons perform calculations on the input data. Each neuron takes in some of the input numbers, processes them using a set of weights and biases, and produces an output. This is where the magic happens – the network learns to adjust the weights and biases during training to make accurate predictions.")
st.write("3. **Output Layer:** After processing through the hidden layer, the data is passed to the output layer, which consists of 10 neurons in this case (since there are 10 possible digits, 0 through 9). Each output neuron corresponds to a specific digit, and the network's goal is to activate the correct output neuron for a given input image.")
st.write("4. **Training:** To make the neural network good at recognizing digits, it needs to be trained using a dataset of labeled examples. During training, the network compares its predictions to the correct answers and adjusts its internal parameters (weights and biases) to minimize the error between the predicted and actual values. This is done using optimization algorithms.")
st.write("5. **Prediction:** Once the neural network has been trained, it can take new, unlabeled images of handwritten digits and make predictions about which digit each image represents. It does this by forward-propagating the input data through the network, and the output neuron with the highest activation value indicates the predicted digit.")
st.write("So, in a nutshell, a neural network is a program that can learn to recognize patterns in data, such as handwritten digits in the MNIST dataset, by adjusting its internal parameters through training to make accurate predictions.")

# Load MNIST dataset
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

# Build and train a neural network
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim / 8)
output_dim = len(np.unique(mnist.target))

# Define default values
num_train_images = 52500
hidden_neurons = 256
activation_function = "ReLU"
use_dropout = True
learning_rate = 0.1
optimizer = "SGD"

st.divider()
# Display interactive elements with default values
st.subheader("Interactive Settings")
st.write("**Number of Training images:** This slider allows you to select the number of training images used to train the neural network. You can vary the training set size to observe how it affects the model's performance. Increasing the number of training images may improve accuracy, while reducing it can lead to faster training but potentially lower accuracy.")
num_train_images = st.slider("Number of Training Images", min_value=100, max_value=len(X_train), step=100, value=num_train_images)

st.write("**Number of Hidden Neurons:** This slider lets you adjust the number of neurons (units) in the hidden layer of the neural network. The hidden layer is a critical component of the network. More neurons can potentially capture more complex patterns in the data, but it can also increase training time and require more data. Fewer neurons may lead to underfitting.")
hidden_neurons = st.slider("Number of Hidden Neurons", min_value=32, max_value=256, step=32, value=hidden_neurons)

st.write("**Activation Function:** You can choose the activation function used in the hidden layer of the neural network. The activation function defines the output of a neuron given its input. The three options are:")
st.write("1. ReLU (Rectified Linear Unit): Commonly used for hidden layers. It introduces non-linearity.")
st.write("2. Sigmoid: Suits binary classification problems well. It squashes the output between 0 and 1.")
st.write("3. Tanh: Also known as hyperbolic tangent activation function. It outputs values between -1 and 1. It can be used for both binary and multiclass classification.")
activation_function = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"], index=0 if activation_function == "ReLU" else 1 if activation_function == "Sigmoid" else 2)

st.write("**Using Dropout:** Dropout is a regularization technique that helps prevent overfitting. When this checkbox is selected, dropout is applied during training. Dropout randomly sets a fraction of the neurons' outputs to zero during each training step, which encourages the network to be more robust and generalize better.")
use_dropout = st.checkbox("Use Dropout", value=use_dropout)

st.write("**Learning Rate:** Learning rate determines the step size in updating the model's parameters during training. You can adjust the learning rate to control how quickly or slowly the model learns. A smaller learning rate may lead to slower but more stable convergence, while a larger learning rate may lead to faster convergence but with the risk of overshooting the optimal solution.")
learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=1.0, step=0.01, value=learning_rate)

st.write("**Optimizer:** The optimizer is responsible for updating the model's parameters based on the calculated gradients during training. You can choose between different optimization algorithms:")
st.write("1. Stochastic Gradient Descent (SGD): A basic optimization algorithm.")
st.write("2. Adam: A popular optimization algorithm that adapts the learning rate during training.")
st.write("3. RMSprop (Root Mean Square Propagation): An adaptive learning rate method that can help overcome some of the limitations of basic SGD.")
optimizer = st.radio("Optimizer", ["SGD", "Adam", "RMSprop"], index=0 if optimizer == "SGD" else 1 if optimizer == "Adam" else 2)
X_train_subset, y_train_subset = X_train[:num_train_images], y_train[:num_train_images]

# Rebuild and retrain the model with updated settings
class CustomClassifierModule(nn.Module):
    def __init__(self, input_dim=mnist_dim, hidden_dim=hidden_neurons, output_dim=output_dim, dropout=0.5):
        super(CustomClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        if activation_function == "ReLU":
            X = F.relu(self.hidden(X))
        elif activation_function == "Sigmoid":
            X = torch.sigmoid(self.hidden(X))
        elif activation_function == "Tanh":
            X = torch.tanh(self.hidden(X))

        if use_dropout:
            X = self.dropout(X)

        X = F.softmax(self.output(X), dim=-1)
        return X

# Define the function to train the model
def train_model():
    # Use the values from interactive elements or defaults
    optimizer_mapping = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "RMSprop": optim.RMSprop,
    }
    optimizer_cls = optimizer_mapping[optimizer]
    net = NeuralNetClassifier(
        module=CustomClassifierModule,
        max_epochs=20,
        lr=learning_rate,
        optimizer=optimizer_cls,  # Use the mapped optimizer class
        device=device,
    )
    net.fit(X_train_subset, y_train_subset)
    return net


if st.button("Train Model"):
    net = train_model()
else:
    net = None

st.divider()
# Display a selection of training images and their labels
st.header("Sample of Training Images and Labels")
plot_example(X_train, y_train)

st.divider()
# Display training statistics in a table
st.subheader("Training Statistics")
st.markdown("1. *Epoch:* The number of times the model has been trained on the entire training dataset.")
st.markdown("2. *Train Loss:* The training loss at the end of each epoch, indicating how well the model is fitting the training data.")
st.markdown("3. *Valid Acc:* Validation accuracy, the accuracy of the model on the validation dataset at the end of each epoch.")
st.markdown("4. *Valid Loss:* Validation loss, the loss on the validation dataset, measuring how well the model generalizes.")
st.markdown("5. *Duration:* The time taken to complete each epoch.")
train_stats = None

if net is not None:
    train_stats = net.history[:, ['epoch', 'train_loss', 'valid_acc', 'valid_loss', 'dur']]
st.table(train_stats)

# Plot the training statistics
st.subheader("Training Statistics Plot")
if train_stats is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ax.plot(x, net.history[:, ['train_loss']], label='Train Loss')
    ax.plot(x, net.history[:, ['valid_acc']], label='Valid Acc')
    ax.plot(x, net.history[:, ['valid_loss']], label='Valid Loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Training Statistics')
    ax.legend()
    st.pyplot(fig)

st.divider()
y_pred=None
# Before making predictions
if net is not None:
    # Make predictions and calculate accuracy
    y_pred = net.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.divider()
    st.header("Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write("Accuracy is the number of correctly predicted data points out of all the data points.")
    st.divider()
else:
    st.warning("The model has not been trained. Please configure the model and train it.")

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

st.divider()
# Visualize misclassified images
error_mask = y_pred != y_test
st.write("Misclassified Images")
plot_example(X_test[error_mask], y_pred[error_mask])
st.divider()


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
    st.header(f"The model predicts that the digit is: {prediction[0]}")
st.divider()

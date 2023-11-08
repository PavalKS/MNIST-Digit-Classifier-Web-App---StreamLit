import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from streamlit_drawable_canvas import st_canvas
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import torch.optim as optim
from skorch import NeuralNetClassifier
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import io
import seaborn as sns

st.title("MNIST Digit Classifier")
st.write("Authored by PAVAL KS - pavalsudhakar@gmail.com")
st.divider()

st.subheader("What is Machine Learning?")
st.write("Imagine you have a computer and you have to teach it to recognize and understand handwritten numbers. If someone writes a number, like '7,' on a piece of paper, this computer can figure out that it's the number 7. How does it do that? That's where machine learning comes in.")
st.image('ml.png')
st.write("1. **Collecting Examples:** To teach our computer to recognize numbers, we need to show it lots of examples. We gather many examples with different handwritten numbers, like 0, 1, 2, and so on. These are our training examples.")
st.write("2. **Learning from Examples:** Our computer doesn't know anything about numbers at first. It's like a blank slate. But we're going to show it these examples, one by one. When we show it a piece of paper with a number, the computer tries to understand the patterns in the way the number is written. It's like a student learning from a teacher.")
st.write("3. **Finding Patterns:** As the computer sees more and more examples, it starts to notice patterns. For example, it might see that the number 7 often has a long vertical line with a little hook at the top. It learns that this pattern usually means '7'.")
st.image('ml1.png')
st.write("4. **Making Predictions:** Now, when we show the computer a new, unseen piece of paper with a handwritten number, it can make a guess. It looks for the patterns it learned during its training and says, **Hmm, I think this looks like a 7.**")
st.write("5. **Feedback and Improvement:** Sometimes, the computer might make a mistake, and that's okay. We can tell it whether it's right or wrong. If it's wrong, the computer can adjust its understanding and get better over time. It learns from its mistakes, just like you do in school.")
st.write("So, machine learning is like teaching a computer how to recognize patterns in data, in this case, patterns in handwritten numbers. It's a bit like teaching a dog tricks or teaching a friend how to play a new game. The more examples it sees, the better it gets at making predictions. The MNIST handwritten digit recognition model is like our computer, and it's really good at recognizing handwritten numbers because it has seen thousands of examples and learned the patterns. Machine learning is used in many other cool ways too, like helping with medical diagnoses, recommending movies, and even self-driving cars!")
st.divider()

st.subheader(" Let's connect the explanation of machine learning with these additional technical concepts:")
st.write("- **Dataset:** Our collection of pieces of paper with handwritten numbers is our dataset. It's like having a big pile of homework for our computer to learn from.")
st.write("- **Train-Test Split:** Before we start teaching the computer, we need to make sure it learns well. So, we divide our dataset into two parts: the training set and the test set. The training set is like the practice problems, and the test set is like the final exam.")
st.write("- **Training:** This is the part where we show our computer the examples from the training set and teach it to recognize patterns in the numbers. It's like a teacher explaining how to solve math problems to a student.")
st.write("- **Training Statistics**: While training, our computer keeps track of its progress. It counts how many times it got a number right and how many times it got it wrong. These statistics help it improve, just like keeping track of your scores in a game helps you get better.")
st.write("- **Testing:** Once our computer has practiced enough, it's time to see how well it learned. We show it the test set, which has new, unseen examples, to see if it can correctly recognize the numbers. It's like taking a test in school.")
st.write("- **Evaluation:** After the test, we evaluate how well our Computer did. We want to know if it recognized the numbers correctly or if it made mistakes.")
st.write("- **Accuracy:** Accuracy is like your score in a game. If our computer has a high accuracy, it means it did a great job recognizing numbers. If it has a low accuracy, it means it made more mistakes.")
st.write("- **Optimizer:**  Our computer can be even smarter with the help of an optimizer. It's like a coach who guides our pen during training, making sure it learns faster and better.")
st.write("- **Learning Rate:** Learning rate is like the speed at which our computer learns. If it learns too fast (high learning rate), it might make mistakes, and if it learns too slowly (low learning rate), it might take forever to improve. So, it's about finding the right balance.")
st.write("- **Activation Function:** Inside our computer, there are small parts called neurons. Activation functions are like rules that help these neurons decide how excited or calm they should be. They help our pen make better predictions.")
st.write("- **Hidden Neurons:** Our computer doesn't just have one neuron; it has many hidden ones. These hidden neurons work together to recognize the patterns in numbers. Think of them as a team of experts who combine their knowledge to solve a tough problem.")
st.write("So, machine learning is like training our computer (the model) using a dataset of handwritten numbers, dividing it into training and test sets, teaching it to recognize patterns during training, testing its ability with new examples, evaluating its performance using metrics like accuracy, and making it smarter with optimization techniques like learning rate and activation functions, all while using hidden neurons to work together. Just like students in school learn from their teachers and tests, our computer learns from its dataset and tests to become really good at recognizing numbers.")
st.divider()

st.subheader("Here's a simplified explanation of how a neural network works:")
st.write("Neural networks are a type of computer program that's inspired by the way our brains work. They're used for a wide range of tasks, including things like recognizing handwritten digits in the MNIST dataset, which is a common example in the field of machine learning.")
st.write("**How a neural network sees and processes images:**")
# Introduction
st.write("Imagine you have a friend named Bob who loves to play with LEGO blocks. Bob is really good at recognizing different shapes and patterns made from LEGO blocks. Now, let's use Bob as an example to help understand how a neural network views images.")
# Basic Building Blocks
st.write("- **Basic Building Blocks**: Bob's brain is like a simple LEGO structure with just a few LEGO blocks. These basic blocks can only recognize simple shapes, like squares, circles, and triangles.")
# Connecting Blocks
st.write("- **Connecting Blocks:** When Bob looks at an image, he can't see it all at once. He takes the image and breaks it down into smaller pieces, like puzzle pieces. Each of these puzzle pieces has some simple shapes, and Bob's basic LEGO blocks try to figure out what's in each piece.")
# Passing Information
st.write("- **Passing Information:** Bob's brain is like a conveyor belt. The puzzle pieces with simple shapes travel through this belt, and each basic LEGO block examines them one by one.")
# Making Sense of the Whole
st.write("- **Making Sense of the Whole:** After all the pieces have been checked, Bob's brain starts to put everything back together. It's like solving a jigsaw puzzle. He combines the simple shapes and patterns he found in each piece to understand what the whole image is about.")
# Complex Images
st.write("- **Complex Images:** For complex images, Bob uses more LEGO blocks, and the conveyor belt passes more pieces. The more blocks he has, the better he can understand intricate patterns and details in the image.")
st.image('pixels.ppm', caption='A visual example of how images are repersented in pixels')
# Transition to Neural Networks
st.image('neuralnetwork1.png', caption='A Simple Neural Network Architecture', use_column_width=True)
st.write("Now, imagine a neural network as an advanced version of Bob's brain. Instead of LEGO blocks, it has special units called neurons. These neurons work together in layers, just like Bob's conveyor belt, to analyze images.")
# Input Layer
st.write("- **Input Layer:** This is where the image is divided into smaller pieces, kind of like the puzzle pieces for Bob. Each neuron in the input layer looks at a specific part of the image in pixels.")
# Hidden Layers
st.write("- **Hidden Layers:** These are layers with lots of neurons that process the information from the input layer. They're like the conveyor belt, each neuron handling a piece of the image.")
# Output Layer
st.write("- **Output Layer:** Here, the neural network combines the information from all the previous layers to make sense of the whole image. It tells us what the network thinks the image contains. For example, if it's a handwritten '7,' the output neuron might say, 'Hey, I think this is the number 7.'")
# Conclusion
st.write("Just like Bob gets better at recognizing complex LEGO patterns by having more LEGO blocks and layers of conveyor belts, neural networks become more powerful by having more layers and neurons. They can recognize not only simple shapes but also complex and detailed patterns in images.")
st.write("So, a neural network 'sees' images by breaking them down into simpler parts and then gradually putting those parts together to understand what's in the picture. It's like having a team of experts, each looking at a piece of the puzzle and collaborating to solve the big picture puzzle.")


# Explanation of the working of the neural network
st.subheader("How the Neural Network Works")
st.image('neuralnetwork.png', caption='Neural Network Architecture', use_column_width=True)
st.write("The neural network is trained on the MNIST dataset, which consists of 28x28 pixel images of handwritten digits (0-9).")
st.write("The neural network uses one hidden layer with 98 neurons and an output layer with 10 neurons, each representing a digit.")
st.write("The images are preprocessed to ensure consistency with the training data.")
st.write("The model is trained using the training data, and accuracy is measured on the test data.")
st.divider()

# Function to preprocess a canvas drawing
def preprocess_canvas_drawing(uploaded_canvas):
    if uploaded_canvas is not None:
        # Save the canvas drawing as an image
        if uploaded_canvas.image_data is not None:
            cv2.imwrite("canvas_img.jpg", uploaded_canvas.image_data)

        # Load the saved image
        image = Image.open("canvas_img.jpg")
        # Preprocess and predict
        image = image.convert("L")  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to MNIST size
        image = transforms.ToTensor()(image)
        image = 1 - image  # Invert black and white
        image = image.view(1, -1)  # Reshape the image to (1, 784)

        return image
    
def preprocess_image(uploaded_image):
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        image = image.convert("L")  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to MNIST size
        image = transforms.ToTensor()(image)
        image = 1 - image  # Invert black and white
        image = image.view(1, -1)  # Reshape the image to (1, 784)

        return image
    
st.subheader("Let's try to teach a machine to recognize numbers like this!")
st.write("First, upload or draw an image of a number you want the computer to recognize.")
st.subheader("Image Upload or Hand-Draw")

# Option to upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Option to hand-draw an image using st_canvas
uploaded_canvas = st_canvas(
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    update_streamlit=True,
    width=300,
    height=300,
    drawing_mode="freedraw",
)

# Check if an image was uploaded
if uploaded_image is not None:
    image_data = preprocess_image(uploaded_image)
elif uploaded_canvas is not None:
    image_data = preprocess_canvas_drawing(uploaded_canvas)


@st.cache_data
# Load MNIST dataset with caching
def load_mnist_data():
    mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int64')
    outputd = mnist.target
    return X, y, outputd

X, y, outputd = load_mnist_data()

# Preprocess data with caching
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(X, y)


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
output_dim = len(np.unique(outputd))

# Define default values
num_train_images = 52500
hidden_neurons = 256
activation_function = "ReLU"
use_dropout = True
learning_rate = 0.1
optimizer = "SGD"

st.divider()
# Display interactive elements with default values
st.subheader("Let's train the computer to recognise numbers as we discussed earlier")
st.write("modify the parameters below and see how it affects the computer's ability to recognise numbers! (Accuracy)")
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


st.subheader("Things to be cautious about while training a neural network:")
st.write("- Underfitting: Imagine you have a friend who's learning to play basketball. This friend is so cautious that they never take a shot at the basket. They don't practice much and always play it safe. As a result, they rarely make any baskets because they are too scared to try. In the world of machine learning, this is like underfitting. It happens when a model is too simple and can't understand the data properly. It's like not trying hard enough to learn, just like our cautious basketball player.")
st.write(" - Overfitting: On the other hand, think of another friend who's practicing basketball. They try to make incredibly difficult shots all the time. They practice so much that they start to make shots even from the most challenging angles. However, when it comes to a real game, they can't perform well because they've only practiced those tricky shots. They're too specialized and can't adapt to the regular game. In machine learning, this is like overfitting. It happens when a model is too complex and learns the training data too well but struggles when faced with new, unseen data.")
st.write(" So, underfitting is like not learning enough and being too simple, while overfitting is like learning too much from your training and being too specialized. The goal in machine learning is to find the right balance, just like becoming a good basketball player who can make both easy and challenging shots. This balance helps the model perform well not only on the training data but also on new data it hasn't seen before.")


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

@st.cache_data
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

# Define the function to display training statistics
@st.cache_data
def display_training_statistics(_net):
    if _net is not None:
        train_stats = _net.history[:, ['epoch', 'train_loss', 'valid_acc', 'valid_loss', 'dur']]
        st.table(train_stats)

        # Plot the training statistics
        st.subheader("Training Statistics Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ax.plot(x, _net.history[:, ['train_loss']], label='Train Loss')
        ax.plot(x, _net.history[:, ['valid_acc']], label='Valid Acc')
        ax.plot(x, _net.history[:, ['valid_loss']], label='Valid Loss')
        ax.set_xlabel('Epoch')
        ax.set_title('Training Statistics')
        ax.legend()
        st.pyplot(fig)


y_pred = net.predict(X_test)
# Define the function to calculate accuracy and display confusion matrix
@st.cache_data
def calculate_accuracy_and_display_confusion_matrix(_net, X_test, y_test):
    if _net is not None:
        y_pred = _net.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.divider()
        st.header("Accuracy")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")
        st.write("Accuracy is the number of correctly predicted data points out of all the data points.")
        st.divider()
        
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

# Define the function to visualize misclassified images
@st.cache_data
def visualize_misclassified_images(X_test, y_pred, y_test):
    error_mask = y_pred != y_test
    st.write("Misclassified Images")
    plot_example(X_test[error_mask], y_pred[error_mask])

# Display training statistics
display_training_statistics(net)

# Calculate accuracy and display confusion matrix
calculate_accuracy_and_display_confusion_matrix(net, X_test, y_test)

# Visualize misclassified images
visualize_misclassified_images(X_test, y_pred, y_test)

# Preprocess and predict
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

if uploaded_canvas is not None:
    image = Image.open("canvas_img.jpg")
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

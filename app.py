import streamlit as st  # Import the Streamlit library for creating web apps
from PIL import Image  # Import the Python Imaging Library (PIL) for handling images
import torch  # Import PyTorch for working with deep learning models
from torchvision import transforms  # Import the transforms module from torchvision for image transformations
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # Import the Faster R-CNN model

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)  # Load the Faster R-CNN model pre-trained on COCO dataset
model.eval()  # Set the model to evaluation mode (disables training-specific features)

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor
])

# COCO classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]  # List of COCO instance category names

def main():
    st.title("Image Component Identifier")  # Display the title of the app
    st.write("Upload an image to identify its components")  # Display a description

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])  # Create a file uploader widget

    if uploaded_file is not None:  # Check if a file is uploaded
        image = Image.open(uploaded_file).convert("RGB")  # Open the uploaded image and convert it to RGB format
        st.image(image, caption='Uploaded Image.', use_column_width=True)  # Display the uploaded image with a caption

        if st.button('Analyse Image'):  # Create a button that triggers the identification process
            # Perform object detection
            components = identify_components(image)  # Call the function to identify components in the image
            st.write("Identified components in the image:")  # Display a heading for the results
            st.write(components)  # Display the list of identified components

def identify_components(image):
    # Transform the image
    image_tensor = transform(image).unsqueeze(0)  # Transform the image to a tensor and add a batch dimension

    # Perform object detection
    with torch.no_grad():  # Disable gradient calculation (useful for inference to save memory and computation)
        outputs = model(image_tensor)  # Get the model predictions for the image tensor

    # Extract the predicted class labels
    pred_classes = outputs[0]['labels'].numpy()  # Extract the predicted class labels and convert them to a NumPy array
    pred_scores = outputs[0]['scores'].detach().numpy()  # Extract the predicted scores and convert them to a NumPy array

    # Get the class labels with high scores
    keep = pred_scores > 0.9  # Filter out predictions with a score lower than 0.9
    labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred_classes[keep]]  # Map the filtered class labels to their names

    return labels  # Return the list of identified component labels

if __name__ == "__main__":
    main()  # Call the main function to run the app

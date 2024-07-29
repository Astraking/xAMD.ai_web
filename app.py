import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import gdown

# Set page config
st.set_page_config(page_title="AMD Screening App", layout="wide")

# Define model classes
class BinaryClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(BinaryClassifier, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)
        x = self.sigmoid(x)
        return x

class AMDModel(nn.Module):
    def __init__(self, num_classes=2):
        super(AMDModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)
        return x

@st.cache_resource
def load_model(model_class, path):
    try:
        model = model_class()
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def download_model(model_id, model_path):
    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_path}..."):
            gdown.download(f'https://drive.google.com/uc?id={model_id}', model_path, quiet=False)

# Define model paths
retinal_model_path = 'binary_classifier.pth'
amd_model_path = 'amd_model.pth'

# Download models
download_model('1nlcoXT4u06jSGVFDKZU5G4gbY0IJlWxr', retinal_model_path)
download_model('1D1WZXSRvFJbarBhn1WGq01Xqd11jEUvw', amd_model_path)

# Load models
retinal_model = load_model(BinaryClassifier, retinal_model_path)
amd_model = load_model(AMDModel, amd_model_path)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RetinalImagePipeline:
    def __init__(self, binary_model, amd_model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.binary_model = binary_model.to(self.device)
        self.amd_model = amd_model.to(self.device)

    def preprocess(self, image):
        image = Image.open(image).convert('RGB')
        return transform(image).unsqueeze(0).to(self.device)

    def is_retinal_image(self, image_tensor):
        with torch.no_grad():
            output = self.binary_model(image_tensor)
            prediction = output.item()
        return prediction > 0.5

    def detect_amd(self, image_tensor):
        with torch.no_grad():
            output = self.amd_model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
        return prediction, confidence

    def generate_gradcam(self, model, image_tensor, target_layer):
        model.eval()
        gradients = []
        activations = []

        def save_gradient(grad):
            gradients.append(grad)

        def forward_hook(module, input, output):
            output.register_hook(save_gradient)
            activations.append(output)
            return output

        hook = target_layer.register_forward_hook(forward_hook)
        output = model(image_tensor)
        hook.remove()

        target_class = torch.argmax(output, dim=1).item()
        model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        gradients = gradients[0].cpu().data.numpy()
        activations = activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(2, 3))
        gradcam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            gradcam += w * activations[0, i, :, :]

        gradcam = np.maximum(gradcam, 0)
        gradcam = cv2.resize(gradcam, (224, 224))
        gradcam = gradcam - gradcam.min()
        gradcam = gradcam / gradcam.max()

        return gradcam

    def visualize_gradcam(self, image_path, gradcam):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_np = np.array(image)

        heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_image = heatmap + np.float32(image_np) / 255
        cam_image = cam_image / np.max(cam_image)

        st.image(np.uint8(255 * cam_image), caption='Grad-CAM', use_column_width=True)

    def run(self, image):
        image_tensor = self.preprocess(image)
        if self.is_retinal_image(image_tensor):
            amd_result, confidence = self.detect_amd(image_tensor)
            gradcam = self.generate_gradcam(self.amd_model, image_tensor, self.amd_model.features[8])
            self.visualize_gradcam(image, gradcam)
            result_text = "AMD Detected" if amd_result == 0 else "No AMD Detected"
            return f"{result_text} with probability {confidence:.4f}"
        else:
            return "Not a Retinal Image"

# Create a pipeline instance
pipeline = RetinalImagePipeline(retinal_model, amd_model)

# Streamlit app interface
st.title("AMD Screening App")

# Navigation
menu = ["Home", "Upload Image", "About AMD"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.header("Welcome to the AMD Screening App")
    st.write("""
    This application allows you to upload retinal images to screen for Age-related Macular Degeneration (AMD).
    Please use the sidebar to navigate through the app.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Macula.svg/1200px-Macula.svg.png", width=300, caption="Illustration of the macula")

elif choice == "Upload Image":
    st.header("Upload Retinal Image")
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                with st.spinner("Processing image..."):
                    result = pipeline.run(uploaded_file)

                st.write(result)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

elif choice == "About AMD":
    st.header("About Age-related Macular Degeneration (AMD)")
    st.write("""
    Age-related macular degeneration (AMD) is an eye disease that can blur your central vision. It happens when aging causes damage to the macula — the part of the eye that controls sharp, straight-ahead vision. 
    The macula is part of the retina (the light-sensitive tissue at the back of the eye). AMD is a common condition — it's a leading cause of vision loss for older adults.
    
    ### Types of AMD
    1. **Dry AMD**: This is the most common type. It happens when parts of the macula get thinner with age and tiny clumps of protein called drusen grow. You slowly lose central vision. There's no way to treat dry AMD yet.
    2. **Wet AMD**: This type is less common but more serious. It happens when new, abnormal blood vessels grow under the retina. These vessels may leak blood or other fluids, causing scarring of the macula. You lose vision faster with wet AMD than with dry AMD.

    ### Risk Factors
    - Age (50 and older)
    - Family history and genetics
    - Race (more common in Caucasians)
    - Smoking
    - Cardiovascular disease

    ### Symptoms
    - Blurry or fuzzy vision
    - Straight lines appear wavy
    - Difficulty seeing in low light
    - Difficulty recognizing faces

    ### Prevention and Management
    While there's no cure for AMD, some lifestyle choices can help reduce the risk:
    - Avoid smoking
    - Exercise regularly
    - Maintain normal blood pressure and cholesterol levels
    - Eat a healthy diet rich in green leafy vegetables and fish
    - Protect your eyes from UV light

    For more information, visit [NEI AMD Information](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration).
    """)
    st.image("https://images.app.goo.gl/8oGiXJms5d7vvT9M8", width=300, caption="Cross-section of the macula")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by xAMD.ai Inc.")

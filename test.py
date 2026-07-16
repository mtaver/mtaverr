import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="🖼️",
    layout="centered"
)

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [
    "Buildings",
    "Forest",
    "Glacier",
    "Mountain",
    "Sea",
    "Street"
]

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # Same classifier used during training
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(CLASS_NAMES))
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            "intel_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()

# ==========================================================
# TITLE
# ==========================================================

st.title("🖼️ Intel Image Classification")

st.write(
    """
Upload an image and let the trained AI classify it.

### Supported Classes

- 🏢 Buildings
- 🌳 Forest
- 🧊 Glacier
- ⛰️ Mountain
- 🌊 Sea
- 🛣️ Street
"""
)

# ==========================================================
# FILE UPLOADER
# ==========================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(img)

        probabilities = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = CLASS_NAMES[prediction.item()]
    confidence_score = confidence.item() * 100

    st.success(
        f"Prediction: **{predicted_class}**"
    )

    st.info(
        f"Confidence: **{confidence_score:.2f}%**"
    )

    # ======================================================
    # PROBABILITIES
    # ======================================================

    st.subheader("Prediction Probabilities")

    for i, cls in enumerate(CLASS_NAMES):

        probability = probabilities[0][i].item()

        st.write(
            f"**{cls}** — {probability*100:.2f}%"
        )

        st.progress(probability)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Model Information")

st.sidebar.markdown(
    """
### Framework
PyTorch

### Architecture
ResNet18

### Dataset
Intel Image Classification

### Number of Classes
6

### Image Size
224 × 224

### Deployment
Streamlit

### Device
{}
""".format(device)
)

st.sidebar.success("Model Loaded Successfully")
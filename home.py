import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes raised")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
# --------------------------
# 1. Define Model Architecture
# --------------------------
class SkinCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


# --------------------------
# 2. Load Model and Classes
# --------------------------

@st.cache_resource
def load_model_and_classes():
    import torch.serialization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.serialization.add_safe_globals([SkinCNN])

    try:
        model = torch.load("skin_cnn_full_model.pth", map_location=device, weights_only=False)
        print("Loaded best model from skin_cnn_full_model.pth")
    except FileNotFoundError:
        model = torch.load("skincnn.pth", map_location=device, weights_only=False)
        print("Loaded fallback model from skincnn.pth")

    model.eval()

    # Load class names
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)

    return model, class_names, device

model, class_names, device = load_model_and_classes()


# --------------------------
# 3. Streamlit UI Setup
# --------------------------
st.title("Skin Cancer Classification")

st.write("""
Upload a skin lesion image to predict the type of skin cancer.

**Possible Classes:**
  
- Melanoma (MEL)
- Vascular lesions (VASC)
- Actinic keratoses (AK)
- Basal cell carcinoma (BCC)
- Benign keratosis-like lesions (BKL)
- Dermatofibroma (DF)
- Melanocytic nevi (NV)

""")

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probabilities, 3)
    return top_prob[0].cpu(), top_class[0].cpu()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing... Please wait"):
            top_prob, top_class = predict_image(image)

        sorted_indices = torch.argsort(top_prob, descending=True)
        top_prob = top_prob[sorted_indices]
        top_class = top_class[sorted_indices]

        best_class = class_names[top_class[0]]
        best_prob = top_prob[0].item() * 100

        st.markdown(
            f"<h2 style='color:#1f77b4; font-size:32px; font-weight:700;'>Predicted: {best_class}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='font-size:22px; color:gray;'>Confidence: <b>{best_prob:.2f}%</b></p>",
            unsafe_allow_html=True
        )

        st.subheader("Predictions:")
        for prob, idx in zip(top_prob, top_class):
            st.markdown(
                f"<p style='font-size:18px;'>• <b>{class_names[idx]}</b> — {prob.item()*100:.2f}%</p>",
                unsafe_allow_html=True
            )

        fig, ax = plt.subplots()
        ax.barh([class_names[i] for i in top_class], top_prob)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        for i, (p, cls) in enumerate(zip(top_prob, top_class)):
            ax.text(p.item(), i, f"{p.item()*100:.1f}%", va="center")
        st.pyplot(fig)

else:
    st.info("Please upload an image to begin.")

st.markdown("---")
st.caption("Jatin Nath — PyTorch + Streamlit")


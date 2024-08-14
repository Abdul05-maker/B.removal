import streamlit as st
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from PIL import Image

# Load the model
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

st.set_page_config(page_title="Background Removal App", page_icon=":frame_photo:", layout="wide")
st.title("Background Removal App")
st.markdown("<h2 style='text-align: center;'>Created by Abdullah Nawaz</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to remove its background and see the magic happen!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    orig_im = Image.open(uploaded_file).convert("RGB")
    orig_im_np = np.array(orig_im)
    orig_im_size = orig_im_np.shape[0:2]

    # Define the input size for the model
    model_input_size = [512, 512]  # Adjust this based on your model requirements

    # Preprocess the image
    image = preprocess_image(orig_im_np, model_input_size).to(device)

    # Perform inference
    result = model(image)

    # Postprocess the result
    result_image = postprocess_image(result[0][0], orig_im_size)

    # Create a PIL image from the result
    result_pil_image = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", result_pil_image.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_im.convert("RGBA"), mask=result_pil_image)

    # Display the result
    st.image(no_bg_image, caption="Image with Background Removed", use_column_width=True)

st.markdown("<footer style='text-align: center;'>Thank you for using the Background Removal App!</footer>", unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 🚀 Load Models with raw string paths
chest_model = tf.keras.models.load_model(r'E:\cnn_project\Chest-xray_model.h5')
brain_model = tf.keras.models.load_model(r'E:\cnn_project\brain_tumour_model.h5')

# 🔖 Class Labels
chest_labels = ["Normal", "Pneumonia"]
brain_labels = ["Cancer", "Not Cancer"]

# 🧠 Prediction Function
def predict_image(model, image, target_size, labels):
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0

    # Convert grayscale to RGB
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # If image has only one color channel (e.g., shape (64, 64, 1)), convert to 3 channels
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array]*3, axis=-1)

    img_array = img_array.reshape(1, target_size[0], target_size[1], 3)
    prediction = model.predict(img_array)[0][0]
    label = labels[1] if prediction >= 0.5 else labels[0]
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return f"{label} ({confidence:.2f} confidence)"

# 🌐 Streamlit UI
st.title("Medical Image Classifier 🔍")
st.sidebar.title("Select Model")

model_type = st.sidebar.selectbox("Choose the model to run:", ("Chest X-Ray", "Brain Tumor"))

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_type == "Chest X-Ray":
        result = predict_image(chest_model, image, (64, 64), chest_labels)
    else:
        result = predict_image(brain_model, image, (64, 64), brain_labels)

    st.success(f"Prediction: {result}")
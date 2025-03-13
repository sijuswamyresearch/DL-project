import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import tensorflow as tf
from tensorflow.keras import layers
import base64

# Define custom layers (same as in your model)
class SobelEdgeLayer(layers.Layer):
    """Custom layer to compute Sobel edge magnitude."""
    def __init__(self, **kwargs):
        super(SobelEdgeLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)
        sobel = tf.image.sobel_edges(inputs)
        sobel_mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1) + 1e-7)
        return sobel_mag

class EdgeAttention(layers.Layer):
    """Custom layer for Edge Attention."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sobel_edge_layer = SobelEdgeLayer()
        self.conv = layers.Conv2D(1, 1, activation='sigmoid', kernel_initializer='he_uniform')
        self.multiply = layers.Multiply()
        self.input_channels = None

    def build(self, input_shape: tf.TensorShape):
        self.input_channels = input_shape[-1]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        sobel_mag = self.sobel_edge_layer(x)
        att = self.conv(sobel_mag)
        att = tf.tile(att, multiples=[1, 1, 1, self.input_channels])
        return self.multiply([x, att])

# Register custom layers
tf.keras.utils.get_custom_objects().update({
    'SobelEdgeLayer': SobelEdgeLayer,
    'EdgeAttention': EdgeAttention
})

# Load the pre-trained model
@st.cache_resource
def load_model():
    #return tf.keras.models.load_model('best_model_generator.h5')
     return tf.keras.models.load_model('XGANmodel.h5')
# Preprocess the uploaded image
def preprocess_image(image, target_size=(128, 128)):
    # Convert to grayscale and resize
    image = np.array(image.convert('L'))
    image = cv2.resize(image, target_size)
    # Normalize to [0, 1]
    image = image / 255.0
    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=(0, -1)).astype(np.float32)
    return image

# Generate edge map
def generate_edge_map(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_map = edge_map / np.max(edge_map)  # Normalize to [0, 1]
    return edge_map

# Function to encode local SVG file to base64
def svg_to_base64(file_path):
    with open(file_path, "rb") as f:
        svg_data = f.read()
    return base64.b64encode(svg_data).decode("utf-8")

# Streamlit app
def main():
    # Custom CSS for the title bar
    st.markdown(
        """
        <style>
        .title-bar {
            background-color: crimson;
            padding: 15px;
            display: flex;
            align-items: right;
            color: white;
            font-size: 54px;
            font-weight: bold;
        }
        .title-bar img {
            height: 70px;
            margin-right: 15px;
        }
        .details {
            margin-top: 10px;
            font-size: 12px;
            color: #f0c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Encode the local SVG logo to base64
    logo_base64 = svg_to_base64("amrita_logo.svg")

    # Title bar with logo and project name
    st.markdown(
        f"""
        <div class="title-bar">
            <img src="data:image/svg+xml;base64,{logo_base64}" alt="Logo">
                  X-GAN Project
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Details of author and supervisor
    st.markdown(
        """
        <div class="details">
            <b>Ideated by:</b> Siju K S<br>
            <b>Course Coordinator:</b> Dr. Sowmya V.<br>
            <b>Supervised by:</b> Dr. Vipin V, Amrita School of AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Upload a noisy medical image to denoise it and visualize the edge map.")

    # Load the model
    model = load_model()

    # Upload image
    uploaded_file = st.file_uploader("Choose a noisy image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Noisy Image", use_container_width=True)

        # Preprocess the image
        input_image = preprocess_image(image)

        # Denoise the image
        denoised_image = model.predict(input_image).squeeze()

        # Generate edge map
        edge_map = generate_edge_map(denoised_image)

        # Calculate PSNR and SSIM (assuming original image is available)
        original_image = np.array(image.convert('L')) / 255.0
        original_image = cv2.resize(original_image, (128, 128))
        psnr_value = psnr(original_image, denoised_image, data_range=1.0)
        ssim_value = ssim(original_image, denoised_image, data_range=1.0)

        # Display results
        st.write("### Results")

        # Create three columns for the images
        col1, col2, col3 = st.columns(3)

        # Display noisy image in the first column
        with col1:
            st.image(image, caption="Noisy Image", use_container_width=True, clamp=True)

        # Display denoised image in the second column
        with col2:
            st.image(denoised_image, caption="Denoised Image", use_container_width=True, clamp=True)

        # Display edge map in the third column
        with col3:
            st.image(edge_map, caption="Edge Map", use_container_width=True, clamp=True)

        # Display metrics
        #st.write("### Performance Metrics")
        #st.write(f"**PSNR:** {psnr_value:.2f}")
        #st.write(f"**SSIM:** {ssim_value:.2f}")

if __name__ == "__main__":
    main()
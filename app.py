import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import random

st.title("Image Viewer")

# Sidebar for file upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])


# Function to resize image
def resize_image(image, max_width=1920, max_height=1080):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            # Landscape image
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            # Portrait image
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image


# Function to create histogram with random sampling
def create_histogram(image_np, channels=['R', 'G', 'B'], sample_size=100000):
    fig = go.Figure()
    if 'R' in channels:
        r_sample = random.sample(list(image_np[:, :, 2].ravel()), min(sample_size, image_np[:, :, 2].size))
        fig.add_trace(go.Histogram(x=r_sample, nbinsx=256, name='Red', marker_color='red', opacity=0.5))
    if 'G' in channels:
        g_sample = random.sample(list(image_np[:, :, 1].ravel()), min(sample_size, image_np[:, :, 1].size))
        fig.add_trace(go.Histogram(x=g_sample, nbinsx=256, name='Green', marker_color='green', opacity=0.5))
    if 'B' in channels:
        b_sample = random.sample(list(image_np[:, :, 0].ravel()), min(sample_size, image_np[:, :, 0].size))
        fig.add_trace(go.Histogram(x=b_sample, nbinsx=256, name='Blue', marker_color='blue', opacity=0.5))

    # Layout settings for the graph
    fig.update_layout(
        title='Histogram of R, G, B channels',
        xaxis_title='Intensity value',
        yaxis_title='Number of pixels',
        bargap=0.2,
        showlegend=True
    )
    return fig


if uploaded_file is not None:
    try:
        # Load the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize the image
        resized_image = resize_image(image)
        st.image(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), caption='Resized Image', use_column_width=True)

        # Check the number of channels
        if len(resized_image.shape) == 2:
            # Grayscale image
            st.write("This image is in grayscale.")

            # Create and display histogram for grayscale image
            gray_sample = random.sample(list(resized_image.ravel()), min(100000, resized_image.size))
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=gray_sample, nbinsx=256, name='Gray', marker_color='black'))

            fig.update_layout(
                title='Grayscale Histogram',
                xaxis_title='Intensity value',
                yaxis_title='Number of pixels',
                bargap=0.2,
                showlegend=True
            )

            # Display the histogram
            st.plotly_chart(fig)

        elif len(resized_image.shape) == 3:
            # Color image
            st.write("This image is in color. Displaying R, G, B channels.")

            # Separate channels
            R_channel = resized_image.copy()
            G_channel = resized_image.copy()
            B_channel = resized_image.copy()

            # Keep only R channel, set G and B to 0
            R_channel[:, :, 1] = 0  # Set G channel to 0
            R_channel[:, :, 0] = 0  # Set B channel to 0
            st.image(cv2.cvtColor(R_channel, cv2.COLOR_BGR2RGB), caption='R Channel (Red)', use_column_width=True)

            # Keep only G channel, set R and B to 0
            G_channel[:, :, 2] = 0  # Set R channel to 0
            G_channel[:, :, 0] = 0  # Set B channel to 0
            st.image(cv2.cvtColor(G_channel, cv2.COLOR_BGR2RGB), caption='G Channel (Green)', use_column_width=True)

            # Keep only B channel, set R and G to 0
            B_channel[:, :, 2] = 0  # Set R channel to 0
            B_channel[:, :, 1] = 0  # Set G channel to 0
            st.image(cv2.cvtColor(B_channel, cv2.COLOR_BGR2RGB), caption='B Channel (Blue)', use_column_width=True)

            # Create and display histogram for R, G, B channels
            fig = create_histogram(resized_image)
            st.plotly_chart(fig)

        else:
            st.write("Unknown image format.")

    except Exception as e:
        st.error(f"Error occurred while processing the image: {e}")

else:
    st.write("Please upload an image file.")

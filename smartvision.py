import streamlit as st
from utils import *
import numpy as np
from PIL import Image
from io import BytesIO

def main():
    st.title("Modular OpenCV Image Processor")
    st.write("Upload an image and apply various OpenCV operations")

    # Initialize session state
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
        st.session_state.skip_color_and_noise = False

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display original image
        original_image = read_image(uploaded_file)
        st.image(original_image, caption="Original Image", use_container_width=True)
        st.session_state.original_image = original_image


        st.sidebar.title("Image options")

        # Image color
        color = st.sidebar.selectbox(
            "Select an operation",
            ["Default", "Grayscale"],
            key='color'
        )

        # Add noise
        noise = st.sidebar.selectbox(
            "Select an operation",
            ["no noise", "salt & paper", "gaussian", "poisson"],
            key='noise'
        )
        if not st.session_state.skip_color_and_noise:
            if 'processed_image' not in st.session_state:
                st.session_state.processed_image = st.session_state.original_image
            if color == "Grayscale":
                st.session_state.processed_image = apply_grayscale(st.session_state.original_image)
            else:
                st.session_state.processed_image = st.session_state.original_image
            if noise == "salt & paper":
                st.session_state.processed_image = apply_salt_and_pepper_noise(st.session_state.processed_image)
            elif noise == "gaussian":
                st.session_state.processed_image = apply_gaussian_noise(st.session_state.processed_image)
            elif noise == "poisson":
                st.session_state.processed_image = apply_poisson_noise(st.session_state.processed_image)
        else:
            st.session_state.skip_color_and_noise = False

        st.image(st.session_state.processed_image, caption="After coloring and adding noise", use_container_width=True)

        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = st.session_state.original_image

        # List of filters to apply
        st.sidebar.title("transform Op's")

        kernel_size = st.sidebar.slider(
            "Select kernel size (must be odd):",
            min_value=1,
            max_value=11,
            value=3,
            step=2
        )

        if st.sidebar.button(
            "Low pass filter",
            key='low pass filter'
        ):
            st.session_state.skip_color_and_noise = True
            st.session_state.processed_image = apply_low_pass_filter(st.session_state.processed_image, kernel_size)

        if st.sidebar.button(
            "High pass filter",
            key='High pass filter'
        ):
            st.session_state.skip_color_and_noise = True
            st.session_state.processed_image = apply_high_pass_filter(st.session_state.processed_image, kernel_size)

        if st.sidebar.button(
            "Mean filter (gray image)",
            key='Mean filter (gray image)'
        ):
            st.session_state.skip_color_and_noise = True
            st.session_state.processed_image = apply_mean_filter(st.session_state.processed_image, kernel_size)

        if st.sidebar.button(
            "Average filter",
            key='Average filter'
        ):
            st.session_state.skip_color_and_noise = True
            st.session_state.processed_image = apply_average_filter(st.session_state.processed_image, kernel_size)

        if st.sidebar.button(
            "dilation filter",
            key='dilation filter'
        ):
            st.session_state.skip_color_and_noise = True
            st.session_state.processed_image = apply_dilation_filter(st.session_state.processed_image, kernel_size)

        if st.sidebar.button(
            "erosion filter",
            key='erosion filter'
        ):
            st.session_state.skip_color_and_noise = True
            st.session_state.processed_image = apply_erosion_filter(st.session_state.processed_image, kernel_size)

        if st.sidebar.button(
            "close filter",
            key='close filter'
        ):
            st.session_state.skip_color_and_noise = True
            st.session_state.processed_image = apply_close_filter(st.session_state.processed_image, kernel_size)

        if st.sidebar.button(
            "open filter",
            key='open filter'
        ):
            st.session_state.skip_color_and_noise = True
            st.session_state.processed_image = apply_open_filter(st.session_state.processed_image, kernel_size)

        st.sidebar.title("detection filters")

        edge_detection = st.sidebar.selectbox(  #doesnt edit the image
            "Edge detection filter",
            ["no filter",
            "Lablacian filter",
            "gaussian filter",
            "vert. sobel filter",
            "horiz. sobel filter",
            "vert. prewitt filter",
            "horiz. prewitt filter",
            "line detection",
            "circles detection"
            ]
        )

        if edge_detection == "Lablacian filter":
            st.session_state.skip_color_and_noise = True
            edged_image = apply_laplacian(st.session_state.processed_image)
        elif edge_detection == "gaussian filter":
            st.session_state.skip_color_and_noise = True
            edged_image = apply_gaussian(st.session_state.processed_image)
        elif edge_detection == "vert. sobel filter":
            st.session_state.skip_color_and_noise = True
            edged_image = apply_vert_sobel(st.session_state.processed_image)
        elif edge_detection == "horiz. sobel filter":
            st.session_state.skip_color_and_noise = True
            edged_image = apply_horiz_sobel(st.session_state.processed_image)
        elif edge_detection == "vert. prewitt filter":
            st.session_state.skip_color_and_noise = True
            edged_image = apply_vert_prewitt(st.session_state.processed_image)
        elif edge_detection == "horiz. prewitt filter":
            st.session_state.skip_color_and_noise = True
            edged_image = apply_horiz_prewitt(st.session_state.processed_image)
        elif edge_detection == "line detection":
            st.session_state.skip_color_and_noise = True
            edged_image = apply_line_detection(st.session_state.processed_image)
        elif edge_detection == "circles detection":
            st.session_state.skip_color_and_noise = True
            edged_image = apply_circles_detection(st.session_state.processed_image)
        else:
            edged_image = st.session_state.processed_image
        st.image(edged_image, caption="After appling filters", use_container_width=True)

        # Download button

        st.sidebar.title("\n\n\n")

        if st.sidebar.button("reset", key='reset'):
            st.session_state.processed_image = st.session_state.original_image

        if st.session_state.processed_image is not None:
            download_button(st.session_state.processed_image)

def read_image(uploaded_file):
    """Read uploaded image and convert to RGB format"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def download_button(image):
    """Create download button for processed image"""
    # Convert to PIL Image
    if len(image.shape) == 2:  # Grayscale
        img_pil = Image.fromarray(image)
    else:  # Color
        img_pil = Image.fromarray(image)
    
    # Save to bytes
    buf = BytesIO()
    img_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    
    st.sidebar.download_button(
        label="Download Processed Image",
        data=byte_im,
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )

if __name__ == "__main__":
    main()

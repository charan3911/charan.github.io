import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import cv2

st.set_page_config(
    page_title="The Codex of Living Numbers",
    page_icon="⚔",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    h1 {
        color: #d4af37;
        text-align: center;
        font-family: 'Cinzel', serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    h2, h3 {
        color: #b8860b;
    }
    .stButton>button {
        background: linear-gradient(145deg, #8b7355, #6b5345);
        color: #d4af37;
        border: 2px solid #d4af37;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 30px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background: linear-gradient(145deg, #6b5345, #8b7355);
        box-shadow: 0 5px 15px rgba(212, 175, 55, 0.3);
    }
    .prediction-box {
        background: rgba(0, 0, 0, 0.5);
        padding: 30px;
        border-radius: 10px;
        border: 2px solid #8b7355;
        text-align: center;
    }
    .big-prediction {
        font-size: 80px;
        color: #d4af37;
        font-weight: bold;
        text-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('mnist_cnn_model.keras')
        return model
    except:
        try:
            model = keras.models.load_model('mnist_cnn_model.h5')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

def preprocess_image(image):
    image = ImageOps.grayscale(image)
    
    img_array = np.array(image)
    
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    _, img_array = cv2.threshold(img_array, 50, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(img_array)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_array.shape[1] - x, w + 2 * margin)
        h = min(img_array.shape[0] - y, h + 2 * margin)
        
        img_array = img_array[y:y+h, x:x+w]
    
    height, width = img_array.shape
    
    if height > width:
        new_height = 22  
        new_width = int(22 * width / height)
    else:
        new_width = 22
        new_height = int(22 * height / width)
    
    img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_array
    
    canvas = canvas.astype('float32') / 255.0
    
    canvas = canvas.reshape(1, 28, 28, 1)
    
    return canvas

st.markdown("<h1>⚔ THE CODEX OF LIVING NUMBERS ⚔</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #b8860b;'>Advanced Digit Recognition System</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b8860b; font-style: italic;'>Powered by Padding-Robust CNN</p>", unsafe_allow_html=True)

st.markdown("---")

model = load_model()

if model is None:
    st.error("⚠ Model not found! Please ensure mnist_cnn_model.keras or mnist_cnn_model.h5 is in the same directory.")
    st.stop()

st.success("✓ Neural network loaded successfully! (Padding-Robust Model)")

tab1, tab2 = st.tabs(["📸 Upload Image", "✍ Draw Digit"])

with tab1:
    st.markdown("### 📜 Upload a Handwritten Digit")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### Processed (28×28)")
            processed = preprocess_image(image)
            st.image(processed.reshape(28, 28), use_container_width=True, clamp=True)
        
        with col3:
            if st.button("🔮 Predict Digit", key="predict_upload"):
                with st.spinner("Analyzing the ancient digits..."):
                    prediction = model.predict(processed, verbose=0)
                    predicted_digit = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_digit] * 100
                    
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                        <div class="big-prediction">{predicted_digit}</div>
                        <p style='color: #d4af37; font-size: 20px;'>
                        Confidence: {confidence:.1f}%
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        if st.button("📊 Show All Confidence Levels", key="confidence_upload"):
            prediction = model.predict(processed, verbose=0)
            st.markdown("### Confidence Levels")
            
            for digit in range(10):
                conf = prediction[0][digit] * 100
                conf = float(conf)
                st.progress(conf / 100, text=f"Digit {digit}: {conf:.2f}%")

with tab2:
    st.markdown("### ✍ Draw a Digit")
    st.info("📝 Use an external drawing tool to create a digit image, then upload it in the 'Upload Image' tab.")
    st.markdown("""
    *Instructions:*
    1. Use MS Paint, or any drawing tool
    2. Draw a digit (0-9) on a white background
    3. Save the image
    4. Upload it in the 'Upload Image' tab
    
    *Tips for better accuracy:*
    - Draw the digit large and centered
    - Use black or dark color on white background
    - Keep the image simple and clear
    - **The system now automatically centers and pads digits - even corner placements work!**
    """)

st.sidebar.markdown("## 📊 Model Information")
st.sidebar.markdown(f"""
- *Architecture:* CNN (Padding-Robust)
- *Layers:* 
  - Conv2D(32, 3×3) + ReLU + BatchNorm
  - MaxPooling2D(2×2) + Dropout(0.25)
  - Conv2D(64, 5×5) + ReLU + BatchNorm
  - MaxPooling2D(2×2) + Dropout(0.25)
  - Flatten
  - Dense(128) + ReLU + BatchNorm + Dropout(0.5)
  - Dense(10, Softmax)
- *Dataset:* MNIST + Augmented Data
- *Training Samples:* 80,000+
- *Test Accuracy:* ~98-99%
- *Special Features:*
  - Trained on various padding scenarios
  - Robust to corner-placed digits
  - Handles different positions & scales
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 How to Use")
st.sidebar.markdown("""
1. Upload an image of a handwritten digit
2. Click 'Predict Digit'
3. View the prediction and confidence
4. Try different digits!
5. Try digits at corners - it works!
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 Image Processing")
st.sidebar.markdown("""
- Auto-detects digit boundaries
- Centers digit on 28×28 canvas
- Adds proper padding (3px each side)
- Maintains aspect ratio
- Matches training preprocessing
- Works with corner-placed digits!
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚔ The Seven Kingdoms Trust This System")
st.sidebar.markdown("*Enhanced with Padding-Robust Training*")
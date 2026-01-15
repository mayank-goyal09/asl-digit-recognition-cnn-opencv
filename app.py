"""
🤟 ASL Digits Recognizer - CNN Deep Learning Application
A premium, interactive web application for recognizing American Sign Language digits (0-9)
Built with Streamlit and TensorFlow
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from streamlit_drawable_canvas import st_canvas

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="ASL Digits Recognizer | CNN AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for Premium UI
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header Styling */
    .hero-section {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #a5b4fc;
        font-weight: 400;
        margin-bottom: 20px;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 5px;
    }
    
    /* Card Styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e7ff;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Prediction Result Styling */
    .prediction-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(168, 85, 247, 0.3) 100%);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        border: 2px solid rgba(168, 85, 247, 0.4);
        margin: 20px 0;
    }
    
    .prediction-digit {
        font-size: 8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f472b6 0%, #8b5cf6 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .confidence-bar {
        height: 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .digit-label {
        color: #a5b4fc;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .confidence-value {
        color: #c4b5fd;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #c4b5fd;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        border: 2px dashed rgba(99, 102, 241, 0.4);
    }
    
    /* Stats Cards */
    .stats-container {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 20px;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px 30px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        flex: 1;
        min-width: 120px;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 5px;
    }
    
    /* Info Section */
    .info-section {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 20px 0;
        border-left: 4px solid #6366f1;
    }
    
    .info-text {
        color: #c4b5fd;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Sample Images Grid */
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-top: 15px;
    }
    
    .sample-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sample-item:hover {
        transform: scale(1.05);
        border-color: #6366f1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px 0;
        margin-top: 40px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #a5b4fc;
    }
    
    .footer a {
        color: #8b5cf6;
        text-decoration: none;
        font-weight: 600;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 10px;
    }
    
    /* Canvas container */
    .canvas-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 5px;
        margin: 10px 0;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        .prediction-digit {
            font-size: 5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model with Caching
# ============================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained CNN model"""
    try:
        model = tf.keras.models.load_model("models/asl_digits_cnn.keras")
        return model
    except:
        try:
            model = tf.keras.models.load_model("models/asl_digits_aug.keras")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

# ============================================
# Image Preprocessing
# ============================================
def preprocess_image(image, target_size=(64, 64)):
    """Preprocess uploaded image for prediction"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to model input size
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # Add channel and batch dimensions
    img_array = img_array.reshape(1, 64, 64, 1)
    
    return img_array

def preprocess_canvas(canvas_result):
    """Preprocess canvas drawing for prediction"""
    if canvas_result.image_data is not None:
        # Get image data and convert to PIL
        img_data = canvas_result.image_data
        
        # Convert RGBA to grayscale
        img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
        img = img.convert('L')
        
        # Invert colors (white background to black, black drawing to white)
        img_array = np.array(img)
        img_array = 255 - img_array
        img = Image.fromarray(img_array)
        
        # Resize
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        
        # Normalize and reshape
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 64, 64, 1)
        
        return img_array
    return None

# ============================================
# Load Sample Images
# ============================================
@st.cache_data
def load_sample_images():
    """Load sample images from the dataset"""
    try:
        X = np.load("X.npy")
        Y = np.load("Y.npy")
        
        # Ensure channel dimension
        if X.ndim == 3:
            X = X[..., np.newaxis]
        
        # Normalize
        if X.max() > 1.0:
            X = X.astype('float32') / 255.0
        
        samples = {}
        for digit in range(10):
            if Y.ndim == 2:
                mask = Y.argmax(axis=1) == digit
            else:
                mask = Y == digit
            digit_images = X[mask]
            if len(digit_images) > 0:
                samples[digit] = digit_images[:5]  # Get first 5 samples
        
        return samples
    except Exception as e:
        return None

# ============================================
# Prediction Function
# ============================================
def predict_digit(model, img_array):
    """Make prediction on preprocessed image"""
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    all_confidences = predictions[0] * 100
    return predicted_class, confidence, all_confidences

# ============================================
# Main Application
# ============================================
def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🤟 ASL Digits Recognizer</div>
        <div class="hero-subtitle">Deep Learning Powered American Sign Language Digit Recognition</div>
        <div>
            <span class="hero-badge">🧠 CNN Neural Network</span>
            <span class="hero-badge">📊 93%+ Accuracy</span>
            <span class="hero-badge">⚡ Real-time Prediction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI Model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please check if the model file exists.")
        return
    
    # Load samples
    samples = load_sample_images()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #e0e7ff; font-weight: 700;">🎯 How to Use</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-section">
            <div class="info-text">
                <strong>1.</strong> Choose input method<br>
                <strong>2.</strong> Draw or upload an ASL digit<br>
                <strong>3.</strong> Click 'Predict' button<br>
                <strong>4.</strong> See AI prediction results
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center;">
            <h3 style="color: #e0e7ff;">📖 About ASL Digits</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-text">
            American Sign Language (ASL) uses specific hand gestures to represent digits 0-9. 
            This AI model recognizes these gestures from images with high accuracy.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center;">
            <h3 style="color: #e0e7ff;">🏗️ Model Architecture</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-text">
            • <strong>Input:</strong> 64×64 grayscale<br>
            • <strong>Conv Layers:</strong> 32→64→128 filters<br>
            • <strong>Dense:</strong> 128 neurons<br>
            • <strong>Output:</strong> 10 classes
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content - Two Column Layout
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">✋ Input Method</div>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio(
            "Choose how to provide the ASL digit image:",
            ["📤 Upload Image", "✏️ Draw on Canvas", "🎲 Use Sample Image"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        img_array = None
        display_image = None
        
        if input_method == "📤 Upload Image":
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">📁 Upload Your Image</div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                display_image = image
                img_array = preprocess_image(image)
                
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        elif input_method == "✏️ Draw on Canvas":
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">🎨 Draw ASL Digit</div>
                <p style="color: #a5b4fc; font-size: 0.9rem;">Draw the hand gesture on the canvas below</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Canvas for drawing
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=15,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            if canvas_result.image_data is not None:
                # Check if canvas has content
                if np.any(canvas_result.image_data[:, :, :3] != 255):
                    img_array = preprocess_canvas(canvas_result)
        
        else:  # Sample Image
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">📚 Select Sample Image</div>
            </div>
            """, unsafe_allow_html=True)
            
            if samples is not None:
                selected_digit = st.selectbox(
                    "Choose a digit:",
                    options=list(range(10)),
                    format_func=lambda x: f"Digit {x}"
                )
                
                if selected_digit in samples:
                    sample_idx = st.slider(
                        "Select sample variation:",
                        0, min(4, len(samples[selected_digit]) - 1), 0
                    )
                    
                    sample_img = samples[selected_digit][sample_idx]
                    display_image = Image.fromarray((sample_img.squeeze() * 255).astype('uint8'), mode='L')
                    img_array = sample_img.reshape(1, 64, 64, 1)
                    
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_b:
                        st.image(display_image, caption=f"Sample Digit {selected_digit}", width=200)
            else:
                st.warning("Sample images not available. Please upload an image or draw one.")
        
        # Predict Button
        if img_array is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Predict Digit", use_container_width=True):
                with st.spinner("🔮 AI is analyzing..."):
                    predicted_class, confidence, all_confidences = predict_digit(model, img_array)
                    st.session_state['prediction'] = {
                        'digit': predicted_class,
                        'confidence': confidence,
                        'all_confidences': all_confidences
                    }
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">🎯 Prediction Results</div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            
            # Main prediction display
            st.markdown(f"""
            <div class="prediction-container">
                <div style="color: #a5b4fc; font-size: 1.1rem; margin-bottom: 10px;">Detected ASL Digit</div>
                <div class="prediction-digit">{pred['digit']}</div>
                <div style="color: #10b981; font-size: 1.5rem; font-weight: 600; margin-top: 15px;">
                    {pred['confidence']:.1f}% Confident
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # All predictions confidence
            st.markdown("""
            <div style="margin-top: 20px;">
                <h4 style="color: #e0e7ff; font-weight: 600;">📊 All Predictions</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for i, conf in enumerate(pred['all_confidences']):
                color = "#10b981" if i == pred['digit'] else "#6366f1"
                st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="digit-label">Digit {i}</span>
                        <span class="confidence-value">{conf:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {conf}%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px; color: #a5b4fc;">
                <div style="font-size: 4rem; margin-bottom: 20px;">🤖</div>
                <div style="font-size: 1.2rem; font-weight: 500;">Waiting for Input</div>
                <div style="font-size: 0.9rem; margin-top: 10px; opacity: 0.7;">
                    Upload an image, draw on canvas, or select a sample to get started
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📈 Model Statistics</div>
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">93%</div>
                <div class="stat-label">Test Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">10</div>
                <div class="stat-label">Digit Classes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">64×64</div>
                <div class="stat-label">Input Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">1.1M</div>
                <div class="stat-label">Parameters</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">3</div>
                <div class="stat-label">Conv Layers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using <strong>TensorFlow</strong> & <strong>Streamlit</strong></p>
        <p style="font-size: 0.85rem; opacity: 0.7;">
            ASL Digits CNN Classifier | Deep Learning Project
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

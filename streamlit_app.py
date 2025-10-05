import streamlit as st
import numpy as np
import json
import pickle
import warnings
import os # Import os for path handling

# --- Configuration ---
ARTIFACTS_DIR = "./artifacts"
COLUMNS_FILE = os.path.join(ARTIFACTS_DIR, 'columns.json')
MODEL_FILE = os.path.join(ARTIFACTS_DIR, 'banglore_home_prices_model.pickle')

# --- Model Loading and Caching (Integrates util.load_saved_artifacts) ---
@st.cache_resource
def load_artifacts_and_model():
    """Load model artifacts and location names once."""
    print("Loading saved artifacts for Streamlit...")
    
    # 1. Initialize variables
    data_columns = None
    locations = None
    model = None
    
    # ADDED: Suppress the specific scikit-learn UserWarning (often related to feature names)
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        # Load column names
        with open(COLUMNS_FILE, 'r') as f:
            data_columns = json.load(f)['data_columns']
            # Locations start from the 4th column (index 3)
            locations = data_columns[3:]
            
        # Load the trained model
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
            
        print("loading saved artifacts...done")
        
        # Return all necessary components
        return data_columns, locations, model
        
    except FileNotFoundError as e:
        error_msg = (
            f"Artifact files not found. Ensure '{COLUMNS_FILE}' and '{MODEL_FILE}' "
            f"exist. Error: {e}"
        )
        st.error(error_msg)
        st.stop()
    except Exception as e:
        error_msg = f"Failed to load artifacts due to an unexpected error. Error: {e}"
        st.error(error_msg)
        st.stop()

# Load all required data (data_columns is often '__data_columns' in util.py)
DATA_COLUMNS, LOCATIONS, MODEL = load_artifacts_and_model()


# --- Prediction Function (Integrates util.get_estimated_price) ---
def get_estimated_price(location, sqft, bhk, bath, data_columns, model):
    """Predicts the price using the loaded model."""
    
    # Ensure all inputs are numeric types
    sqft = float(sqft)
    bhk = int(bhk)
    bath = int(bath)

    try:
        # Find index of the location (convert to lowercase as done during training)
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1 # Handle location not found gracefully

    # Create the feature vector x (np.zeros uses the length of data_columns)
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1 # Set the corresponding location dummy variable to 1

    # Predict the price and round to 2 decimal places
    return round(model.predict([x])[0], 2)


# --- Streamlit App Configuration and Styling ---

st.set_page_config(
    page_title="Bangalore Home Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for a dark, modern look, better alignment, and background image
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    /* Global Styles and Background */
    .stApp {
        background-color: #1e1e2d; 
        color: #f0f0f0; 
        font-family: 'Poppins', sans-serif;
        /* Background image with blur/overlay effect */
        background-image: url('https://source.unsplash.com/random/1920x1080?bangalore,modern-house,city');
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(30, 30, 45, 0.9);
        backdrop-filter: blur(4px);
        z-index: -1;
    }

    /* Form Card Container - Targets Streamlit's main content area */
    .stApp > header, .stApp > div:first-child > div:nth-child(2) {
        max-width: 500px;
        margin: auto;
    }

    .stApp > div:first-child > div:nth-child(2) > div:first-child {
        background-color: rgba(30, 30, 45, 0.95);
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Image Placeholder styling (this section is now less relevant without st.image, but kept for general styling) */
    .stImage { 
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 20px;
    }
    
    /* Titles and Headings */
    h1 {
        text-align: center;
        color: #00bcd4; /* Primary Teal */
        font-weight: 600;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    
    h2 {
        color: #a7f3d0;
        font-size: 1em;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    /* Input/Select/Number Input Styling */
    .stTextInput>div>div>input, .stSelectbox>div>div, .stNumberInput>div>div>input {
        background-color: #2d2d3e !important;
        border-color: #44445c !important;
        color: #f0f0f0 !important;
        border-radius: 6px;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #00bcd4 !important;
        color: #1e1e2d !important;
        padding: 15px 20px !important;
        border-radius: 6px !2important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        margin-top: 25px !important;
        border: none !important;
    }

    /* Radio Group (BHK/Bath) Styling */
    div[role="radiogroup"] {
        display: flex;
        overflow: hidden;
        border-radius: 6px;
        border: 1px solid #44445c;
    }
    div[role="radiogroup"] label {
        padding: 8px 0;
    }
    
    /* Result Display Styling */
    .result-box {
        background-color: #ffc107;
        color: #1e1e2d;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Application Interface ---

# st.image("home.png", caption="Architectural Home Design and Specifications", use_container_width=True) # REMOVED

st.title("Bangalore Home Price Predictor 🏠")
st.markdown("<p style='text-align: center; color: #f0f0f0;'>Estimate the value of your property in Indian Lakhs.</p>", unsafe_allow_html=True)


# Area (Square Feet)
st.markdown("<h2>Area (Square Feet)</h2>", unsafe_allow_html=True)
sqft = st.number_input(
    label="Square Feet",
    min_value=500,
    max_value=10000,
    value=2000,
    step=100,
    label_visibility="collapsed",
    key="sqft_input"
)

# BHK and Bath Alignment using Columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2>BHK</h2>", unsafe_allow_html=True)
    bhk_options = [1, 2, 3, 4, 5]
    bhk = st.radio(
        label="BHK",
        options=bhk_options,
        index=1, # Default to 2 BHK
        horizontal=True,
        label_visibility="collapsed",
        key="bhk_radio"
    )

with col2:
    st.markdown("<h2>Bathrooms</h2>", unsafe_allow_html=True)
    bath_options = [1, 2, 3, 4, 5]
    bathrooms = st.radio(
        label="Bathrooms",
        options=bath_options,
        index=1, # Default to 2 Bath
        horizontal=True,
        label_visibility="collapsed",
        key="bath_radio"
    )


# Location Dropdown
st.markdown("<h2>Location</h2>", unsafe_allow_html=True)
location = st.selectbox(
    label="Location",
    options=LOCATIONS if LOCATIONS else ["Error Loading Locations"],
    index=0 if LOCATIONS else 0,
    label_visibility="collapsed",
    key="location_select"
)

# --- Prediction Logic and Button ---

if st.button("Get Estimate", key="submit_button"):
    try:
        # 1. Input Validation
        if sqft is None or sqft <= 0:
            st.error("Please enter a valid area in square feet.")
            st.stop()
            
        # 2. Call the integrated prediction function
        # Pass the cached model components
        estimated_price = get_estimated_price(location, sqft, bhk, bathrooms, DATA_COLUMNS, MODEL)
        
        # 3. Format the result using Python's f-string
        formatted_price = f"{estimated_price:,.2f}"

        # 4. Display the result
        st.markdown(
            f"""
            <div class="result-box">
                <h3>₹ {formatted_price} Lakh</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Prediction failed. An internal error occurred. Error: {e}")

else:
    # Initial instruction message
    st.markdown(
        """
        <div class="result-box">
            <h3>Enter Details Above</h3>
            <p>Your predicted price will appear here</p>
        </div>
        """,
        unsafe_allow_html=True
    )
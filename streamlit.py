import os
import json
import base64
import pandas as pd
import requests
import streamlit as st
from src.preprocess import Preprocessor  # Assuming this preprocesses data for Flask API

# Configuration Paths
DATA_PATH = os.getenv("DATA_PATH", default="dataset/nigeria_houses_data.csv")
COLUMN_CONFIG = os.getenv("COLUMN_CONFIG", default="config/columns.json")
TOWN_CONFIG = os.getenv("TOWN_CONFIG", default="config/town_freq_enc.json")
STATE_CONFIG = os.getenv("STATE_CONFIG", default="config/state_target_enc.json")
TITLE_CONFIG = os.getenv("TITLE_CONFIG", default="config/title.json")

# Background image path
BG_IMAGE_PATH = "image/housing.png"

@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file, opacity=0.5):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg(BG_IMAGE_PATH, opacity=0.3)

st.title("Nigeria Housing Price Prediction")

# Load and cache the data and configurations
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_config(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

data = load_data()
columns = load_config(COLUMN_CONFIG)
town = load_config(TOWN_CONFIG)
state = load_config(STATE_CONFIG)
title = load_config(TITLE_CONFIG)

# Sidebar for input
with st.sidebar:
    st.write("### Property Configuration")
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    toilets = st.number_input("Toilets", min_value=1, max_value=10, value=2)
    parking_space = st.number_input("Parking Space", min_value=1, max_value=10, value=1)
    town_name = st.selectbox("Town", list(town.keys()))
    property_type = st.selectbox("Property Type", list(title.keys()))
    state_name = st.selectbox("State", list(state.keys()))

    payload = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "toilets": toilets,
        "parking_space": parking_space,
        "town": town_name,
        "property_type": property_type,
        "state": state_name,
    }

    # Convert to DataFrame to show config
    df = pd.DataFrame([payload])

st.write("### Selected Property Configuration")
st.write(df)

# Convert the JSON payload and send the prediction request
st.write("### JSON Payload")
st.json(payload, expanded=False)

if st.button("Predict"):
    with st.spinner("Making Prediction..."):
        try:
            # Preprocess the input data using the Preprocessor from preprocess.py
            preprocessor = Preprocessor()
            processed_data = preprocessor.preprocess(pd.DataFrame([payload]))  # Preprocessing before sending

            # Send the preprocessed data to the Flask API for prediction
            response = requests.post("http://127.0.0.1:5000/predict", json=processed_data.to_dict(orient="records"))

            if response.status_code == 200:
                prediction = response.json().get("prediction")  # Adjust based on API response structure
                st.write(f"### Predicted Price: ₦{prediction[0]:,.2f}")
                st.success("Prediction made successfully!")
                st.balloons()
            else:
                st.error("Failed to make prediction")
                st.write(response.content)
        except Exception as e:
            st.error("Error occurred while connecting to the API")
            st.write(e)

# Footer function
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):
    style = """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 50px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        display="flex",
        justify_content="space-between",
        align_items="flex-end",
        padding=px(10, 20),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )
    style_hr = styles(
        display="block",
        margin=px(0, "auto"),
        border_style="none",
        border_width=px(0.5),
        color='rgba(0,0,0,.5)'
    )

    body = div(style=styles(display="flex", justify_content="space-between", width=percent(100)))()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)
    for arg in args:
        if isinstance(arg, str):
            pass
        elif isinstance(arg, HtmlElement):
            body(arg)
    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        div()(
            "Designed by ",
            link("https://linkedin.com/in/john-olalemi/", "John Olalemi", color="#4682B4"),
        )
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()
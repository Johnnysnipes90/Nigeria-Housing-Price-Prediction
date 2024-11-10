import os
import json
import base64
import pandas as pd
import requests
import streamlit as st
from src.preprocess import Preprocessor

DATA_PATH = os.getenv("DATA_PATH", default="dataset/nigeria_houses_data.csv")
COLUMN_CONFIG = os.getenv("COLUMN_CONFIG", default="config/columns.json")
TOWN_CONFIG = os.getenv("TOWN_CONFIG", default="config/town.json")
PROPERTY_TYPE_CONFIG = os.getenv("PROPERTY_TYPE_CONFIG", default="config/property_type.json")
STATE_CONFIG = os.getenv("STATE_CONFIG", default="config/state.json")

BG_IMAGE_PATH = "image/housing.png"  # Set the background image for the app

@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file, opacity=0.5):
    # Encode the binary file to base64
    bin_str = get_base64_of_bin_file(png_file)
    # CSS to set the background image with reduced contrast using an overlay
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

# Load and cache the data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)
data = load_data()

# Load the column configuration
@st.cache_data
def load_columns():
    with open(COLUMN_CONFIG, "r") as f:
        return json.load(f)
columns = load_columns()

# Load the town configuration
@st.cache_data
def load_town():
    with open(TOWN_CONFIG, "r") as f:
        return json.load(f)
town = load_town()

# Load the property type configuration
@st.cache_data
def load_property_type():
    with open(PROPERTY_TYPE_CONFIG, "r") as f:
        return json.load(f)
property_type = load_property_type()

# Sidebar to collect inputs from the user
with st.sidebar:
    st.write("### Property Configuration")

    # Receive the number of bedrooms from a dropdown
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

    # Receive the number of bathrooms from a dropdown
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

    # Receive the number of toilets from a dropdown
    toilets = st.number_input("Toilets", min_value=1, max_value=10, value=2)

    # Receive parking space information
    parking_space = st.number_input("Parking Space", min_value=1, max_value=10, value=1)

    # Choose the town from a dropdown
    town_name = st.selectbox("Town", list(town.keys()))

    # Choose the property type from a dropdown
    property_type_name = st.selectbox("Property Type", list(property_type.keys()))


    # Encode the state
    state = st.selectbox("State", list(data['state'].unique()))

    payload = {
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "toilets": [toilets],
        "parking_space": [parking_space],
        "town": [town_name],
        "property_type": [property_type_name],
        "state": [state],
    }

    # Convert the input to a DataFrame
    df = pd.DataFrame(payload)

# Display the selected property configuration
st.write("### Selected Property Configuration")
st.write(df)

# Display the JSON payload
st.write("### JSON Payload")
st.json(payload, expanded=False)

# Send the prediction request when the button is clicked
if st.button("Predict"):
    st.spinner("Making Prediction...")
    # Assuming there's an API or model to make the prediction
    response = requests.post("http://localhost:5000/predict", json=df.to_dict())
    if response.status_code == 200:
        prediction = response.json()[0]  # Assuming the response returns the price prediction
        st.write(f"### Predicted Price: ₦{prediction:,.2f}")
        st.success("Prediction made successfully!")
        st.balloons()
    else:
        st.error("Failed to make prediction")
        st.write(response.content)

#------------------------------------------------------------------------------------------------------------
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import streamlit as st

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
        align_items="flex-end",  # Align content to the bottom
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
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

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
            link("https://linkedin.com/in/joshua-olalemi/", "John Olalemi", color="#4682B4"),
        )
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()


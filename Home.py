import streamlit as st
from openghg.standardise import summary_source_formats
import pandas as pd
from PIL import Image
import os

#@st.cache_data(show_spinner=False)
def overview_sites():
    st.title('Overview sites')
    summary = summary_source_formats()
    st.dataframe(summary)

def main():
    overview_sites()


# Setup Page Configuration
st.set_page_config(page_title="OpenGHG Data Explorer")

# Title
st.title("Welcome to OpenGHG Data Explorer")
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
image_dir = "image"
#  Defining subpage information
subpages = [
    {
        "title": "Observation",
        "file": "1_Observation.py",
        "description": "Explore, analyze, and visualize observational data with advanced options including data smoothing and scale conversion.",
        "image": os.path.join(image_dir, "observation_image.png"),
        "info": "Access real-time and historical observational data, apply data smoothing techniques, and perform calibration scale conversions."
    },
    {
        "title": "Footprint",
        "file": "2_Footprint.py",
        "description": "Analyze regional outputs from LPDM models and visualize footprint data interactively or as animations.",
        "image": os.path.join(image_dir, "footprint_image.gif"),
        "info": "Select specific sites, domains, inlets, and species to examine and compare footprint data using interactive Plotly plots or GIF animations."
    },
    {
        "title": "Flux",
        "file": "3_Flux.py",
        "description": "Explore and visualize species flux and emissions estimates within regions using interactive tools.",
        "image": os.path.join(image_dir, "flux_image.png"),
        "info": "Select specific sources, domains, and species to examine flux data, and visualize it through interactive Plotly plots."
    },
    {
        "title": "ModelScenario",
        "file": "4_ModelScenario.py",
        "description": "Create and analyze model scenarios for comparative studies.",
        "image": os.path.join(image_dir, "model_scenario_image.png"),
        "info": "Link observations, footprints, fluxes, and boundary conditions."
    },
    {
        "title": "Unified Data Explorer",
        "file": "0_Unified_Data_Explorer.py",
        "description": "Link observational data with footprint data for integrated analysis.",
        "image": os.path.join(image_dir, "unified_data_explorer_image.png"),
        "info": "Use this tool to visualize and interact with integrated observational and footprint datasets."
    }
]

# layout 2 column
col1, col2 = st.columns(2)

for i, page in enumerate(subpages):
    with [col1, col2][i % 2]:
        st.subheader(page["title"])
        st.image(page["image"], use_column_width=True)
        st.write(page["description"])
        page_link = f"./pages/{page['file']}"
        st.page_link(label=f"Go to {page['title']}", page=page_link)
        st.info(page["info"])
        st.markdown("---")  

# Add
st.markdown("---")
st.write("© 2024 OpenGHG Data Explorer. All rights reserved.")

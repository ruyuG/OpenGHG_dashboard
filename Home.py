import streamlit as st
from openghg.retrieve import search_surface
from openghg.tutorial import use_tutorial_store
from openghg.tutorial import populate_surface_data
from openghg.standardise import summary_source_formats
from openghg.util import get_domain_info
import pandas as pd
from openghg.plotting import plot_timeseries

#@st.cache_data(show_spinner=False)
def overview_sites():
    st.title('Overview sites')
    summary = summary_source_formats()
    #summary = search_surface()
    st.dataframe(summary)

def main():
    overview_sites()
    #search_time_series()
    #search_spatial_data()


if __name__ == "__main__":
    main()


import streamlit as st
from openghg.retrieve import search_surface
import pandas as pd
from openghg.plotting import plot_timeseries
from pandas import to_datetime 
from datetime import datetime, timedelta
import xarray as xr
from openghg.retrieve import get_footprint, search_footprints, search, search_flux
import plotly.graph_objects as go
import numpy as np
import json
import plotly.express as px


import logging
from typing import List, Optional, Union, Dict, Tuple
import base64
from typing import TYPE_CHECKING
from openghg.util import get_species_info, synonyms, get_datapath, load_internal_json
from openghg.plotting._timeseries import _plot_legend_position, _plot_logo, _plot_remove_gaps, _latex2html
from openghg.util._species import get_species_info
from scipy import stats


def main():
    st.title("OpenGHG Data Explorer")
    # Initialize session state
    initialize_session_state()
    
    # Observation data search
    observation_data = search_observation_data()
    
    # Plot observation data
    if 'observation_results' in st.session_state and st.session_state['observation_results'] is not None:
        display_table()
        plot_observation_data()

        footprint_datasets = search_footprint_data()
        # Plot footprint data if available
        if footprint_datasets is not None and not footprint_datasets.empty:
            plot_footprint_data(footprint_datasets)


def initialize_session_state():
    keys = ['site', 'species', 'inlet', 'network', 'instrument', 'compound_group', 'observation_results', 'observation_fig']
    for key in keys:
        if key not in st.session_state:
            if key in ['observation_results', 'observation_fig']:
                st.session_state[key] = None
            else:
                st.session_state[key] = []

def handle_parameters():
    network = None if st.session_state['network'] == "" else st.session_state['network']
    sites = None if not st.session_state['site'] else st.session_state['site']
    species = None if not st.session_state['species'] else st.session_state['species']
    inlets = None if not st.session_state['inlet'] else st.session_state['inlet']
    instruments = None if not st.session_state['instrument'] else st.session_state['instrument']
    return sites, species, inlets, network, instruments


def search_observation_data():
    with st.container():
        st.title("Observation Data Search")

        # display the full list
        summary = search_surface()
        summary_df = summary.results

        # Initialize session state for user selections
        if 'site' not in st.session_state:
            st.session_state['site'] = []
        if 'species' not in st.session_state:
            st.session_state['species'] = []
        if 'inlet' not in st.session_state:
            st.session_state['inlet'] = []
        if 'network' not in st.session_state:
            st.session_state['network'] = ""
        if 'instrument' not in st.session_state:
            st.session_state['instrument'] = ""
        if 'compound_group' not in st.session_state:
            st.session_state['compound_group'] = ""

        # Compound group selection
        compound_groups = {
        "CFCs": ["cfc11", "cfc12", "cfc13", "cfc112", "cfc113", "cfc114", "cfc115"],
        "HCFCs": ["hcfc22", "hcfc124", "hcfc132b", "hcfc133a", "hcfc141b", "hcfc142b"],
        "HFCs": ["hfc23", "hfc32", "hfc125", "hfc134a", "hfc143a", "hfc152a", "hfc227ea", "hfc236fa", "hfc245fa", "hfc365mfc", "hfc4310mee"],
        "HFOs": ["hfo1234yf", "hfo1234zee", "hcfo1233zde"],
        "Halons": ["halon1211", "halon1301", "halon2402"],
        "PFCs": ["cf4", "c2f6", "c3f8", "c4f8", "c4f10", "c6f14"],
        "Alkanes": ["ch4", "c2h6", "c3h8", "cc3h8"],
        "Aromatics": ["c6h6", "c6h5ch3"],
        "Chlorinated Organics": ["ch3cl", "chcl3", "ccl4", "ch3ccl3", "c2hcl3", "c2cl4", "ch2cl2", "clch2ch2cl"],
        "Brominated Organics": ["ch3br", "chbr3", "ch2br2"],
        "Other Halides": ["ch3i", "sf6", "sf5cf3", "nf3", "so2f2"],
        "Other Organics": ["c2h2", "c2f4", "desflurane"],
        "Inorganic Compounds": ["co2", "co", "n2o", "h2", "cos"]
        } 
        # selecbox       
        #compound_group = st.selectbox("Select Compound Group", [""] + list(compound_groups.keys()), key="compound_group_select")
        #st.session_state['compound_group'] = compound_group

        # Adjust species options based on the selected compound group
        #species_options = summary_df['species'].unique()
        #if compound_group:
        #    species_options = [species for species in compound_groups[compound_group] if species in species_options]

        selected_groups = []
        cols = st.columns(5)  # 5 columns
        for i, (group, species_list) in enumerate(compound_groups.items()):
            with cols[i % 5]:
                if st.checkbox(f"{group}", key=f"checkbox_{group}"):
                    selected_groups.extend(species_list)

        # species
        if selected_groups:
            species_options = list(set(selected_groups).intersection(summary_df['species'].unique()))
        else:
            species_options = list(summary_df['species'].unique())


        # Network selection
        network = st.selectbox("Select network", options=[""] + list(summary_df['network'].unique()), key="network_select")
        st.session_state['network'] = network

        # Filter based on network selection
        filtered_df = summary_df
        if st.session_state['network']:
            filtered_df = filtered_df[filtered_df['network'] == st.session_state['network']]

        # Site selection (allows multiple selection)
        sites = st.multiselect("Select site codes", options=list(filtered_df['site'].unique()), key="site_select")
        st.session_state['site'] = sites

        if st.session_state['site']:
            filtered_df = filtered_df[filtered_df['site'].isin(st.session_state['site'])]
        
        # Species selection (allows multiple selection)
        species = st.multiselect("Select species", options=list(species_options), key="species_select")
        st.session_state['species'] = species

        if st.session_state['species']:
            filtered_df = filtered_df[filtered_df['species'].isin(st.session_state['species'])]

        # Inlet selection (allows multiple selection)
        inlets = st.multiselect("Select inlet heights", options=list(filtered_df['inlet'].unique()), key="inlet_select")
        st.session_state['inlet'] = inlets

        # Instrument selection (allows multiple selection)
        instruments = st.multiselect("Select instrument", options=list(filtered_df['instrument'].unique()), key="instrument_select")
        st.session_state['instrument'] = instruments
        
        if st.button("Search Data"):
            sites, species, inlets, network, instruments = handle_parameters()
            results = search_surface(site=sites, species=species, inlet=inlets, network=network, instrument=instruments)
            if hasattr(results, 'results') and not results.results.empty:
                st.session_state['observation_results'] = results.results
            else:
                st.warning("No results found.")
                st.session_state['observation_results'] = None

def display_table():
    """
    Always display this table,When we trigger an action in Streamlit (like clicking a button), 
    it causes the entire interface to re-run, which means that all non-static content is reset unless state information is explicitly saved.
    """
    with st.container():
        if 'observation_results' in st.session_state and st.session_state['observation_results'] is not None:
            st.dataframe(st.session_state['observation_results'])

def calculate_rolling_average(data, window=365):
    st.write(f"Window size: {window}")
    print("Sample before rolling average:", data.head())
    result = data.rolling(window=window, min_periods=1).mean()
    print("Sample after rolling average:", result.head())
    return result

def add_linear_regression(fig, datasets):
    for dataset in datasets:
        data_xr = dataset.data  
        species_name = dataset.metadata["species"]
        site = dataset.metadata["site"]
        inlet = dataset.metadata["inlet"]
        
        data_df = data_xr.to_dataframe().reset_index()
        x_data = data_df['time']
        y_data = data_df[species_name]
        
        # Convert datetime to numeric values (seconds since start)
        x_numeric = (x_data - x_data.min()).dt.total_seconds().values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_data)
        line = slope * x_numeric + intercept
        
        # Adding regression line to the plot
        fig.add_trace(go.Scatter(
            x=x_data,
            y=line,
            mode='lines',
            name=f'Linear Regression - {species_name} - {site.upper()} ({inlet})',
            line=dict(color='red', dash='dash'),
        ))
        
        # Updating plot title to include regression info
        current_title = fig.layout.title.text if fig.layout.title else "Plot"
        new_title = f"{current_title}<br>R-squared: {r_value**2:.4f}, Slope: {slope:.4e}"
        fig.update_layout(title=new_title)

    return fig

def create_rolling_average_plot(updated_datasets):
    fig = go.Figure()
    species_info = get_species_info()
    attributes_data = load_internal_json("attributes.json")
    
    species_strings = []
    unit_strings = []
    for dataset in updated_datasets:
        metadata = dataset.metadata
        species_name = metadata["species"]
        site = metadata["site"]
        inlet = metadata["inlet"]
        
        species_string = _latex2html(species_info[synonyms(species_name, lower=False)]["print_string"])
        legend_text = f"Rolling Avg - {species_string} - {site.upper()} ({inlet})"

        data_xr = dataset.data
        data_df = data_xr.to_dataframe().reset_index()
        
        x_data = data_df['time']
        y_data = data_df[species_name] #df
        #data_units = y_data.attrs.get("units", "1")
        data_units = data_xr[species_name].attrs.get("units", "1")
        unit_value = data_units
        unit_conversion = 1
        
        y_data *= unit_conversion
        rolling_data = calculate_rolling_average(y_data)

        unit_string = attributes_data["unit_print"][unit_value]
        unit_string_html = _latex2html(unit_string)

        x_data_plot, y_data_plot = _plot_remove_gaps(x_data.values, rolling_data.values)

        fig.add_trace(go.Scatter(
            name=legend_text,
            x=x_data_plot,
            y=y_data_plot,
            mode="lines",
            #line=dict(color='red', width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br> %{y:.1f} " + unit_string_html,
        ))

        unit_strings.append(unit_string_html)
        species_strings.append(species_string)

    # Determine whether data is ascending or descending
    y_data_diff = y_data.diff().mean()
    ascending = y_data_diff >= 0

    # Update y-axis title
    ytitle = ", ".join(set(species_strings)) + " (" + unit_strings[0] + ")"
    fig.update_yaxes(title=ytitle)

    # Update x-axis title
    fig.update_xaxes(title="Date")

    # Position the legend
    legend_pos, logo_pos = _plot_legend_position(ascending)
    fig.update_layout(
        legend=legend_pos, 
        template="seaborn",
        title={
            "text": "Rolling Average of Observation Data",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        font={"size": 14},
        margin={"l": 20, "r": 20, "t": 20, "b": 20}
    )

    # Add OpenGHG logo
    logo_dict = _plot_logo(logo_pos)
    fig.add_layout_image(logo_dict)

    return fig
def plot_observation_data():
    with st.container():
        show_rolling_average = st.checkbox("Show Rolling Average", value=False)
        show_linear_regression = st.checkbox("Show Linear Regression", value=False)
        if 'observation_results' in st.session_state and st.session_state['observation_results'] is not None:
            min_dates = pd.to_datetime(st.session_state['observation_results']['start_date'])
            max_dates = pd.to_datetime(st.session_state['observation_results']['end_date'])

            if len(st.session_state['observation_results']) > 1:
                date_range_option = st.radio("Select date range option", ["Intersection", "Union"], key="date_range_option")
                if date_range_option == "Intersection":
                    selected_min_date = max(min_dates)
                    selected_max_date = min(max_dates)
                else:
                    selected_min_date = min(min_dates)
                    selected_max_date = max(max_dates)
            else:
                selected_min_date = min_dates.iloc[0]
                selected_max_date = max_dates.iloc[0]

            default_start = selected_min_date.to_pydatetime() if selected_min_date else pd.Timestamp.now() - pd.Timedelta(days=365)
            default_end = selected_max_date.to_pydatetime() if selected_max_date else pd.Timestamp.now()

            date_range = st.slider("Select date range for all datasets", min_value=default_start, max_value=default_end, value=(default_start, default_end), key="date_range_all")
            st.session_state['observation_time_range'] = date_range
            # create an empty container 
            plot_container = st.empty()
            if st.button("Plot Observation Data", key="plot_observation_data"):
                sites, species, inlets, network, instruments = handle_parameters()
                results = search_surface(site=sites, species=species, inlet=inlets, network=network, instrument=instruments,
                                         start_date=date_range[0].strftime('%Y-%m-%d'), end_date=date_range[1].strftime('%Y-%m-%d'))
                if not hasattr(results, 'results') or results.results.empty:
                    st.warning("No results found or empty results.")
                    return

                updated_datasets = results.retrieve_all()
                if not isinstance(updated_datasets, list):
                        updated_datasets = [updated_datasets]                

                if show_rolling_average:
                    fig = create_rolling_average_plot(updated_datasets)
                else:
                    fig = plot_timeseries(updated_datasets, xvar='time')
                if show_linear_regression and isinstance(fig, go.Figure):
                    fig = add_linear_regression(fig, updated_datasets)
                if isinstance(fig, go.Figure):
                    st.session_state['observation_fig'] = fig
                    plot_container.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Unable to create a valid plot. Please check your data.")
            elif 'observation_fig' in st.session_state and isinstance(st.session_state['observation_fig'], go.Figure):
                plot_container.plotly_chart(st.session_state['observation_fig'], use_container_width=True)


def search_footprint_data():
    #st.header("Footprint Data")
    
    if st.session_state['observation_results'] is None:
        st.warning("Please search for observation data first.")
        return None
    
    footprint_data_results = search_footprints().results
    species_data = get_species_info()  # species info
    
    # species
    observed_species = st.session_state['species']
    search_species = []
    for species in observed_species:
        if species == "co2":
            search_species.append(species)  #  CO2
        elif "lifetime" in species_data.get(species, {}) or "lifetime_monthly" in species_data.get(species, {}):
            search_species.append(species)  # # Non-inert gases with lifetime data
        else:
            search_species.append("inert")  # For other species，use inert as an alternative
    
    # Avoid repeated searches
    search_species = list(set(search_species))
    
    footprint_data_results = search_footprints().results
    
    # Filter footprint data based on observation parameters
    filtered_data = footprint_data_results[
        (footprint_data_results['site'].isin(st.session_state['site'])) &
        (footprint_data_results['species'].isin(search_species)) &
        (footprint_data_results['inlet'].isin(st.session_state['inlet']) if st.session_state['inlet'] else True)
    ]

    if filtered_data.empty:
        st.warning("No corresponding footprint data found for the selected observation parameters.")
        return None
    st.dataframe(filtered_data)
    return filtered_data
def plot_footprint_data(filtered_data):
    if not filtered_data.empty:
        # intersection
        observed_min_date, observed_max_date = st.session_state['observation_time_range']
        footprint_min_date = pd.to_datetime(filtered_data['start_date']).min()
        footprint_max_date = pd.to_datetime(filtered_data['end_date']).max()
        start_date = max(observed_min_date, footprint_min_date)
        end_date = min(observed_max_date, footprint_max_date)
        print(f"start_date: {start_date}, type: {type(start_date)}")
        print(f"end_date: {end_date}, type: {type(end_date)}")
        #print(f"Sample time point: {time_points[0]}, type: {type(time_points[0])}")
        if start_date > end_date:
            st.warning("No overlapping dates between observation and footprint data.")
            return
        site_fp = filtered_data['site'].iloc[0]  # Adjusted for single value
        domain_fp = filtered_data['domain'].iloc[0]
        inlet_fp = filtered_data.get('inlet').iloc[0]
        height_fp = filtered_data.get('height').iloc[0]
        model_fp = filtered_data.get('model').iloc[0]
        species_fp = filtered_data.get('species').iloc[0]  # Ensure this is a single string
        footprint_data = get_footprint(
                        site=site_fp,
                        domain=domain_fp ,
                        inlet=inlet_fp,
                        height=height_fp,
                        model=model_fp,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        species=species_fp
                    )
        # Extract the list of time points from filtered_data
        time_points = pd.to_datetime(footprint_data.data['time']).sort_values().unique()
        # Ensure time points are within intersection range
        #valid_time_points = [time for time in time_points if start_date <= time <= end_date]
        
        #if not valid_time_points:
        #    st.warning("No valid time points within the selected date range.")
        #    return

        # User selects a specific point in time
        selected_time = st.select_slider(
            "Select a specific time point for footprint data",
            options=time_points,
            format_func=lambda x: x.strftime('%Y-%m-%d %H:%M')
        )
        print(f'selected_time is ==== {selected_time}')
        # get footprint data
        if st.button('Fetch and Plot Footprint Data'):
            
            selected_data = footprint_data.data.sel(time=selected_time, method='nearest')
            plot_selected_footprint(selected_data)


def plot_selected_footprint(footprint_data):
    # Function to plot the selected footprint
    if footprint_data:
        fig = go.Figure(data=go.Heatmap(
            z=np.log10(footprint_data.fp.values + 1e-5),
            x=footprint_data.lon,
            y=footprint_data.lat,
            colorscale='Viridis'
        ))
        fig.update_layout(title=f"Footprint ", xaxis_title='Longitude', yaxis_title='Latitude')
        st.plotly_chart(fig, use_container_width=True)



  
if __name__ == "__main__":
    main()

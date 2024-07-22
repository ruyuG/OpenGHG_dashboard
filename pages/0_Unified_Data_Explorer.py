import streamlit as st
from openghg.retrieve import search_surface
from openghg.tutorial import use_tutorial_store
from openghg.standardise import summary_source_formats
from openghg.util import get_domain_info
import pandas as pd
from openghg.plotting import plot_timeseries
from pandas import to_datetime
import xarray as xr
from openghg.retrieve import get_footprint, search_footprints, search, search_flux
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

#@st.cache_data(show_spinner=False)
def overview():
    st.title('Overview')
    #summary = summary_source_formats()
    summary = search_surface()
    st.dataframe(summary.results,height=250)

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

        # Use observation parameters to search footprint data
        footprint_data = search_footprint_data()

        # Plot footprint data if available
        if footprint_data is not None and not footprint_data.empty:
            plot_footprint_data(footprint_data)

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

        # display the full list of species 
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

def plot_observation_data():
    with st.container():
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

            if st.button("Plot Observation Data", key="plot_observation_data"):
                sites, species, inlets, network, instruments = handle_parameters()
                results = search_surface(site=sites, species=species, inlet=inlets, network=network, instrument=instruments,
                                         start_date=date_range[0].strftime('%Y-%m-%d'), end_date=date_range[1].strftime('%Y-%m-%d'))

                if hasattr(results, 'results') and not results.results.empty:
                    updated_datasets = results.retrieve_all()

                    if not isinstance(updated_datasets, list):
                        updated_datasets = [updated_datasets]

                    fig = plot_timeseries(updated_datasets, xvar='time')
                    if fig:
                        st.session_state['observation_fig'] = fig
                        st.plotly_chart(fig)
                    else:
                        st.error("Unable to generate the plot. Please check the selected options.")
                else:
                    st.warning("No results found for the selected date range.")
        # Display stored observation figure if it exists
        if 'observation_fig' in st.session_state and st.session_state['observation_fig'] is not None:
            st.plotly_chart(st.session_state['observation_fig'])

def search_footprint_data():
    st.header("Footprint Data Search")
    
    if st.session_state['observation_results'] is None:
        st.warning("Please search for observation data first.")
        return None
    
    footprint_data_results = search_footprints().results
    
    # Filter footprint data based on observation parameters
    filtered_data = footprint_data_results[
        (footprint_data_results['site'].isin(st.session_state['site'])) &
        (footprint_data_results['species'].isin(st.session_state['species'])) &
        (footprint_data_results['inlet'].isin(st.session_state['inlet']) if st.session_state['inlet'] else True)
    ]
    
    if filtered_data.empty:
        st.warning("No corresponding footprint data found for the selected observation parameters.")
        return None
    st.dataframe(filtered_data)
    return filtered_data

def plot_footprint_data(filtered_data):
    st.header("Footprint Data Plot")
    
    if not filtered_data.empty:
        selected_indices = st.multiselect('Select datasets to plot:', options=filtered_data.index, key="selected_indices_fp")
        
        if selected_indices:
            selected_data = filtered_data.loc[selected_indices]
            start_date = pd.to_datetime(selected_data['start_date']).max().date()
            end_date = pd.to_datetime(selected_data['end_date']).min().date()
            
            selected_dates = st.date_input(
                "Select date range:",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date
            )

            if st.button('Plot Footprints'):
                footprint_data = {}
                for idx in selected_indices:
                    site_data = filtered_data.loc[idx]
                    footprint_data[f"{site_data['site']}_{site_data.get('species')}_{site_data.get('height')}_{idx}"] = get_footprint(
                        site=site_data['site'],
                        domain=site_data['domain'],
                        inlet=site_data.get('inlet'),
                        height=site_data.get('height'),
                        model=site_data.get('model'),
                        start_date=selected_dates[0].strftime('%Y-%m-%d'),
                        end_date=selected_dates[1].strftime('%Y-%m-%d'),
                        species=site_data.get('species')
                    )
                
                plot_footprint_interactive(footprint_data)

def plot_footprint_interactive(footprint_data):
    num_datasets = len(footprint_data)
    fig = make_subplots(rows=num_datasets, cols=1,
                        subplot_titles=list(footprint_data.keys()),
                        shared_xaxes=True, vertical_spacing=0.1)

    # Get the total number of time points for progress bar
    time_values = next(iter(footprint_data.values())).data.time.values
    total_steps = num_datasets * len(time_values)
    current_step = 0
    progress_bar = st.progress(0)

    for i, (site, data_obj) in enumerate(footprint_data.items()):
        dataset = data_obj.data
        lat = dataset.lat.values
        lon = dataset.lon.values

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        lat_bounds = [np.min(lat), np.max(lat)]
        lon_bounds = [np.min(lon), np.max(lon)]

        for t, time_value in enumerate(time_values):
            footprint = dataset.fp.isel(time=t).values
            z = np.log10(footprint + 1e-5)
            zmin = np.min(z[np.isfinite(z)])
            zmax = np.max(z)

            trace = go.Heatmap(
                x=lon_grid[0],
                y=lat_grid[:, 0],
                z=z,
                colorscale='Viridis',
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title="Log(Footprint)", x=1.02,len=0.9,thickness=20,yanchor="top", y=1,ticks="outside"),
                showscale=(t == 0),
                visible=(t == 0)  # only for the first time point
            )
            fig.add_trace(trace, row=i + 1, col=1)

            current_step += 1
            progress_percentage = int(100 * current_step / total_steps)
            progress_bar.progress(progress_percentage)

        fig.update_xaxes(range=lon_bounds, row=i+1, col=1)
        fig.update_yaxes(range=lat_bounds, row=i+1, col=1)

    # Create steps for the slider
    steps = []
    for t, time_value in enumerate(time_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Time: {time_value}"}],
            label=str(time_value)
        )
        for i in range(num_datasets):
            step["args"][0]["visible"][i * len(time_values) + t] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time: "},
        steps=steps,
        pad={"t": 50}
    )]
    
    fig.update_layout(
        sliders=sliders,
        height=400*num_datasets,
        title_text="Footprint Comparison",
        margin=dict(r=80, t=100, b=50), # for colorbar
    )

    st.plotly_chart(fig, use_container_width=True)
       
if __name__ == "__main__":
    main()


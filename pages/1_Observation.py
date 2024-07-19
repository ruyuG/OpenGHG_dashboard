import streamlit as st
from openghg.retrieve import search_surface
from openghg.tutorial import use_tutorial_store
from openghg.standardise import summary_source_formats
from openghg.util import get_domain_info
import pandas as pd
from openghg.plotting import plot_timeseries
from pandas import to_datetime
import xarray as xr
#@st.cache_data(show_spinner=False)
def overview():
    st.title('Overview')
    #summary = summary_source_formats()
    summary = search_surface()
    st.dataframe(summary.results,height=250)

def main():
    overview()
    search_time_series_slider_mul()
    display_table()
    plot_data()

def handle_parameters():
    network = None if st.session_state['network'] == "" else st.session_state['network']
    sites = None if not st.session_state['site'] else st.session_state['site']
    species = None if not st.session_state['species'] else st.session_state['species']
    inlets = None if not st.session_state['inlet'] else st.session_state['inlet']
    instruments = None if not st.session_state['instrument'] else st.session_state['instrument']
    return sites, species, inlets, network, instruments


class ObsData:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

def wrap_dataset_with_metadata(dataset, metadata):
    return ObsData(data=dataset, metadata=metadata)


def prepare_and_plot_data(datasets, date_range):
    obs_data_list = []
    for dataset in datasets:
        metadata = {
            "species": dataset.attrs.get('species', 'Unknown'),
            "site": dataset.attrs.get('site', 'Unknown'),
            "inlet": dataset.attrs.get('inlet', 'Unknown')
        }
        filtered_dataset = dataset.sel(time=slice(*date_range))
        obs_data = wrap_dataset_with_metadata(filtered_dataset, metadata)
        obs_data_list.append(obs_data)
    fig = plot_timeseries(obs_data_list, xvar='time') 
    return fig




def search_time_series():
    st.title("OpenGHG Observation Data")

    # Retrieve the full list of species 
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
    species = st.multiselect("Select species", options=list(filtered_df['species'].unique()), key="species_select")
    st.session_state['species'] = species

    if st.session_state['species']:
        filtered_df = filtered_df[filtered_df['species'].isin(st.session_state['species'])]

    # Inlet selection (allows multiple selection)
    inlets = st.multiselect("Select inlet heights", options=list(filtered_df['inlet'].unique()), key="inlet_select")
    st.session_state['inlet'] = inlets

    # Add instruments here

    # Handle these parameters
    network = None if st.session_state['network'] == "" else st.session_state['network']
    sites = None if not st.session_state['site'] else st.session_state['site']
    species = None if not st.session_state['species'] else st.session_state['species']
    inlets = None if not st.session_state['i    '] else st.session_state['inlet']

    if st.button("Search Data"):
        results = search_surface(site=sites, species=species, inlet=inlets, network=network)
        if hasattr(results, 'results') and not results.results.empty:
            st.session_state['data'] = results.retrieve_all()  
            st.dataframe(results.results)
        else:
            st.warning("No results found.")
            st.session_state['data'] = None

    if st.button("Retrieve and Plot Data"):
        if 'results' in st.session_state and not st.session_state['results'].results.empty:
            results = st.session_state['results']
            # Retrieve data
            data = results.retrieve_all()  
            print(data)
            # Plot the data
            if data:
                plot_species = ", ".join(st.session_state['species'])  # Adjust for multiple species
                plot_sites = ", ".join(st.session_state['site'])  # Adjust for multiple sites
                st.write(f"Plotting data for {plot_species} at {plot_sites}, inlets {'all' if not inlets else ', '.join(inlets)}")
                fig = plot_timeseries(data)
                st.plotly_chart(fig)
            else:
                st.warning("No data available for plotting.")
        else:
            st.error("No search has been performed or no results are available.")


def search_time_series_slider_single():
    st.title("OpenGHG Observation Data")

    # Retrieve the full list of species 
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
    species = st.multiselect("Select species", options=list(filtered_df['species'].unique()), key="species_select")
    st.session_state['species'] = species

    if st.session_state['species']:
        filtered_df = filtered_df[filtered_df['species'].isin(st.session_state['species'])]

    # Inlet selection (allows multiple selection)
    inlets = st.multiselect("Select inlet heights", options=list(filtered_df['inlet'].unique()), key="inlet_select")
    st.session_state['inlet'] = inlets

    # Handle these parameters
    network = None if st.session_state['network'] == "" else st.session_state['network']
    sites = None if not st.session_state['site'] else st.session_state['site']
    species = None if not st.session_state['species'] else st.session_state['species']
    inlets = None if not st.session_state['inlet'] else st.session_state['inlet']

    if st.button("Search Data"):
        results = search_surface(site=sites, species=species, inlet=inlets, network=network)
        if hasattr(results, 'results') and not results.results.empty:
            st.session_state['data'] = results.retrieve_all()
            st.dataframe(results.results)
        else:
            st.warning("No results found.")
            st.session_state['data'] = None

    if 'data' in st.session_state and st.session_state['data'] is not None:
        dataset = st.session_state['data'].data
        min_date = to_datetime(dataset.time.min().values).to_pydatetime()
        max_date = to_datetime(dataset.time.max().values).to_pydatetime()
        date_range = st.slider("Select date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

        if st.button("Plot Data"):
            fig = prepare_and_plot_data(dataset, date_range)
            if fig:
                st.plotly_chart(fig)
            else:
                st.error("Unable to generate the plot. Please check the selected options.")



def search_time_series_slider_mul():
    with st.container():
        st.title("OpenGHG Observation Data")

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

        # Handle these parameters
        sites, species, inlets, network, instruments = handle_parameters()

        if st.button("Search Data"):
            results = search_surface(site=sites, species=species, inlet=inlets, network=network, instrument=instruments)
            if hasattr(results, 'results') and not results.results.empty:
                #if isinstance(results.retrieve_all(), list):
                #    st.session_state['data'] = results.retrieve_all()
                # If it's not a list, there's only one dataset, put it in the list
                #else:
                #    st.session_state['data'] = [results.retrieve_all()]

                # save the results for display the table
                st.session_state['search_results'] = results.results
                #st.dataframe(results.results)
            else:
                st.warning("No results found.")
                st.session_state['data'] = None
def display_table():
    """
    Always display this table,When we trigger an action in Streamlit (like clicking a button), 
    it causes the entire interface to re-run, which means that all non-static content is reset unless state information is explicitly saved.
    """
    with st.container():
        if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
            st.dataframe(st.session_state['search_results'])

def plot_data_bak():
    with st.container():
        if 'data' in st.session_state and st.session_state['data'] is not None:

            datasets = [d.data for d in st.session_state['data']]  # allows a list of ObsData
            if datasets:
                # There are two cases of multiple datasets, one is different sites or species, to take the intersection. 
                # The second is the existence of the same site,species,inlet, but with a time break(maybe measured by a different instrument)
                # consider how to identify and take the union set.
                min_date = min([to_datetime(ds.time.min().values).to_pydatetime() for ds in datasets])
                max_date = max([to_datetime(ds.time.max().values).to_pydatetime() for ds in datasets])
                date_range = st.slider("Select date range for all datasets", min_value=min_date, max_value=max_date, value=(min_date, max_date), key="date_range_all")

                if st.button("Plot All Data", key="plot_all_data"):
                    if 'search_results' in st.session_state:
                        st.dataframe(st.session_state['search_results'])
                    fig = prepare_and_plot_data(datasets, date_range)
                    if fig:
                        st.plotly_chart(fig)
                    else:
                        st.error("Unable to generate the plot. Please check the selected options.")

def plot_data():
    with st.container():
        if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
            min_dates = pd.to_datetime(st.session_state['search_results']['start_date'])
            max_dates = pd.to_datetime(st.session_state['search_results']['end_date'])

            if len(st.session_state['search_results']) > 1:
                # select date range
                date_range_option = st.radio("Select date range option", ["Intersection", "Union"], key="date_range_option")
                if date_range_option == "Intersection":
                    selected_min_date = max(min_dates)
                    selected_max_date = min(max_dates)
                else:  # Union
                    selected_min_date = min(min_dates)
                    selected_max_date = max(max_dates)
            else:
                # One dataset
                selected_min_date = min_dates.iloc[0]
                selected_max_date = max_dates.iloc[0]


            # 
            default_start = selected_min_date.to_pydatetime() if selected_min_date else datetime.datetime.now() - datetime.timedelta(days=365)
            default_end = selected_max_date.to_pydatetime() if selected_max_date else datetime.datetime.now()

            date_range = st.slider("Select date range for all datasets", min_value=default_start, max_value=default_end, value=(default_start, default_end), key="date_range_all")

            if st.button("Plot All Data", key="plot_all_data"):
                # Retrieve data based on updated date range
                sites, species, inlets, network, instruments = handle_parameters()
                results = search_surface(site=sites, species=species, inlet=inlets, network=network, instrument=instruments,
                                         start_date=date_range[0].strftime('%Y-%m-%d'), end_date=date_range[1].strftime('%Y-%m-%d'))
                
                if hasattr(results, 'results') and not results.results.empty:
                    updated_datasets = results.retrieve_all()

                    # Depending on whether the datasets are a list or a single dataset
                    if not isinstance(updated_datasets, list):
                        updated_datasets = [updated_datasets]

                    # plot data
                    fig = plot_timeseries(updated_datasets, xvar='time')
                    if fig:
                        st.plotly_chart(fig)
                    else:
                        st.error("Unable to generate the plot. Please check the selected options.")
                else:
                    st.warning("No results found for the selected date range.")


if __name__ == "__main__":
    main()


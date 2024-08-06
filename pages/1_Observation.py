import streamlit as st
from openghg.retrieve import search_surface
from openghg.tutorial import use_tutorial_store
from openghg.standardise import summary_source_formats
from openghg.util import get_domain_info
import pandas as pd
from openghg.plotting import plot_timeseries
from pandas import to_datetime
import xarray as xr
import plotly.graph_objects as go
# rolling average plot 
from openghg.util import get_species_info, synonyms, get_datapath, load_internal_json
from openghg.plotting._timeseries import _plot_legend_position, _plot_logo, _plot_remove_gaps, _latex2html
from openghg.util._species import get_species_info
from scipy import stats
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
def calculate_rolling_average(data, window=300):
    #st.write(f"Window size: {window}")
    result = data.rolling(window=window, min_periods=1).mean()
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
def plot_data():
    with st.container():
        show_rolling_average = st.checkbox("Show Rolling Average", value=False)
        show_linear_regression = st.checkbox("Show Linear Regression", value=False)
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
            # create an empty container, for display plot
            plot_container = st.empty()
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


if __name__ == "__main__":
    main()


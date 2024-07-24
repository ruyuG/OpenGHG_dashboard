import streamlit as st
from openghg.analyse import ModelScenario
from openghg.retrieve import get_obs_surface, get_footprint, get_flux, get_bc, search_bc, search_flux, search_footprints, search_surface
import matplotlib.pyplot as plt
from functools import reduce

def load_data(site, species, height, domain, source, start_date, end_date, bc_input):
    obs = get_obs_surface(site=site, species=species, inlet=height, start_date=start_date, end_date=end_date)
    footprint = get_footprint(site=site, species='inert', domain=domain, height=height, start_date=start_date, end_date=end_date)
    flux = get_flux(species=species, domain=domain, source=source, start_date=start_date, end_date=end_date)
    bc = get_bc(species=species, domain=domain, bc_input=bc_input, start_date=start_date, end_date=end_date)
    return obs, footprint, flux, bc


def search_flux_data():
    flux_data_results = search_flux().results
    if flux_data_results.empty:
        st.warning("No flux data available.")
        return None, None
    unique_sources = flux_data_results['source'].unique()
    unique_domains = flux_data_results['domain'].unique()
    return unique_sources, unique_domains


def search_bc_data():
    bc_data_results = search_bc().results
    if bc_data_results.empty:
        st.warning("No bc data available.")
        return None
    unique_bc_input = bc_data_results['bc_input'].unique()
    return unique_bc_input


def search_footprint_data():
    footprint_data_results = search_footprints().results
    if footprint_data_results.empty:
        st.warning("No footprint data available.")
        return None, None
    unique_sites = footprint_data_results['site'].unique()
    unique_domains = footprint_data_results['domain'].unique()
    unique_inlets = footprint_data_results['inlet'].unique()
    return unique_sites, unique_domains, unique_inlets


def search_obs_data():
    surface_data_results = search_surface().results
    if surface_data_results.empty:
        st.warning("No surface data available.")
        return None, None
    unique_sites = surface_data_results['site'].unique()
    unique_species = surface_data_results['species'].unique()
    return unique_sites, unique_species

def get_common():
    '''
    get common sites(obs, footprint) and domain(fp, flux)
    '''
    surface_sites, _ = search_obs_data()
    footprint_sites, footprint_domains, _ = search_footprint_data()
    _, flux_domains = search_flux_data()
        
    common_sites = list(set(footprint_sites) & set(surface_sites))

    common_domains = list(set(footprint_domains) & set(flux_domains))
    return common_sites,common_domains

def main():
    st.title('Create a Model Scenario')

    # Get data for dropdowns
    unique_sources, _ = search_flux_data()
    unique_bc_inputs = search_bc_data()
    footprint_sites, _, unique_inlets= search_footprint_data()
    surface_sites, unique_species = search_obs_data()

    # Get common sites
    common_sites, common_domains = get_common()



    # Initialize session state for dropdowns 
    if 'site' not in st.session_state:
        st.session_state.site = common_sites[0] if len(common_sites) > 0 else None
    if 'species' not in st.session_state:
        st.session_state.species = unique_species[0] if len(unique_species) > 0 else None
    if 'domain' not in st.session_state:
        st.session_state.domain = common_domains[0] if len(common_domains) > 0 else None
    if 'height' not in st.session_state:
        st.session_state.height = unique_inlets[0] if len(unique_inlets) > 0 else None
    if 'source' not in st.session_state:
        st.session_state.source = unique_sources[0] if len(unique_sources) > 0 else None
    if 'bc_input' not in st.session_state:
        st.session_state.bc_input = unique_bc_inputs[0] if len(unique_bc_inputs) > 0 else None

    # Dropdowns with state 
    site = st.selectbox('Select Site', common_sites, key='site')
    species = st.selectbox('Select Species', unique_species, key='species')
    domain = st.selectbox('Select Domain', common_domains, key='domain')
    height = st.selectbox('Height', unique_inlets, key='height')
    source = st.selectbox('Source:', options=unique_sources, key='source')
    bc_input = st.selectbox('BC Input:', options=unique_bc_inputs, key='bc_input')
    start_date = st.date_input('Start Date')
    end_date = st.date_input('End Date')

    if st.button('Load Data'):
        obs, footprint, flux, bc = load_data(site, species, height, domain, source, start_date, end_date, bc_input)
        scenario = ModelScenario(obs=obs, footprint=footprint, flux=flux, bc=bc)
        # Store scenario in session state
        st.session_state['scenario'] = scenario  

    if 'scenario' in st.session_state:
        st.write('Model Scenario Created.')
        scenario = st.session_state['scenario']
        modelled_observations = scenario.calc_modelled_obs()
        fig, ax = plt.subplots()
        modelled_observations.plot(ax=ax)
        st.pyplot(fig)

        st.write('Comparing Modelled and Observed Data:')
        comparison_fig = scenario.plot_comparison()
        st.plotly_chart(comparison_fig)

if __name__ == "__main__":
    main()
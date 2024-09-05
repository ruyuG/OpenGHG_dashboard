import streamlit as st
from openghg.util import get_domain_info
import pandas as pd
from openghg.retrieve import get_flux, search_flux
from datetime import datetime
import matplotlib.pyplot as plt 
from PIL import Image
import io
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    display_domain_info()
    with st.container():
        filtered_data = search_flux_data()
    with st.container():
        if filtered_data is not None and not filtered_data.empty:
            plot_flux_data(filtered_data)


def display_domain_info():
    st.title('Overview Domain')
    domain_info = get_domain_info()
    df = pd.DataFrame(domain_info)
    st.dataframe(df.transpose())



def search_flux_data():
    st.title('Flux')
    flux_data_results = search_flux().results

    if not flux_data_results.empty:
        unique_sources = flux_data_results['source'].unique()
        selected_source = st.selectbox('Select a source:', options=[""] + list(unique_sources), key="selected_source_flux")

        unique_domains = flux_data_results['domain'].unique()
        selected_domain = st.selectbox('Select a domain:', options=[""] + list(unique_domains), key="selected_domain_flux")

        unique_species = flux_data_results['species'].unique()
        selected_species = st.selectbox('Select a species:', options=[""] + list(unique_species), key="selected_species_flux")

        # Filter data
        conditions = {}
        if selected_source:
            conditions['source'] = selected_source
        if selected_domain:
            conditions['domain'] = selected_domain
        if selected_species:
            conditions['species'] = selected_species

        if conditions:
            filtered_data = flux_data_results.loc[
                (flux_data_results[list(conditions)] == pd.Series(conditions)).all(axis=1)
            ]
        else:
            filtered_data = flux_data_results

        # Show datasets
        st.dataframe(filtered_data)

        return filtered_data

def plot_flux_data(filtered_data):
    st.title('Flux Data Plot')
    if not filtered_data.empty:
        selected_indices = st.multiselect('Select datasets to plot:', options=filtered_data.index, key="selected_indices_flux")
        
        if selected_indices:
            selected_data = filtered_data.loc[selected_indices]
            
            flux_data = {}
            all_time_values = set()
            for idx in selected_indices:
                source_data = filtered_data.loc[idx]
                flux = get_flux(
                    source=source_data['source'],
                    domain=source_data['domain'],
                    species=source_data['species']
                )
                flux_data[f"{source_data['source']}_{source_data['species']}_{idx}"] = flux
                all_time_values.update(flux.data.time.values)
            
            all_time_values = sorted(all_time_values)
            start_year = pd.to_datetime(all_time_values[0]).year
            end_year = pd.to_datetime(all_time_values[-1]).year
            
            selected_years = st.slider(
                "Select year range:",
                min_value=start_year,
                max_value=end_year,
                value=(start_year, end_year)
            )

            if st.button('Plot Flux'):
                plot_flux_interactive(flux_data, selected_years)

def plot_flux_interactive(flux_data, selected_years):
    num_datasets = len(flux_data)
    fig = make_subplots(rows=num_datasets, cols=1,
                        subplot_titles=list(flux_data.keys()),
                        shared_xaxes=True, vertical_spacing=0.1)

    for i, (source, data_obj) in enumerate(flux_data.items()):
        dataset = data_obj.data
        lat = dataset.lat.values
        lon = dataset.lon.values
        time_values = dataset.time.values

        # Filter time values based on selected years
        time_mask = ((pd.to_datetime(time_values).year >= selected_years[0]) & 
                    (pd.to_datetime(time_values).year <= selected_years[1]))
        time_values = time_values[time_mask]

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        lat_bounds = [np.min(lat), np.max(lat)]
        lon_bounds = [np.min(lon), np.max(lon)]

        for t, time_value in enumerate(time_values):
            flux = dataset.flux.sel(time=time_value).values
            #z = np.log10(flux + 1e-5)
            z = np.log10(flux + 1e-12)
            #z = flux
            zmin = np.min(z[np.isfinite(z)])
            zmax = np.max(z)


            trace = go.Heatmap(
                x=lon_grid[0],
                y=lat_grid[:, 0],
                z=z,
                colorscale='Viridis',
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title="Log(Flux)", x=1.02, len=0.9, thickness=20, yanchor="top", y=1, ticks="outside"),
                #colorbar=dict(title="Flux",x=1.02,len=0.9,thickness=20,yanchor="top",y=1,ticks="outside",tickformat=".2e"),
                showscale=True,
                visible=(t == 0)
            )
            fig.add_trace(trace, row=i + 1, col=1)

        fig.update_xaxes(range=lon_bounds, row=i+1, col=1)
        fig.update_yaxes(range=lat_bounds, row=i+1, col=1)

    steps = []
    for t, time_value in enumerate(time_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Year: {pd.to_datetime(time_value).year}"}],
            label=str(pd.to_datetime(time_value).year)
        )
        for i in range(num_datasets):
            step["args"][0]["visible"][i * len(time_values) + t] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        steps=steps,
        pad={"t": 50}
    )]
    
    fig.update_layout(
        sliders=sliders,
        height=400*num_datasets,
        title_text="Flux Comparison",
        margin=dict(r=80, t=100, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from typing import Optional
from xarray import Dataset

def plot_flux(data: xr.Dataset, ax, label: Optional[str] = None, 
              vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """Plot a flux dataset on an existing axis.

    Args:
        data: xarray Dataset containing the flux variable, typically time series data.
        ax: Matplotlib axis to plot on.
        label: Label for y-axis and colorbar.
        vmin: Minimum value for y-axis.
        vmax: Maximum value for y-axis.

    Returns:
        None
    """
    if 'time' in data.dims and 'flux' in data.data_vars:
        # Extract the time series and flux values
        time = data['time'].values
        flux = data['flux'].values
        
        # Check and apply user-defined y-axis limits
        if vmin is None:
            vmin = np.nanmin(flux)
        if vmax is None:
            vmax = np.nanmax(flux)

        # Plotting the time series
        line, = ax.plot(time, flux, label='Flux over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel(label if label else 'Flux')
        ax.set_ylim(vmin, vmax)

        # Adding legend and grid
        ax.legend()
        ax.grid(True)

        ax.set_title('Flux Time Series')

        return line
    else:
        raise ValueError("Dataset must contain 'time' and 'flux' dimensions/variables.")


if __name__ == "__main__":
    main()

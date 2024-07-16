import streamlit as st
from openghg.util import get_domain_info
import pandas as pd
from openghg.retrieve import get_footprint, search_footprints, search, search_flux
#from openghg.plotting import plot_footprint
from datetime import datetime
import matplotlib.pyplot as plt 
from PIL import Image
import io
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    search_spatial_data()
    with st.container():
        filtered_data = search_footprint_data()
    with st.container():
        if filtered_data is not None and not filtered_data.empty:
            plot_data(filtered_data)


def search_spatial_data():
    st.title('Overview Domain')
    domain_info = get_domain_info()
    display_domain_info(domain_info)



def display_domain_info(domain_info):
    data = []
    for region, details in domain_info.items():
        row = {
            'Region': region,
            'Description': details['description'],
            'Latitude Range': ', '.join(map(str, details['latitude_range'])),
            'Longitude Range': ', '.join(map(str, details['longitude_range'])),
            'Latitude Increment': details['latitude_increment'],
            'Longitude Increment': details['longitude_increment'],
            'Latitude File': details['latitude_file'],
            'Longitude File': details['longitude_file']
        }
        data.append(row)

    df = pd.DataFrame(data)
    st.dataframe(df)

def search_footprint_data():
    st.title('Footprint')
    footprint_data_results = search_footprints().results


    if not footprint_data_results.empty:
        unique_sites = footprint_data_results['site'].unique()
        selected_site = st.selectbox('Select a site:',  options=[""] + list(unique_sites),key="selected_site_fp")

        unique_domains = footprint_data_results['domain'].unique()
        selected_domain = st.selectbox('Select a domain:', options=[""] + list(unique_domains),key="selected_domain_fp")

    
        unique_inlets = footprint_data_results['inlet'].dropna().unique()
        selected_inlet = st.selectbox('Select an inlet:', options=[""] + list(unique_inlets),key="selected_inlet_fp")


        unique_species = footprint_data_results['species'].unique()
        selected_species = st.selectbox('Select a species:', options=[""] + list(unique_species),key="selected_species_fp")

        # Fliter data
        conditions = {}
        if selected_site:
            conditions['site'] = selected_site
        if selected_domain:
            conditions['domain'] = selected_domain
        if selected_inlet:
            conditions['inlet'] = selected_inlet
        if selected_species:
            conditions['species'] = selected_species

        if conditions:
            filtered_data = footprint_data_results.loc[
                (footprint_data_results[list(conditions)] == pd.Series(conditions)).all(axis=1)
            ]
        else:
            filtered_data = footprint_data_results

        # Show datasets
        st.dataframe(filtered_data)


        return filtered_data
def plot_data(filtered_data):
    st.title('Footprint Data Plot')
    if not filtered_data.empty:
        selected_indices = st.multiselect('Select datasets to plot:', options=filtered_data.index, key="selected_indices_fp")
        
        if selected_indices:
            selected_data = filtered_data.loc[selected_indices]
            # Need to consider the date, currently showing union, it should be changed to intersection(multiple).
            #start_date = pd.to_datetime(selected_data['start_date']).min().date()
            #end_date = pd.to_datetime(selected_data['end_date']).max().date()

            # Intersection
            start_date = pd.to_datetime(selected_data['start_date']).max().date()
            end_date = pd.to_datetime(selected_data['end_date']).min().date()
            # Should add error message here, report error if there is no intersection
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
                    # Using unique keys by combining the site name with the index
                    # which allows plotting multiple datasets from the same site without data overlap.
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
       




def plot_data_bak(filtered_data):
    st.title('Footprint Data Plot')
    if not filtered_data.empty:
        selected_index = st.selectbox('Select a dataset to plot:', options=filtered_data.index, key="selected_index_fp")
        selected_data = filtered_data.loc[selected_index]
        
        start_date = pd.to_datetime(selected_data['start_date']).date()
        end_date = pd.to_datetime(selected_data['end_date']).date()

        selected_dates = st.select_slider(
            "Select date range:",
            options=pd.date_range(start_date, end_date).date,
            value=(start_date, end_date)  
        )
    if st.button('Plot Footprint'):
        #selected_data = filtered_data.loc[selected_index]
        footprint_data = get_footprint(
            site=selected_data['site'],
            domain=selected_data['domain'],
            inlet=selected_data.get('inlet'),
            height=selected_data.get('height'),
            model=selected_data.get('model'),
            start_date=selected_dates[0].strftime('%Y-%m-%d'),
            end_date=selected_dates[1].strftime('%Y-%m-%d'),
            species=selected_data.get('species')
        )
        anim = plot_footprint_animation(footprint_data.data)
        st.image(anim)
    
def plot_footprint_gif(data):
    # modified original plot_footprint
    frames = []  # Initialize the frames list
    num_time_points = len(data.fp.time)
    progress_bar = st.progress(0)

    # Determine the number of frames for test
    #num_time_points = min(len(data.fp.time), 30)

    if num_time_points == 0:
        st.error("No time points available for animation.")
        return None

    # Create animation frames
    for frame in range(num_time_points):
        fig, ax = plt.subplots()
        
        plot_footprint_pyplot(data=data, ax=ax, time_index=frame, label="Concentration")

        ax.set_title(f"Footprint at time {data.fp.time.values[frame]}")

        # Convert Matplotlib figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        plt.close(fig)

        progress_bar.progress((frame + 1) / num_time_points)  # Update progress
    # Save as GIF
    gif_path = '/tmp/animation.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return gif_path




import matplotlib.colors as colors
import matplotlib.pyplot as plt
from typing import Optional
from xarray import Dataset

def plot_footprint_pyplot(
    data: Dataset, ax, time_index=0, label: Optional[str] = None, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> None:
    """Plot a footprint on an existing axis.

    Args:
        data: Dataset containing fp variable
        ax: Matplotlib axis to plot on
        label: Label for colourbar
        vmin: Minimum value for colours
        vmax: Maximum value for colours
    Returns:
        None
    """
    # Plot footprints as a 2D colour map
    data_fp = data.fp.isel(time=time_index)  # First time point
    lat = data_fp.lat
    lon = data_fp.lon
    footprint = data_fp.values

    # Apply user-defined color limits
    if vmin is None:
        vmin = 1e-5  # min is 0 and can't use 0 for a log scale
    if vmax is None:
        vmax = footprint.max()

    im = ax.pcolormesh(
        lon, lat, footprint, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading="auto"
    )
    cb = plt.colorbar(im, ax=ax)

    if label:
        cb.set_label(label)




if __name__ == "__main__":
    main()


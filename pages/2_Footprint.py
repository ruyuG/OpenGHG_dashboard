import streamlit as st
from openghg.util import get_domain_info
import pandas as pd
from openghg.retrieve import get_footprint, search_footprints, search, search_flux
from datetime import datetime
import matplotlib.pyplot as plt 
from PIL import Image
import io
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as colors
from typing import Optional
from xarray import Dataset
# for country line
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
import windrose
from windrose import WindroseAxes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from windrose import WindroseAxes
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
            start_date = pd.to_datetime(selected_data['start_date']).max().date()
            end_date = pd.to_datetime(selected_data['end_date']).min().date()

            selected_dates = st.date_input(
                "Select date range:",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date
            )
            plot_type = st.radio("Choose plot type:", ("Interactive Plotly", "GIF Animation"))

            if st.button('Generate Plot'):
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
                
                if plot_type == "Interactive Plotly":
                    plot_footprint_interactive(footprint_data)
                elif plot_type == "GIF Animation":
                    plot_footprint_gif(footprint_data)
                # Display Pollution Rose after the main plot
                plot_wind_rose(footprint_data)
def plot_footprint_interactive(footprint_data):
    num_datasets = len(footprint_data)
    fig = make_subplots(rows=num_datasets, cols=1,
                        subplot_titles=list(footprint_data.keys()),
                        shared_xaxes=True, vertical_spacing=0.1)
    # Get world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
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
        # Add heat maps for all points in time
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
                showscale=True,
                visible=(t == 0)  # only for the first time point
            )
            fig.add_trace(trace, row=i + 1, col=1)

            current_step += 1
            progress_percentage = int(100 * current_step / total_steps)
            progress_bar.progress(progress_percentage)
        # Add country line
        for _, country in world.iterrows():
            if isinstance(country['geometry'], Polygon):
                x, y = country['geometry'].exterior.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='black', width=1),
                                         showlegend=False, visible=True), row=i+1, col=1)
            elif isinstance(country['geometry'], MultiPolygon):
                for polygon in country['geometry'].geoms:
                    x, y = polygon.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='black', width=1),
                                             showlegend=False, visible=True), row=i+1, col=1)
        fig.update_xaxes(range=lon_bounds, row=i+1, col=1)
        fig.update_yaxes(range=lat_bounds, row=i+1, col=1)

    # Create steps for the slider
    steps = []
    for t, time_value in enumerate(time_values):
        short_time = pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M')
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Time: {short_time}"}],
            label=short_time
        )
        for i in range(num_datasets):
            heatmap_index = i * (len(time_values) + len(world)) + t
            step["args"][0]["visible"][heatmap_index] = True
            #step["args"][0]["visible"][heatmap_index] = True
            # Set all country lines visible
            for j in range(len(world)):
                country_index = i * (len(time_values) + len(world)) + len(time_values) + j
                step["args"][0]["visible"][country_index] = True
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
       


def plot_footprint_gif(footprint_data):
    num_datasets = len(footprint_data)
    frames = []
    
    # Get the total number of time points for progress bar
    time_values = next(iter(footprint_data.values())).data.time.values
    total_steps = num_datasets * len(time_values)
    current_step = 0
    progress_bar = st.progress(0)

    for t, time_value in enumerate(time_values):
        fig = plt.figure(figsize=(10, 5 * num_datasets))
        short_time = pd.to_datetime(time_value).strftime('%Y-%m-%d %H:%M')
        for i, (site, data_obj) in enumerate(footprint_data.items()):
            #  Make sure that each subplot (ax) is correctly created using Cartopy projection before calling plot_footprint_pyplot, then add_feature method is supported.
            ax = fig.add_subplot(num_datasets, 1, i + 1, projection=ccrs.PlateCarree())
            dataset = data_obj.data
            plot_footprint_pyplot(data=dataset, ax=ax, time_index=t, label=f"{site} Footprint")
            ax.set_title(f"{site} at {short_time}")

            current_step += 1
            progress_percentage = int(100 * current_step / total_steps)
            progress_bar.progress(progress_percentage)
        # Convert Matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        plt.close(fig)

    # Save as GIF
    gif_path = './data/footprint_animation.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
    
    # Display GIF
    st.image(gif_path, caption='Footprint Animation')
    # Provide download link
    with open(gif_path, "rb") as file:
        btn = st.download_button(
            label="Download GIF",
            data=file,
            file_name="footprint_animation.gif",
            mime="image/gif"
        )

def plot_footprint_pyplot(data: Dataset, ax, time_index=0, label: Optional[str] = None, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='grey')  # Add country line

    data_fp = data.fp.isel(time=time_index)
    lat = data_fp.lat
    lon = data_fp.lon
    footprint = data_fp.values

    if vmin is None:
        vmin = 1e-5
    if vmax is None:
        vmax = footprint.max()

    im = ax.pcolormesh(
        lon, lat, footprint, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading="auto",  transform=ccrs.PlateCarree()
    )
    cb = plt.colorbar(im, ax=ax)

    if label:
        cb.set_label(label)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    ax.coastlines('50m', linewidth=0.8, edgecolor='grey')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
def calculate_footprint_metric(fp_data):
    # 方法1：加权平均
    weighted_avg = np.average(fp_data, weights=fp_data)
    
    # 方法2：阈值方法（例如，只考虑top 10%的值）
    threshold = np.percentile(fp_data, 90)
    top_values = fp_data[fp_data >= threshold]
    threshold_avg = np.mean(top_values)
    
    # 方法3：最大值
    max_value = np.max(fp_data)
    
    return weighted_avg, threshold_avg, max_value
def plot_wind_rose(footprint_data):
    for site, data_obj in footprint_data.items():
        dataset = data_obj.data  # 直接使用数据，不再进行时间选择
        species = dataset.species
        # 获取站点位置
        site_lat = dataset.release_lat.values[0]
        site_lon = dataset.release_lon.values[0]
        
        # 获取风向和风速数据
        wind_speed = dataset.wind_speed.values
        wind_direction = dataset.wind_from_direction.values
        
        # 计算footprint指标
        metrics = [calculate_footprint_metric(fp_slice) for fp_slice in dataset.fp.values]
        metrics = np.array(metrics)
        
        #weighted_avg, threshold_avg, max_value = metrics.T

        # 计算平均footprint强度（即气体浓度）
        #concentration = weighted_avg

        # 移除NaN值
        #mask = ~np.isnan(wind_speed) & ~np.isnan(wind_direction) & ~np.isnan(concentration)
        mask = ~np.isnan(wind_speed) & ~np.isnan(wind_direction)
        wind_speed = wind_speed[mask]
        wind_direction = wind_direction[mask]
        #concentration = concentration[mask]

        if len(wind_speed) == 0 or np.all(np.isnan(wind_speed)) or np.all(np.isnan(wind_direction)):
            st.warning(f"No valid data available for site {site} (lat: {site_lat}, lon: {site_lon}).")
            continue

        # 确保风向在0-360度范围内
        wind_direction = np.mod(wind_direction, 360)

        # 如果风速全为零，添加一个小的偏移量
        if np.all(wind_speed == 0):
            wind_speed += 0.1
                # 设置固定的风速区间
        bins = [0, 0.5, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
        # 创建自定义颜色映射
        colors = ["#E6F3FF", "#CCEBFF", "#99D6FF", "#66C2FF", "#33ADFF", "#0099FF", "#0066CC", "#003366"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue", colors, N=len(bins)-1)

        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = WindroseAxes.from_ax(fig=fig)
        

        # 创建污染玫瑰图
        # 创建风向图
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100, subplot_kw=dict(projection='windrose'))
        try:
            #ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white',
                   #bins=np.arange(0, max(wind_speed), max(wind_speed)/8), nsector=16)
            ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white',
                   bins=bins, cmap=cmap, nsector=16)
            ax.set_title(f"Wind Rose for site {site}\nLat: {site_lat:.4f}, Lon: {site_lon:.4f}", fontsize=10)
            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=bins[0], vmax=bins[-1]))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', aspect=30, pad=0.1)
            cbar.set_label('Wind Speed (m/s)', fontsize=10)

        except Exception as e:
            print(f"Error creating wind rose: {e}")
            st.error(f"An error occurred while creating the wind rose for {site}: {e}")
            continue
        #col1, col2, co13 = st.columns(3)
        #with col2:
        st.pyplot(fig)


if __name__ == "__main__":
    main()
def plot_bak():
    for site, data_obj in footprint_data.items():
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='windrose'))
        try:
            ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white', 
                   bins=np.arange(0, max(wind_speed), max(wind_speed)/8), nsector=16, 
                   colors=plt.cm.jet(plt.Normalize()(concentration)))
            ax.set_title(f"Wind Rose for site {site}\nLat: {site_lat:.4f}, Lon: {site_lon:.4f}")
            if ax.collections:
                plt.colorbar(ax.collections[0], label=f'Average {species} Concentration')
            else:
                print("No collections in ax, trying scatter plot")
                sc = ax.scatter(wind_direction, wind_speed, c=concentration, cmap=plt.cm.jet)
                plt.colorbar(sc, label=f'Average {species} Concentration')
        except Exception as e:
            print(f"Error creating wind rose: {e}")
            st.error(f"An error occurred while creating the wind rose for {site}: {e}")
            continue
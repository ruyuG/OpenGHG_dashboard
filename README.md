# OpenGHG Platform Streamlit Interface

This repository hosts the interface for the OpenGHG platform, designed to improve accessibility and visualization of greenhouse gas (GHG) and ozone-depleting substances (ODS) data. This project enhances the platform's usability.

## Features

### General Layout
The interface comprises a home page and five main sub-interfaces: Observation, Footprint, Flux, ModelScenario, and Unified Data Explorer. The home page provides a concise introduction and quick links to each section, equipped with thumbnail plots for a visual guide.

### Sub-interfaces
- **Observation**: Supports searching, visualizing, and analyzing observational data, with features for data smoothing and linear regression.
- **Footprint**: Enables the retrieval, visualization, and analysis of footprint data with options for interactive graphics or animations.
- **Flux**: Offers tools to explore and visualize flux and emissions estimates, emphasizing regional interaction characteristics.
- **ModelScenario**: Facilitates the creation and analysis of model scenarios, allowing comparisons between modeled and actual observational data.
- **Unified Data Explorer**: Integrates observations and footprints for combined analysis, enhancing data interaction and visualization.

## Installation on BluePebble

This setup is specifically tailored for deployment on the BluePebble server. Follow these steps to install and configure the environment:

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/ruyuG/openghg-interface.git
   ```
2. **Run the Setup Script**:
   ```sh
   bash setup_script.sh
   ```
   This script automates the setup process, including environment creation and dependency installation.

## Usage

To launch the interface after installation:
```bash
streamlit run Home.py
```
Access the platform via a web browser at `http://localhost:8501`.

## Directory Structure

- `data/`: Sample datasets and GIFs for demonstration.
- `image/`: Static images used within the interface.
- `pages/`: Scripts for the interface's sections.
- `Home.py`: Main entry script for the Streamlit application.

## Contributing

Contributions to improve or extend the platform's functionalities are welcome. Please submit pull requests or create issues for any enhancements or bugs.

## License

Distributed under the MIT License. See `LICENSE` for more information.



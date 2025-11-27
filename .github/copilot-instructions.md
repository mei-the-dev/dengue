# Dengue Epidemiological Analysis Project

## Project Overview
This is a Python data science project for epidemiological analysis of dengue fever data in Brazil.

## Project Structure
- `src/` - Source code modules
  - `tarefa1_semanas_epidemiologicas.py` - Epidemiological weeks calendar analysis
  - `tarefa2_normalizacao.py` - Data normalization by population and total infected
  - `tarefa3_distancias.py` - L1 and L2 distance calculation between time series
  - `tarefa4_complexo_simplicial.py` - Simplicial complex creation from distance matrices
  - `utils.py` - Utility functions for data loading and processing
- `data/` - Data directory for dengue cases and population data
- `output/` - Output directory for results and visualizations

## Key Concepts
- **Epidemiological Weeks**: Standard WHO/CDC epidemiological weeks (most years have 52, some have 53)
- **Epidemic Period (PE)**: Period from start to end of a dengue epidemic
- **Normalization 1**: Cases per population (Census 2010) - incidence rate
- **Normalization 2**: Cases normalized by total infected in period - unit area series
- **L1 Distance**: Manhattan distance between time series
- **L2 Distance**: Euclidean distance between time series
- **Simplicial Complex**: Topological structure created from distance matrix with threshold

## Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- matplotlib: Visualization
- scipy: Distance calculations
- epiweeks: Epidemiological week calculations

## Usage
Run the main analysis script or individual task modules as needed.

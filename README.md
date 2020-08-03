# COVID-19 Modeling and Non-Pharmaceutical Interventions in a Dialysis Unit

## Installation

The Python packages used in this project are installed in a conda environment.
Create environment from these files (`*.yml`).
Once this is done, no additional installation of packages is needed.

```
conda env create -f Dialysis_COVID19.yml
conda env create -f igraph.yml
```

## Requirements

If for some reason you fail to create conda environments with above `*.yml` files, install the following libraries. Python3 is used.

- numpy
- pandas
- matplotlib
- networkx
- python-igraph

## Data

- Download data from the following data repository: [somewhere..]()
- Place the data in the `dialysis/data/`

## Run the simulations to generate results in the paper

- Use `Dialysis_COVID19` for the conda environment
```
source activate Dialysis_COVID19
```

- Prepare contact arrays that are used in simulation
```
cd dialysis
./prepare_contact_networks.sh
```

- Compute alpha (scaling parameter for disease models)
The disease model used in the paper are D2 and D3.

```
cd ..
python compute_alpha.py
```

- Run COVID-19 simulation
This takes roughly one day to run using Vinci server (used 60 cores).
Change the value of `cpu` to the number of CPU available in your device.
For each intervention setting, disease model, R0, and infection source (patient or HCP), 
I saved daily infection count, transmission routes, daily population (in case for addition HCP to unit), R0, and the generation time.
All of these are saved in `dialysis/results/day10/` directory with filenames of `final_scenario0.npz` (Scenario 1 in the paper) and `final_scenario1.npz` (Scenario 2 in the paper)
```
python COVID19_dialysis_final.py -cpu 60
```

- Draw figures
The above simulation script do not run simulations on disease models D0 and D1, 
and this generates warning at this stage. Simply ignore the warnings and print results for D2 and D3.
Figures are saved in `dialysis/plots/day10/`
```
python -W ignore COVID19_dialysis_plots_final.py
```

- Generate tables
`COVID19_dialysis_tables_final.py` generates many tables from the simulation result.
Tables are saved in `dialysis/tables/day10/`
```
python -W ignore COVID19_dialysis_tables_final.py
```

## Drawing the contact network and generating network statistics

- Generating network statistics
Results are saved in `tables/statistics/`
```
python generate_statistics.py
python generate_instantaneous_statistics.py
```

- Draw the Contact network
This script uses `python-igraph` package. (this package uses the Cairo library for plotting)
Deactivate `Dialysis_COVID19` environment and activate `igraph` environment.

```
source deactivate
source activate igraph
cd dialysis
python draw_contact_network.py
```

## Cite
This project is currently in submission for the peer-review for the PLOS Computational Biology.
```
...
```

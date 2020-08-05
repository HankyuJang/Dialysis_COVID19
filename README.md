# COVID-19 Modeling and Non-Pharmaceutical Interventions in a Dialysis Unit

This document provides a step-by-step guide to replicate results in the paper.

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

- Download data from the following data repository: [Healthcare Personnel Movement Data](https://www.kaggle.com/hankyujang/healthcare-personnel-movement-data/)
- Place the data in the `dialysis/data/`

## Run the simulations to generate results in the main paper

- Use `Dialysis_COVID19` for the conda environment
```
source activate Dialysis_COVID19
```

- Prepare contact arrays that are used in simulation (Day10)
```
cd dialysis
./prepare_contact_networks.sh
```

- Compute alpha (scaling parameter for shedding models)
The shedding models used in the paper are exp/exp (5%) and exp/exp (35%) that are D2 and D3 in the script.

```
cd ..
python compute_alpha.py
```

- Run COVID-19 simulation
This takes roughly one day to run using Vinci server (used 60 cores).
Change the value of `cpu` to the number of CPU available in your device.
For each intervention setting, disease model, R0, and infection source (patient or HCP), 
we keep track of the following:
(i) daily infection count, (ii) transmission routes, (iii) daily population (in case for addition HCP to unit), (iv) R0, and (v) the generation time.
All of these are saved in `dialysis/results/day10/` directory with filenames of `final_scenario0.npz` (Scenario 1 in the paper) and `final_scenario1.npz` (Scenario 2 in the paper).
```
python COVID19_dialysis_final.py -cpu 60
```

- Draw figures
There are 4 shedding models embedded in the simulator, but for first two models are not used in simulation (D0 and D1).
This generates warning when drawing figures at this stage. 
Simply ignore the warnings and use results for D2 and D3.
Figures are saved in `dialysis/tiff/plots/day10/`.
Note that `-s 0` correspond to Scenario 1 and `-s 1` correspond to Scenario 2.
```
python -W ignore COVID19_dialysis_plots_tiff.py -s 0
python -W ignore COVID19_dialysis_plots_tiff.py -s 1
```

- Generate tables
`COVID19_dialysis_tables_final.py` generates many tables from the simulation result.
Tables are saved in `dialysis/tables/day10/`
```
python -W ignore COVID19_dialysis_tables_final.py -s 0
python -W ignore COVID19_dialysis_tables_final.py -s 1
```

## Drawing the contact network and generating network statistics

- Generating network statistics
Results are saved in `tables/statistics/`
```
python generate_statistics.py
python generate_instantaneous_statistics.py
```

- Draw the contact network
This script uses `python-igraph` package. (this package uses the Cairo library for plotting)
Deactivate `Dialysis_COVID19` environment and activate `igraph` environment.
The contact network is saved in `plots/contact_network/`

```
source deactivate
source activate igraph
cd dialysis
python draw_contact_network.py
```

## Generate results in the Supporting Information section

- Use `Dialysis_COVID19` for the conda environment
```
source deactivate
source activate Dialysis_COVID19
```

- Prepare contact arrays for other days (Day2, Day6, Day7, Day8, Day9)
```
./prepare_contact_networks_other_days.sh
```

- Run COVID-19 simulation
Run on four sets of interventions: Baseline, Baseline+, Baseline++, Baseline+++ (refer to the paper for details).
```
cd ..
./COVID19_dialysis_other_days.sh
```

- Draw figures
```
./COVID19_dialysis_plots_tiff_other_days.sh
```

- Draw contact networks
```
source deactivate
source activate igraph
cd dialysis
./draw_contact_networks_other_days.sh
```

## Cite
This project is currently in submission for the peer-review for the PLOS Computational Biology.
```
...
```

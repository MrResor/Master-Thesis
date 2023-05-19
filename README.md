# Master-Thesis

Program created for my Master Thesis. It computes solutions to Traveling Salesperson Problem (TSP) using different methods. These are:
* Ant Colony System
* Genetic Algorithm
* 2-opt
* Shortest Edge Algorithm
* Particle Swarm Optimisation
* Concorde

## Installation

Install python 3.9.5 ([link](https://www.python.org/downloads/release/python-395/)). In the location where you want to run the application first pull the git repository. Then run these two commands:

```bash
python -m venv ./
python -m pip install -r requirements.txt
```

NOTE: Usage of Concorde requires downloading of the program itself from this [site](https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm) and placing it in the program directory.

## Usage

```bash
python main.py ./path/to/dataset.csv method
```

The dataset passed to the program should be in the same form as [this](https://github.com/MrResor/Master-Thesis/blob/main/five.csv). It is a upper triangle of distance matrix, without main diagonal. From column can be interpreted as index of a row and to as index of a column, while dist is the value that will be found under those indexes. Please note that the indexing in the file starts from 1. Several datasets are provided with a program, those are:

* [five.csv](https://github.com/MrResor/Master-Thesis/blob/main/five.csv)
* [ulysses16.csv](https://github.com/MrResor/Master-Thesis/blob/main/ulysses16.csv)
* [bays29.csv](https://github.com/MrResor/Master-Thesis/blob/main/bays29.csv)
* [att48.csv](https://github.com/MrResor/Master-Thesis/blob/main/att48.csv)
* [pr76.csv](https://github.com/MrResor/Master-Thesis/blob/main/pr76.csv)
* [eil101.csv](https://github.com/MrResor/Master-Thesis/blob/main/eil101.csv)
* [qa194.csv](https://github.com/MrResor/Master-Thesis/blob/main/qa194.csv)
* [xqg237.csv](https://github.com/MrResor/Master-Thesis/blob/main/xqg237.csv)

Equivalents of those files with .points extension contain 2d locations of the nodes for a presentation purpose. One can create his own dataset, either from existing datasets (such as most of the provided ones, excluding five.csv) from [tsplib95](https://www.math.uwaterloo.ca/tsp/data/index.html) or generate your own data (as was done with five.csv) In that case however please remember that the values in the distance matrix should be integers, otherwise the concorde program will not accept them.

Next, available methods can be chosen, names in the program are as follows:

* ants
* genetic
* 2-opt
* sea
* pso
* concorde

For the algorithms appropriate modifications to the variables impacting the solution may be done. Examples are number of tours for Ant colony system or probability of mutation for genetic algorithm. In any case please run -h flag to see the list of variable that may be modified as well as default values.

Note: Activating venv before the given command may be necessary to use the versions of packages used during developement


## License

None
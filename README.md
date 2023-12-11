# Implementation details

The program was developed using *Python* programming language (version 3.8.11), using libraries *Numpy* and *Pandas*, as well as their dependencies. The data sets that the program uses are provided in *.csv* format, with headers 'from', 'to' and 'dist'. Third column, 'dist', holds a distance between nodes 'from' and 'too'. Both 'from' and 'to' are indexes of nodes (starting from 1). These *.csv* files were obtained by transforming distance matrix for a symmetric TSP problem (all the considered data sets come from TSPLIB). It is done by first transforming said matrix into the upper triangle matrix and then writing it down row by row in a *.csv* file, where each row contains 'row' number, 'column' number and distance. Each row corresponds to one non-zero value from the upper triangle matrix, and 'row' and 'column' numbers represent the row and column indexes under which the non-zero value is found (examples of ready data can be seen in [datasets](https://github.com/MrResor/Master-Thesis/tree/develop/datasets) folder). 

It is possible to use a different data set, but the format of the input data must be the same. It is important to note that the Concorde implementation will ignore the decimal part of any digits in 'dist' column as they will be cast to integer values.

## Software installation

Before using the program, the following steps should be done. First, installation of *Python*. Although version 3.8.11 is the best one to use, it cannot be installed on the Windows operating system, though, the *Python* version 3.8.10 can also be used to run the software package. *Python* can be downloaded from official web page (https://www.python.org/downloads/release/python-3810/). During installation it should be ensured that the *pip* is installed and the installation path should be recorded (selecting 'Install now' option installs *pip* package, and the installation path is as in the example: ```C:\Users\User_Name\AppData\Local\Programs\Python\Python38```). Then the following commands should be executed in the location of the program (in PowerShell)

```powershell
installation_path\python.exe -m venv ./
./Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

where the requirements file is provided with the program files. Occasionally, the second command will cause an error and in this case the following command should be executed first

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
```

These steps ensure that most of the program functionality is ready to be used, however Concorde procedure requires two extra files:

* Concorde executable (https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm),
* *cygwin1.dll* file (https://www.dll-files.com/cygwin1.dll.html).

Putting them both in the program location also enables the Concorde software to work.

## Application usage

The following commands should be entered in the command line to run the program

```
python main.py [-h] [-l] path {ants,genetic,sea,pso,2-opt,concorde}
```

where '-h' is a help flag for the program, '-l' is a flag that enables logging, 'path' is the path to a data set file, , and the last element is the algorithm to be run

```powershell
python main.py -l ./datasets/five.csv genetic
```

Each of the algorithms has its own help flag ('-h'), where the modifiable modifiable elements are described. Example

```powershell
python main.py ./datasets/five.csv genetic -h
```

results in output

```
usage: TSP solver path genetic [-h] [-P] [-n] [-p] [-T]

optional arguments:
  -h, --help                  Show this help message and exit.
  -P, --initial-population    Size of initial population. (default: 250)
  -n, --children-multiplier   Multiplier for children populaion size (P * n) (default: 0.8)
  -p, --mutation-probability  Mutation rate for offsprings. (default: 0.5)
  -T, --generation-count      Number of generations. (default: 500)
```

An example of changing algorithm parameters can be seen bellow, where for the ant algorithm the number of tours ('-t') was to 100 and the evaporation coefficient ('-p') was set to 0.4

```
python main.py ./datasets/ulysses16.csv ants -t 100 -p 0.4
```

The results are displayed in three lines. The first line reports the execution time in seconds, the next line presents the shortest distance found by the algorithm, and the last shows the order of nodes that provided the shortest distance.

```
3.1937631
6889.0
15->2->1->3->7->0->11->12->13->6->5->14->4->10->8->9
```
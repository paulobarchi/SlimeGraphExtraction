# SlimeGraphExtraction
Extracting and exploring an undirected graph from the Large-Scale Structure of the Universe using a density field fit of Physarum Machine particles as input.

## python requirements

```bash
pip install -r requirements.txt
```

## File structure

Cwd should have these files:

          .
          ├── configHelper.py
          ├── config.ini
          ├── graph-tool.py
          ├── requirements.txt
          └── subCubeGraph.ipynb

## Configuration file

A configuration file is required to run *graph-tool.py*: 

	config.ini

This file has the following sections: **[Paths]**, **[Files]**, **[PreProcess]**, **[Log]**, **[Graph]**, and **[Operations]**.

It is recommended to save a backup version before changing this config file:
```bash
cp  config.ini config_BCKP.ini
```

Configure each item from each section. Commentaries about the configuration are provided below.

### [Paths]

* ***outputPath***: path where all output graphs will be saved.

### [Files]
* ***depositBin***: name and path of the bin file with the deposit information.
* ***depositCSV***: name and path of the csv file with the deposit information (to be created or read).
* ***kdtree***: name and path of the kdtree file (to be created or read).
* ***baseGraph***: name and path of the base graph file (to be created or read).
* ***edgesFile***: name and path of the edges file (to be saved).
* ***uniqueDF***: name and path of the unique csv dataframe with all information about the graph (to be saved).
* ***connectedGraph***: name and path of the file with the connected graph (to be saved -- output from *connectVertices* operation).

#### Not in use yet
* ***partitionGraph***: name and path of the file with the partition graph (to be saved -- output from *getGraphPartition* operation).
* ***hyperGraph***: name and path of the file with the hypergraph (to be saved -- output from *getHyperGraph* operation).
* ***createHyperGraphFrom***: name and path of the file with the base graph to create the hypergraph (*baseGraph* or *connectedGraph*).

### [PreProcess]
* ***brickSize***: size of each dimension of the data cube (default = 360; when using high resolution, 720).
* ***normalize***: *True* or *False* value to normalize (or not) data values (default = True).
* ***densityThreshold***: minimum value for keeping throughput from data cube when creating the graph (default = 0.9). 

### [Log]

The system logs everything that happens in the following format:

	<datetime> [<LEVEL>]	<file> - <msg>

Example of a line of the log file (first line):

	[2019-06-26 10:48:02,652] [INFO]	preprocess - Starting Pre-Processing for Object Detection.

* ***logFile***: log file name.

* ***level***: you can specify one of two log levels: ***INFO*** or ***DEBUG***. 

  * ***INFO*** is the default value for log level, with which you will get *INFO*, *WARNING* and *CRITICAL* messages.

  * ***DEBUG*** outputs all messages from the system, including values of the variables in each calculation step -- this should be used only on debug runs -- if something goes wrong or if you want to trace every detail. 

### [Graph]
* ***nSubSample***: number of sub cubes for local analysis and/or parallelizing (default = 27)
* ***centerShift***: number of sub cubes to shift when analysing a single sub cube (default = 13)
* ***maxDistEdge***: maximum distance to define an edge (default = 1)
* ***minVertEdges***: minimum vertex degree (default = 1). If *connectVertices = True*, for all vertices with degree = 0, we create *minVertEdges* edges with the closest vertices.

### [Operations] (all *True* or *False*)
* ***createDataFiles***: create *depositCSV* (with coordinates and values for non-zero throughputs of the data cube) and *kdtree* file.
* ***createGraph***: create and save *baseGraph*
* ***connectVertices***: for all vertices with degree = 0, we create *minVertEdges* edges with the closest vertices, and save *connectedGraph*.
* ***exportEdges***: create *edgesFile* with edges data
* ***getUniqueDataFrame***: create *uniqueDF* with edges and vertices (coordinates + throughput values) information.
* ***getGraphPartition***: create graph partitions. Still not implemented in *graph-tool.py [TODO]*, but working on *subCubeGraph.ipynb*.

#### Not in use yet
* ***getHyperGraph***: create hypergraph [TODO].

## Running

* First we run graph-tool.py to preprocess the input (bin) file and save the KDTree with distances among all throughput values.
* Then, we can run subCubeGraph.ipynb to load the KDTree, generate graphs and run community detection algorithms:
```bash
# conda activate <your-environment> # if needed
jupyter notebook
```

## KSPA 2019 report
This project is part of the Kavli Summer Program in Astrophysics (KSPA 2019): Machine Learning in the era of large astronomical surveys. For more details about the project, please see the report at https://www.overleaf.com/read/hnbkgqrwtqzv.

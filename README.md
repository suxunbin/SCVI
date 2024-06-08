# SCVI
SCVI: An Effective Subgraph Matching Algorithm by Combining Edge Verification and Set Intersection

## Compile
Under the root directory of the project, execute the following commands to compile the source code.

```zsh
cd src
make
```

## Run
```
Options:
-h,   the help message
-d,   the filename of the data graph
-q,   the filename of the query graph
-t,   specify the number of matches to find
```

```
Examples:
./SCVI -d data/yeast.SCVI.nt -q query/yeast/query_dense_4_100.graph -t 100000
```


## Input File Format

Data graph file format is a text format to store an undirected graph. 
- The first line of the file should be "t #vertices #edges #labels"
- Following lines of "v vertex-ID vertex-label" indicate the vertices in the graph.
- The vertices should be written in the file in ascending order of their IDs, and a vertex ID should be in [0, #vertices - 1].
- Following lines of "e vertex-ID1 vertex-ID2 edge-label" after the vertices indicate the undirected edges in the graph.

For example:
```
Line "t 3112 12519 71" means the start of a data graph with #vertices=3112, #edges=12519 and #labels=3112.
Line "v 0 0" means there is a vertex with ID=0 and label=1 in the graph.
Line "e 0 1745 0" means there is an undirected edge between vertices with IDs 0 and 1745, where edge label is 0.
```

Query graph file format is a text format to store undirected graphs.
- The first line of a graph should be "t #vertices #edges"
- Following lines of "v vertex-ID vertex-label" indicate the query vertices in the graph.
- The query vertices should be written in the file in ascending order of their IDs, and a vertex ID should be in [0, #vertices - 1].
- Following lines of "e vertex-ID1 vertex-ID2 edge-label" after the vertices indicate the undirected edges in the graph.

For example:
```
Line "t 4 4" means the start of a query graph with #vertices=4 and #edges=4.
Line "v 0 20" means there is a query vertex with ID=0 and label=20 in the query graph.
Line "e 0 1 0" means there is an undirected edge between query vertices with IDs 0 and 1, where edge label is 0.
```

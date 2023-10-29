#ifndef QCSR__h_
#define QCSR__h_
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sys/time.h>
#include "Graph.h"
#include "Query.h"

using namespace std;

class QCSR
{
public:
	unsigned data_vertex_num;
	unsigned query_vertex_num;
	unsigned start_vertex;
	unsigned start_vertex_label;
	unsigned start_vertex_offset;
	bool **vfs; //vertex filtering state: vfs[i][j]=true represents that data vertex i has been already filtered in regard to query vertex j
	unsigned *offset_list;
	vector<unsigned> adjacency_list;
	unsigned **candidate_vertex_set;
	unsigned *candidate_size;
	unsigned **candidate_valid;
	unsigned **candidate_frequency;
	double **candidate_workload; //the workload of the subtree rooted at this candidate
	unsigned *max_mapping_width; //the maximum width of the data vertices of the query veretx
	QCSR(Graph& G, Query& Q);
	~QCSR();
	void build(Graph& G, Query& Q, bool opt_flag);
};
#endif
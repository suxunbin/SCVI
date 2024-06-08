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

	unordered_map<unsigned, pair<unsigned, unsigned>>** offset_list; 
	vector<unsigned> adjacency_list; 

	unsigned **candidate_set; 
	unsigned *candidate_size; 
	unsigned **candidate_flag; 

	QCSR(Graph& G, Query& Q);
	~QCSR();
	void build(Graph& G, Query& Q);
};
#endif
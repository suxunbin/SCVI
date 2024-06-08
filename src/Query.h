#ifndef Query__h_
#define Query__h_

#include <iostream>
#include <vector>
#include <set>
#include <queue>
#include <stack>
#include <algorithm>
#include <string.h>
#include <unordered_map>
#include <string>
#include <fstream>

using namespace std;

class Query
{
public:
	unsigned vertex_num; 
	unsigned edge_num;	
	unsigned *label_list; 
	bool **edge_relation; 
    vector<unsigned> *neighbor_list; 
	unordered_map<unsigned, unsigned> *neighbor_label_frequency;
	vector<unsigned> *backward_neighbor_set; 
	vector<unsigned> *forward_neighbor_set;
	vector<unsigned> *black_backward_neighbor; 
	vector<unsigned> *white_backward_neighbor; 
	vector<unsigned> *repeated_list; 
	vector<unsigned> *repeated_vertex_set;
	uint16_t *vertex_state; 
	vector<unsigned> match_order; 
	vector<unsigned> leaf; 
	unsigned *leaf_num; 
	bool* CEB_flag; 
	bool* CEB_valid; 
	unsigned** CEB; 
	unsigned* CEB_iter; 
	vector<unsigned> *children; 
	vector<unsigned> white_vertex_set; 
	bool* encoding; 

	double *score;
	unsigned start_vertex;
	unsigned start_vertex_nlabel;

	Query();
	~Query();
	virtual void read(const string& filename);
	void CFL_decomposition();
};
#endif
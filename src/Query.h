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
	unsigned *edge_relation;
    vector<unsigned> *neighbor_list;
	unordered_map<unsigned, unsigned> *neighbor_label_frequency;
	vector<unsigned> *backward_neighbor; //the subscript of the backward neighbor of the query vertex located in the matching order
	vector<unsigned> *forward_neighbor; //the forward neighbors (not leaf vertices) of the query vertex
	vector<unsigned> *repeated_list; //the repeated list of each label (record the subscript in the matching order)
	vector<unsigned> *repeated_vertex_set; //the backward repeated vertex of each vertex (record the subscript in the matching order)
	uint16_t *vertex_state; // 0: core_vertex, 1: forest_vertex, 2: leaf_vertex
	uint16_t query_state; // 0: core, 1: core + leaf, 2: tree + leaf, 3: core + forest + leaf
	vector<unsigned> match_order;
	vector<unsigned> core;
	vector<unsigned> core_match_order;
	vector<unsigned> forest;
	vector<unsigned> *forest_match_order;
	vector<unsigned> tree;
	vector<unsigned> tree_match_order;
	vector<unsigned> leaf;
	unsigned *leaf_num;
	unsigned repeat_leaf_begin_loc;
	unsigned repeat_two_leaf_begin_loc;
	bool* CEB_flag; //default false:no CEB, true:enable CEB
	bool* CEB_update; //false:CEB is invalid, true:CEB is valid
	unsigned* CEB_width; //the CEB width
	unsigned* CEB_father_idx; //the parent vertex of each vertex
	unsigned* CEB_write; //(x > 0) means that we write the common results to CEB[x]
	unsigned** CEB;
	unsigned* CEB_iter;
	vector<unsigned> *parent; //record the children of the parent vertex
	unsigned* repeat;
	bool* repeat_state; //false:not repeated, true:repeated
	unsigned* NEC; //0:no equivalence relation, >0: denotes the equivalence class

	double *score;
	unsigned start_vertex;
	unsigned start_vertex_nlabel;
	unsigned max_degree = 0;

	Query();
	~Query();
	virtual void read(const string& filename);
	void CFL_decomposition();
};
#endif
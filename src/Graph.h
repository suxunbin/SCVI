#ifndef Graph__h_
#define Graph__h_

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <string.h>
#include <string>
#include <fstream>

using namespace std;

class Graph
{
public:
	unsigned vertex_num;
	unsigned edge_num;
	unsigned label_num;
	unsigned *label_list;
	unsigned *label_frequency;
	unsigned *label_index_list;
	bool **nsa; //neighborhood structure array
	unsigned *offset_list;
	unsigned *adjacency_list;
	vector<unsigned> **partitioned_vertex_list;
	bool state;

	unsigned max_degree = 0;
	unsigned label_max_frequency = 0;
	unsigned average_degree;

	pair<unsigned, unsigned>** edge_index; 
	vector<unsigned*> head_pointer_list;
	bool **BloomFilter1;
	bool **BloomFilter2;
	//bool **BloomFilter3;
	uint64_t ***hash_bucket;
	unsigned range;
	Graph();
	~Graph();
	virtual void read(const string& filename);
	void count_edge();
	void setBloomFilter();
	void readBloomFilter(const string& filename);
	void writeBloomFilter(const string& filename);
	static uint32_t BKDR_hash(uint32_t value, uint32_t threshold);
	static uint32_t AP_hash(uint32_t value, uint32_t threshold);
	static uint32_t DJB_hash(uint32_t value, uint32_t threshold);
	static uint32_t bitSumHash(uint32_t value, uint32_t threshold);
	static unsigned long long MurmurHash64B(const void *key, int len);
};
#endif
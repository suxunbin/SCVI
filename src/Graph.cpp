#include "Graph.h"

Graph::Graph()
{
}

Graph::~Graph()
{
	delete[] this->label_list;
	delete[] this->label_frequency;
	delete[] this->label_index_list;
	for (unsigned i = 0; i < this->vertex_num; ++i)
		delete[] this->nsa[i];
	delete[] this->offset_list;
	delete[] this->adjacency_list;
	for (unsigned i = 0; i < this->label_num; ++i)
		delete[] this->partitioned_vertex_list[i];	
}

bool cmp(const pair<unsigned, unsigned>& x, const pair<unsigned, unsigned>& y)
{
	return x.second < y.second;
}

uint32_t Graph::BKDR_hash(uint32_t value, uint32_t threshold) {
	//BKDR hash
	uint32_t seed = 31;
	uint32_t hash = 0;
	while(value) {
		hash = hash * seed + (value&0xF);
		value >>= 4;
	}
	return (hash & 0x7FFFFFFF) % threshold;
}

uint32_t Graph::AP_hash(uint32_t value, uint32_t threshold) {
	//AP hash
	unsigned int hash = 0;
	int i;
	for (i=0; value != 0; i++) {
		if ((i & 1) == 0) {
			hash ^= ((hash << 7) ^ (value&0xF) ^ (hash >> 3));
		} else {
			hash ^= (~((hash << 11) ^ (value&0xF) ^ (hash >> 5)));
		}
		value >>= 4;
	}
	return (hash & 0x7FFFFFFF)%threshold;
}

uint32_t Graph::DJB_hash(uint32_t value, uint32_t threshold) {
    unsigned int hash = 5381;
    while (value)
    {
        hash += (hash << 5) + (value&0xFF);
		value >>= 8;
    }
    return (hash & 0x7FFFFFFF) % threshold;
}

uint32_t Graph::bitSumHash(uint32_t value, uint32_t threshold) {
	return ((value & 0xff)+((value>>8) & 0xff)+((value>>16) &0xff)+((value >> 24)&0xff))%threshold;
}

unsigned long long Graph::MurmurHash64B(const void *key, int len)
{
	const unsigned int seed = 0xEE6B27EB;
	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	unsigned int h1 = seed ^ len;
	unsigned int h2 = 0;

	const unsigned int *data = (const unsigned int *)key;

	while (len >= 8) {
		unsigned int k1 = *data++;
		k1 *= m;
		k1 ^= k1 >> r;
		k1 *= m;
		h1 *= m;
    	h1 ^= k1;
    	len -= 4;

    	unsigned int k2 = *data++;
    	k2 *= m;
    	k2 ^= k2 >> r;
    	k2 *= m;
    	h2 *= m;
    	h2 ^= k2;
    	len -= 4;
  	}

  	if (len >= 4) {
    	unsigned int k1 = *data++;
    	k1 *= m;
    	k1 ^= k1 >> r;
    	k1 *= m;
    	h1 *= m;
    	h1 ^= k1;
    	len -= 4;
  	}

  	switch (len) {
  	case 3:
    	h2 ^= ((unsigned char *)data)[2] << 16;
  	case 2:
    	h2 ^= ((unsigned char *)data)[1] << 8;
  	case 1:
    	h2 ^= ((unsigned char *)data)[0];
    	h2 *= m;
  	};

  	h1 ^= h2 >> 18;
  	h1 *= m;
  	h2 ^= h1 >> 22;
  	h2 *= m;
  	h1 ^= h2 >> 17;
  	h1 *= m;
  	h2 ^= h1 >> 19;
  	h2 *= m;

  	unsigned long long h = h1;

  	h = (h << 32) | h2;

  	return h;
}

void Graph::read(const string& filename) // read data graph
{
	char type;
	unsigned vid;
	unsigned src_vid;
	unsigned dst_vid;
	unsigned label;
	vector<unsigned> *edge_list;
	ifstream fin(filename.c_str());
	while (fin>>type)
	{
		if (type == 't')
		{
			fin >> this->vertex_num;
			fin >> this->edge_num;
			fin >> this->label_num;
			this->label_list = new unsigned[this->vertex_num];
			edge_list = new vector<unsigned>[this->vertex_num];
			this->label_frequency = new unsigned[this->label_num];
			memset(this->label_frequency, 0, sizeof(unsigned)*this->label_num);
			this->label_index_list = new unsigned[this->label_num+1];
			this->nsa = new bool*[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
			{
				this->nsa[i] = new bool[this->label_num];
				for (unsigned j = 0; j < this->label_num; ++j)
					this->nsa[i][j] = false;
			}
			this->offset_list = new unsigned[this->vertex_num*this->label_num+1];
			this->adjacency_list = new unsigned[this->edge_num*2];
			this->partitioned_vertex_list = new vector<unsigned>*[this->label_num];
			for (unsigned i = 0; i < this->label_num; ++i)
				this->partitioned_vertex_list[i] = new vector<unsigned>[this->label_num];
		}
		else if (type == 'v')
		{
			fin >> vid >> label;
			this->label_list[vid] = label;
			this->label_frequency[label]++;	
		}
		else if (type == 'e')
		{
			fin >> src_vid >> dst_vid >> label;
			edge_list[src_vid].push_back(dst_vid); 
			edge_list[dst_vid].push_back(src_vid);
		}
	}
	fin.close();
	unsigned label_index = 0;
	for (unsigned i = 0; i < this->label_num; ++i)
	{
		this->label_index_list[i] = label_index;
		label_index += this->label_frequency[i];
		if (this->label_frequency[i] > this->label_max_frequency)
			this->label_max_frequency = this->label_frequency[i];
	}
	this->label_index_list[this->label_num] = label_index;

	//reorder the vertex
	pair<unsigned, unsigned>* vertex_list = new pair<unsigned, unsigned>[this->vertex_num]; //current_id to old_id
	unsigned* iter = new unsigned[this->label_num];
	for (unsigned i = 0; i < this->label_num; ++i)
		iter[i] = this->label_index_list[i];
	for (unsigned i = 0; i < this->vertex_num; ++i)
	{
		vertex_list[iter[this->label_list[i]]] = make_pair(i, edge_list[i].size());
		iter[this->label_list[i]]++;
	}
	for (unsigned i = 0; i < this->label_num; ++i)
		sort(vertex_list+this->label_index_list[i], vertex_list+this->label_index_list[i+1], cmp);


	//vertex relabeling
	unsigned* inverted_index = new unsigned[this->vertex_num]; //old_id to current_id
	for (unsigned i = 0; i < this->vertex_num; ++i)
		inverted_index[vertex_list[i].first] = i;

	unsigned offset = 0;
	unsigned neighbor;
	for (unsigned i = 0; i < this->vertex_num; ++i)
	{
		for (unsigned j = 0; j < this->label_num; ++j)
			this->offset_list[this->label_num*i+j] = offset;
		for (unsigned j = 0; j < vertex_list[i].second; ++j)
		{
			neighbor = edge_list[vertex_list[i].first][j];
			label = this->label_list[vertex_list[i].first];
			this->adjacency_list[offset+j] = inverted_index[neighbor];
			if (!this->nsa[i][this->label_list[neighbor]])
			{
				this->nsa[i][this->label_list[neighbor]] = true;
				this->partitioned_vertex_list[this->label_list[neighbor]][label].push_back(i);
			}
			for (unsigned k = this->label_list[neighbor]+1; k < this->label_num; ++k)
				this->offset_list[this->label_num*i+k]++;
		}
		sort(this->adjacency_list+offset,this->adjacency_list+offset+vertex_list[i].second);	
		if (vertex_list[i].second > this->max_degree)
			this->max_degree = vertex_list[i].second;
		offset += vertex_list[i].second;
	}
	this->offset_list[this->vertex_num*this->label_num] = offset;
	this->average_degree = (2 * this->edge_num) / this->vertex_num;

	offset = 0;
	for (unsigned i = 0; i < this->label_num; ++i)
	{
		for (unsigned j = 0; j < this->label_frequency[i]; ++j)
		{
			this->label_list[offset] = i;
			offset++;
		}
	}
}

void Graph::setBloomFilter() // set ECI
{
	unsigned rid1, cid1;
	unsigned rid2, cid2;
	uint64_t len;
	uint64_t edge;
	this->range = sqrt(this->edge_num*64);
	this->BloomFilter1 = new bool*[this->range];
	this->BloomFilter2 = new bool*[this->range];
	this->hash_bucket = new uint64_t**[this->range];
	for (unsigned i = 0; i < this->range; ++i)
	{
		this->BloomFilter1[i] = new bool[this->range];
		this->BloomFilter2[i] = new bool[this->range];
		this->hash_bucket[i] = new uint64_t*[this->range];
		for (unsigned j = 0; j < this->range; ++j)
		{
			this->BloomFilter1[i][j] = false;
			this->BloomFilter2[i][j] = false;
			this->hash_bucket[i][j] = new uint64_t[5];
			for (unsigned k = 0; k < 5; ++k)
				this->hash_bucket[i][j][k] = 0;
		}
	}
	for (unsigned i = 0; i < this->vertex_num; ++i)
	{
		for (unsigned j = this->offset_list[i*this->label_num]; j < this->offset_list[(i+1)*this->label_num]; ++j)
		{
			if (this->adjacency_list[j] > i)
			{
				rid1 = BKDR_hash(i, range);
				cid1 = BKDR_hash(this->adjacency_list[j], range);
				rid2 = AP_hash(i, range);
				cid2 = AP_hash(this->adjacency_list[j], range);
				len = this->hash_bucket[rid1][cid1][0]; 
				edge = ((uint64_t)i << 32) | ((uint64_t)this->adjacency_list[j]); 
				if (len < 4)
				{
					this->hash_bucket[rid1][cid1][0]++;
					this->hash_bucket[rid1][cid1][this->hash_bucket[rid1][cid1][0]] = edge;
				}
				else
				{
					this->hash_bucket[rid2][cid2][0]++;
					this->hash_bucket[rid2][cid2][this->hash_bucket[rid2][cid2][0]] = edge;
					this->hash_bucket[rid1][cid1][0]++;
				}
				this->BloomFilter1[rid1][cid1] = true;
				this->BloomFilter2[rid2][cid2] = true;
			}
		}
	}
}

void Graph::writeBloomFilter(const string& filename) // write ECI to a file
{
	ofstream fout(filename.c_str());
	for (unsigned i = 0; i < this->range; ++i)
	{
		for (unsigned j = 0; j < this->range; ++j)
			fout << this->BloomFilter1[i][j] << " ";
		fout << endl;
	}

	for (unsigned i = 0; i < this->range; ++i)
	{
		for (unsigned j = 0; j < this->range; ++j)
			fout << this->BloomFilter2[i][j] << " ";
		fout << endl;
	}	

	unsigned size;
	for (unsigned i = 0; i < this->range; ++i)
	{
		for (unsigned j = 0; j < this->range; ++j)
		{
			fout << this->hash_bucket[i][j][0] << " ";
			if (this->hash_bucket[i][j][0] <= 4)
				size = this->hash_bucket[i][j][0];
			else
				size = 4;
			for (unsigned k = 1; k <= size; ++k)
				fout << this->hash_bucket[i][j][k] << " ";
		}
		fout << endl;
	}
	fout.close();
}

void Graph::readBloomFilter(const string& filename) // read ECI from a file
{
	this->range = sqrt(this->edge_num*64);
	this->BloomFilter1 = new bool*[this->range];
	this->BloomFilter2 = new bool*[this->range];
	this->hash_bucket = new uint64_t**[this->range];
	for (unsigned i = 0; i < this->range; ++i)
	{
		this->BloomFilter1[i] = new bool[this->range];
		this->BloomFilter2[i] = new bool[this->range];
		this->hash_bucket[i] = new uint64_t*[this->range];
		for (unsigned j = 0; j < this->range; ++j)
			this->hash_bucket[i][j] = new uint64_t[5];
	}

	ifstream fin(filename.c_str());
	for (unsigned i = 0; i < this->range; ++i)
	{
		for (unsigned j = 0; j < this->range; ++j)
			fin >> this->BloomFilter1[i][j];
	}

	for (unsigned i = 0; i < this->range; ++i)
	{
		for (unsigned j = 0; j < this->range; ++j)
			fin >> this->BloomFilter2[i][j];
	}	

	unsigned size;
	for (unsigned i = 0; i < this->range; ++i)
	{
		for (unsigned j = 0; j < this->range; ++j)
		{
			fin >> this->hash_bucket[i][j][0];
			if (this->hash_bucket[i][j][0] <= 4)
				size = this->hash_bucket[i][j][0];
			else
				size = 4;
			for (unsigned k = 1; k <= size; ++k)
				fin >> this->hash_bucket[i][j][k];
			for (unsigned k = size+1; k <= 4; ++k)
				this->hash_bucket[i][j][k] = 0;
		}
	}
	fin.close();
}
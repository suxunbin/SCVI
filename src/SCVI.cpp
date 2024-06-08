#include <iostream>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <string>
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include <sys/stat.h>
#include <x86intrin.h>
#include <omp.h>
#include <ctime>
#include <dlfcn.h>
#include <immintrin.h>
#include "QCSR.h"
#include <algorithm>
#define hybrid_opt

using namespace std;

struct timeval begin_tv, end_tv;
struct timeval temp_begin_tv, temp_end_tv;
__m256i vec_256_1, vec_256_2, vec_256_res;
__m128i vec_128_1, vec_128_2, vec_128_res;

bool file_exist(const string& filename)
{
	struct stat buffer;
	return (stat(filename.c_str(), &buffer) == 0);
}

string getArgValue(int argc, char* argv[], string op, string default_value)
{
    for (int i = 1; i < argc; ++i)
	{
		if (argv[i] == op)
		{
			if (i + 1 >= argc)
				return "";
			else
				return argv[i + 1]; 
		}
	}
	return default_value;
}

void getStartVertex(Graph& G, Query& Q)
{
	unsigned label;
	unsigned neighbor;
	unsigned nlabel;
	unsigned min_size;
	unsigned min_nlabel;
	double min_score= 1000000000.0;

	for (unsigned i = 0; i < Q.vertex_num; ++i)
	{
    	if (Q.vertex_state[i] != 2)
    	{
			label = Q.label_list[i];
			min_size = 1000000000;
			for (unsigned j = 0; j < Q.neighbor_list[i].size(); ++j)
			{
				neighbor = Q.neighbor_list[i][j];
				nlabel = Q.label_list[neighbor];
				if (G.partitioned_vertex_list[nlabel][label].size() < min_size)
				{
					min_size = G.partitioned_vertex_list[nlabel][label].size();
					min_nlabel = nlabel; 
				}
			}
			Q.score[i] = (double)min_size / (double)Q.neighbor_list[i].size();
			if (Q.score[i] < min_score)
			{
				min_score = Q.score[i];
				Q.start_vertex = i;
				Q.start_vertex_nlabel = min_nlabel;
			}
    	}
	}
}

bool edge_check(Graph& G, Query& Q, int& depth, unsigned& cnt, unsigned** &vec, unsigned* &begin, unsigned* &end, unsigned** &embedding, unsigned* &temp_candidate_set, unsigned& temp_size, unsigned& min_idx, bool encoding)
{
	unsigned vertex = Q.match_order[depth];
	if (Q.encoding[vertex])
	{
		unsigned x, x1, x2;
		unsigned idx;
		unsigned *id = new unsigned[cnt];
		uint64_t edge;
		unsigned mask;
		bool flag;
		unsigned len = 0; 
		unsigned h1[cnt];
		unsigned h2[cnt];
		for (unsigned i = 0; i < cnt; ++i)
		{
			if (i == min_idx)
				continue;
			idx = Q.black_backward_neighbor[vertex][i];
			id[i] = embedding[idx][embedding[idx][0]];
			h1[i] = Graph::BKDR_hash(id[i], G.range);
			h2[i] = Graph::AP_hash(id[i], G.range);
		}
		for (unsigned i = begin[min_idx]; i < end[min_idx]; ++i)
		{
			x = vec[min_idx][i];
			flag = true;	
			for (unsigned j = 0; j < Q.repeated_vertex_set[vertex].size(); ++j) 
			{
				if (embedding[Q.repeated_vertex_set[vertex][j]][embedding[Q.repeated_vertex_set[vertex][j]][0]] == x)	
				{
					flag = false;
					break;
				}
			}
			if (!flag)
				continue;
			x1 = Graph::BKDR_hash(x, G.range);
			x2 = Graph::AP_hash(x, G.range);
			for (unsigned j = 0; j < cnt; ++j)
			{
				if (j == min_idx)
					continue;
				if (x < id[j])
				{
					if ((!G.BloomFilter1[x1][h1[j]]) || (!G.BloomFilter2[x2][h2[j]]))
					{
						flag = false;
						break;
					}
					edge = ((uint64_t)x << 32) | ((uint64_t)id[j]);
					vec_256_1 = _mm256_set1_epi64x(edge);
					vec_256_2 = _mm256_loadu_si256((__m256i *)(G.hash_bucket[x1][h1[j]]+1));
					vec_256_res = _mm256_cmpeq_epi64(vec_256_2, vec_256_1);
					mask = _mm256_movemask_epi8(vec_256_res);
					if (mask == 0)
					{
						if (G.hash_bucket[x1][h1[j]][0] <= 4)
						{
							flag = false;
							break;
						}
						else
						{
							vec_256_2 = _mm256_loadu_si256((__m256i *)(G.hash_bucket[x2][h2[j]]+1));
							vec_256_res = _mm256_cmpeq_epi64(vec_256_2, vec_256_1);
							mask = _mm256_movemask_epi8(vec_256_res);
							if (mask == 0)
							{
								flag = false;
								break;							
							}						
						}
					}
				}
				else
				{
					if ((!G.BloomFilter1[h1[j]][x1]) || (!G.BloomFilter2[h2[j]][x2]))
					{
						flag = false;
						break;
					}
					edge = ((uint64_t)id[j] << 32) | ((uint64_t)x);
					vec_256_1 = _mm256_set1_epi64x(edge);
					vec_256_2 = _mm256_loadu_si256((__m256i *)(G.hash_bucket[h1[j]][x1]+1));
					vec_256_res = _mm256_cmpeq_epi64(vec_256_2, vec_256_1);
					mask = _mm256_movemask_epi8(vec_256_res);
					if (mask == 0)
					{
						if (G.hash_bucket[h1[j]][x1][0] <= 4)
						{
							flag = false;
							break;
						}
						else
						{
							vec_256_2 = _mm256_loadu_si256((__m256i *)(G.hash_bucket[h2[j]][x2]+1));
							vec_256_res = _mm256_cmpeq_epi64(vec_256_2, vec_256_1);
							mask = _mm256_movemask_epi8(vec_256_res);
							if (mask == 0)
							{
								flag = false;
								break;							
							}						
						}
					}
				}
			}
			if (flag)
			{
				if (Q.white_backward_neighbor[vertex].size() == 0)
				{
					embedding[depth][0]++;
					embedding[depth][embedding[depth][0]] = x;
					len++;
				}
				else
				{
					temp_candidate_set[temp_size] = x;
					temp_size++;					
				}
			}
		}
		if (len > 0)
			return true;
		if (temp_size > 0)
			return true;
		return false;
	}
	else 
	{
		if (encoding) 
		{
			unsigned x, x1, x2;
			unsigned idx;
			unsigned *id = new unsigned[cnt];
			uint64_t edge;
			unsigned mask;
			bool flag;
			unsigned len = 0;
			unsigned h1[cnt];
			unsigned h2[cnt];
			for (unsigned i = 0; i < cnt; ++i)
			{
				if (i == min_idx)
					continue;
				idx = Q.black_backward_neighbor[vertex][i];
				id[i] = embedding[idx][embedding[idx][0]];
				h1[i] = Graph::BKDR_hash(id[i], G.range);
				h2[i] = Graph::AP_hash(id[i], G.range);
			}
			for (unsigned i = begin[min_idx]; i < end[min_idx]; ++i)
			{
				x = vec[min_idx][i];
				flag = true;	
				for (unsigned j = 0; j < Q.repeated_vertex_set[vertex].size(); ++j)
				{
					if (embedding[Q.repeated_vertex_set[vertex][j]][embedding[Q.repeated_vertex_set[vertex][j]][0]] == x)	
					{
						flag = false;
						break;
					}
				}
				if (!flag)
					continue;
				x1 = Graph::BKDR_hash(x, G.range);
				x2 = Graph::AP_hash(x, G.range);
				for (unsigned j = 0; j < cnt; ++j)
				{
					if (j == min_idx)
						continue;
					if (x < id[j])
					{
						if ((!G.BloomFilter1[x1][h1[j]]) || (!G.BloomFilter2[x2][h2[j]]))
						{
							flag = false;
							break;
						}
						edge = ((uint64_t)x << 32) | ((uint64_t)id[j]);
						vec_256_1 = _mm256_set1_epi64x(edge);
						vec_256_2 = _mm256_loadu_si256((__m256i *)(G.hash_bucket[x1][h1[j]]+1));
						vec_256_res = _mm256_cmpeq_epi64(vec_256_2, vec_256_1);
						mask = _mm256_movemask_epi8(vec_256_res);
						if (mask == 0)
						{
							if (G.hash_bucket[x1][h1[j]][0] <= 4)
							{
								flag = false;
								break;
							}
							else
							{
								vec_256_2 = _mm256_loadu_si256((__m256i *)(G.hash_bucket[x2][h2[j]]+1));
								vec_256_res = _mm256_cmpeq_epi64(vec_256_2, vec_256_1);
								mask = _mm256_movemask_epi8(vec_256_res);
								if (mask == 0)
								{
									flag = false;
									break;							
								}						
							}
						}
					}
					else
					{
						if ((!G.BloomFilter1[h1[j]][x1]) || (!G.BloomFilter2[h2[j]][x2]))
						{
							flag = false;
							break;
						}
						edge = ((uint64_t)id[j] << 32) | ((uint64_t)x);
						vec_256_1 = _mm256_set1_epi64x(edge);
						vec_256_2 = _mm256_loadu_si256((__m256i *)(G.hash_bucket[h1[j]][x1]+1));
						vec_256_res = _mm256_cmpeq_epi64(vec_256_2, vec_256_1);
						mask = _mm256_movemask_epi8(vec_256_res);
						if (mask == 0)
						{
							if (G.hash_bucket[h1[j]][x1][0] <= 4)
							{
								flag = false;
								break;
							}
							else
							{
								vec_256_2 = _mm256_loadu_si256((__m256i *)(G.hash_bucket[h2[j]][x2]+1));
								vec_256_res = _mm256_cmpeq_epi64(vec_256_2, vec_256_1);
								mask = _mm256_movemask_epi8(vec_256_res);
								if (mask == 0)
								{
									flag = false;
									break;							
								}						
							}
						}
					}
				}
				if (flag)
				{
					if (Q.white_backward_neighbor[vertex].size() == 0)
					{
						embedding[depth][0]++;
						embedding[depth][embedding[depth][0]] = x;
						len++;
					}
					else
					{
						temp_candidate_set[temp_size] = x;
						temp_size++;					
					}
				}
			}	
			if (len > 0)
			{
				embedding[depth][0]++;
				embedding[depth][embedding[depth][0]] = len;
				return true;
			}	
			if (temp_size > 0)
				return true;
			return false;		
		}	
	}
}

bool set_intersection(Query& Q, int& depth, unsigned& cnt, unsigned** &vec, unsigned* &begin, unsigned* &end, unsigned** &embedding, unsigned* &temp_candidate_set, unsigned& temp_size, bool encoding)
{ 
	unsigned vertex = Q.match_order[depth]; 
	if (Q.encoding[vertex])
	{
		unsigned cmn = 1; 
		unsigned len = 0; 
		bool flag;
		bool bflag;
		//unsigned idx;
		unsigned cur = 0; 
		unsigned pre = cnt - 1; 
		unsigned x, y;
		while (true)
		{
			bflag = true;
			for (unsigned i = 0; i < cnt; ++i)
			{
				if (begin[i] == end[i])
				{
					bflag = false;
					break;
				}
			}
			if (!bflag)
				break;
			x = vec[cur][begin[cur]];
			y = vec[pre][begin[pre]]; 
			if (x < y)
			{
				begin[cur]++;
			}
			else if (x > y)
			{
				cmn = 1;
				begin[pre]++;
				pre = cur;
				if (cur == cnt - 1)
					cur = 0;
				else
					cur++;
			}
			else
			{
				cmn++;
				pre = cur;
				if (cur == cnt - 1)
					cur = 0;
				else
					cur++;	
				if (cmn == cnt)
				{
					for (unsigned i = 0; i < cnt; ++i)
						begin[i]++;			
					cmn = 1;


					flag = true;
					for (unsigned i = 0; i < Q.repeated_vertex_set[vertex].size(); ++i) 
					{
						if (embedding[Q.repeated_vertex_set[vertex][i]][embedding[Q.repeated_vertex_set[vertex][i]][0]] == x)	
						{
							flag = false;
							break;
						}
					}
					if (flag)
					{
						if (Q.white_backward_neighbor[vertex].size() == 0)
						{
							embedding[depth][0]++;
							embedding[depth][embedding[depth][0]] = x;
							len++;
						}
						else
						{
							temp_candidate_set[temp_size] = x;
							temp_size++;					
						}
					}
				}		
			}
		}
		if (len > 0)
			return true;
		if (temp_size > 0)
			return true;
		return false;	
	}
	else 
	{
		if (encoding) 
		{
			unsigned cmn = 1; 
			unsigned len = 0; 
			bool flag;
			bool bflag;
			//unsigned idx;
			unsigned cur = 0; 
			unsigned pre = cnt - 1; 
			unsigned x, y;
			while (true)
			{
				bflag = true;
				for (unsigned i = 0; i < cnt; ++i)
				{
					if (begin[i] == end[i])
					{
						bflag = false;
						break;
					}
				}
				if (!bflag)
					break;
				x = vec[cur][begin[cur]];
				y = vec[pre][begin[pre]]; 
				if (x < y)
				{
					begin[cur]++;
				}
				else if (x > y)
				{
					cmn = 1;
					begin[pre]++;
					pre = cur;
					if (cur == cnt - 1)
						cur = 0;
					else
						cur++;
				}
				else
				{
					cmn++;
					pre = cur;
					if (cur == cnt - 1)
						cur = 0;
					else
						cur++;	
					if (cmn == cnt)
					{
						for (unsigned i = 0; i < cnt; ++i)
							begin[i]++;			
						cmn = 1;

						flag = true;
						for (unsigned i = 0; i < Q.repeated_vertex_set[vertex].size(); ++i) 
						{
							if (embedding[Q.repeated_vertex_set[vertex][i]][embedding[Q.repeated_vertex_set[vertex][i]][0]] == x)	
							{
								flag = false;
								break;
							}
						}
						if (flag)
						{
							if (Q.white_backward_neighbor[vertex].size() == 0)
							{
								embedding[depth][0]++;
								embedding[depth][embedding[depth][0]] = x;
								len++;
							}
							else
							{
								temp_candidate_set[temp_size] = x;
								temp_size++;					
							}
						}
					}		
				}
			}	
			if (len > 0)
			{
				embedding[depth][0]++;
				embedding[depth][embedding[depth][0]] = len;
				return true;
			}	
			if (temp_size > 0)
				return true;
			return false;
		}
		else 
		{
			unsigned cmn = 1; 
			unsigned len = 0; 
			bool flag;
			bool bflag;
			//unsigned idx;
			unsigned cur = 0; 
			unsigned pre = cnt - 1; 
			unsigned x, y;
			while (true)
			{
				bflag = true;
				for (unsigned i = 0; i < cnt; ++i)
				{
					if (begin[i] == end[i])
					{
						bflag = false;
						break;
					}
				}
				if (!bflag)
					break;
				x = vec[cur][begin[cur]];
				y = vec[pre][begin[pre]]; 
				if (x < y)
				{
					begin[cur]++;
				}
				else if (x > y)
				{
					cmn = 1;
					begin[pre]++;
					pre = cur;
					if (cur == cnt - 1)
						cur = 0;
					else
						cur++;
				}
				else
				{
					cmn++;
					pre = cur;
					if (cur == cnt - 1)
						cur = 0;
					else
						cur++;	
					if (cmn == cnt)
					{
						for (unsigned i = 0; i < cnt; ++i)
							begin[i]++;			
						cmn = 1;

						flag = true;
						for (unsigned i = 0; i < Q.repeated_vertex_set[vertex].size(); ++i) 
						{
							if (embedding[Q.repeated_vertex_set[vertex][i]][embedding[Q.repeated_vertex_set[vertex][i]][0]] == x)	
							{
								flag = false;
								break;
							}
						}
						if (flag)
						{
							embedding[depth][0]++;
							embedding[depth][embedding[depth][0]] = x;
							len++;
						}
					}		
				}
			}	
			if (len > 0)
			{
				embedding[depth][0]++;
				embedding[depth][embedding[depth][0]] = len;
				return true;
			}		
			return false;
		}
	}
}


void enumerate(Graph& G, Query& Q, QCSR& qcsr, unsigned** &embedding, unsigned& res_num, unsigned& limited_num)
{
	int depth = 0; 
	unsigned* begin_loc = new unsigned[Q.vertex_num]; 
	unsigned* end_loc = new unsigned[Q.vertex_num]; 
	unsigned* temp_begin_loc = new unsigned[Q.vertex_num]; 
	unsigned* temp_end_loc = new unsigned[Q.vertex_num]; 
	unsigned** temp_address = new unsigned*[Q.vertex_num]; 
	unsigned v; 
	unsigned idx; 
	unsigned vertex; 
	unsigned cur_vertex; 
	unsigned next_vertex; 
	unsigned* white_mapping = new unsigned[Q.vertex_num]; 
	unsigned* white_begin_loc = new unsigned[Q.vertex_num]; 
	unsigned* white_end_loc = new unsigned[Q.vertex_num]; 
	unsigned* white_len = new unsigned[Q.vertex_num]; 
	unsigned black_size; 
	unsigned white_size; 
	unsigned* temp_candidate_set = new unsigned[10000]; 
	unsigned temp_size; 
	unsigned* iter_before_split = new unsigned[Q.vertex_num]; 
	unsigned min_len; 
	unsigned total_len; 
	unsigned min_idx; 
	unsigned temp_iter; 
	unsigned len;
	unsigned new_len; 
	unsigned size;
	bool ret; 
	bool flag;
	unsigned cnt;
	unsigned label;
	unsigned leaf_begin_loc;
	unsigned leaf_end_loc;
	unordered_map<unsigned, pair<unsigned, unsigned>>::iterator iter;

	for (unsigned i = 0; i < qcsr.candidate_size[Q.start_vertex]; ++i)
	{
		embedding[0][0]++;
		embedding[0][embedding[0][0]] = qcsr.candidate_set[Q.start_vertex][i];
	}

	while (depth != -1) //DFS match
	{
		if (depth == Q.match_order.size())
		{
			cnt = 1;
			for (unsigned i = 0; i < Q.white_vertex_set.size(); ++i)
			{
				idx = Q.white_vertex_set[i];
				cnt *= embedding[idx][embedding[idx][0]];
			}
			for (unsigned i = 0; i < Q.leaf.size(); ++i)
			{
				idx = Q.backward_neighbor_set[Q.leaf[i]][0];
				vertex = Q.match_order[idx];
				if (Q.encoding[vertex]) 
				{
					v = embedding[idx][embedding[idx][0]];
					label = Q.label_list[Q.leaf[i]];
					leaf_begin_loc = G.offset_list[v*G.label_num+label];
					leaf_end_loc =  G.offset_list[v*G.label_num+label+1];
					len = 0;
					for (unsigned j = leaf_begin_loc; j < leaf_end_loc; ++j)
					{
						flag = true;
						for (unsigned k = 0; k < Q.repeated_vertex_set[Q.leaf[i]].size(); ++k) 
						{
							if (embedding[Q.repeated_vertex_set[Q.leaf[i]][k]][embedding[Q.repeated_vertex_set[Q.leaf[i]][k]][0]] == G.adjacency_list[j])
							{
								flag = false;
								break;
							}
						}
						if (flag)
							len++;
					}
					cnt *= len;
				}
				else 
				{
					size = embedding[idx][embedding[idx][0]];
					len = 0;
					for (unsigned t = embedding[idx][0]-size; t < embedding[idx][0]; ++t)
					{
						v = embedding[idx][t];
						label = Q.label_list[Q.leaf[i]];
						leaf_begin_loc = G.offset_list[v*G.label_num+label];
						leaf_end_loc =  G.offset_list[v*G.label_num+label+1];	
						for (unsigned j = leaf_begin_loc; j < leaf_end_loc; ++j)
						{
							flag = true;
							for (unsigned k = 0; k < Q.repeated_vertex_set[Q.leaf[i]].size(); ++k) 
							{
								if (embedding[Q.repeated_vertex_set[Q.leaf[i]][k]][embedding[Q.repeated_vertex_set[Q.leaf[i]][k]][0]] == G.adjacency_list[j])
								{
									flag = false;
									break;
								}
							}
							if (flag)
								len++;
						}					
					}
					cnt = cnt / size * len;
				}
			}
			res_num += cnt;
			if (res_num >= limited_num)
				return;
			depth--;
			vertex = Q.match_order[depth];
			if (Q.encoding[vertex]) 
			{
				embedding[depth][0]--;
			}
			else 
			{
				len = embedding[depth][embedding[depth][0]];
				embedding[depth][0] -= (len + 1);
			}
			for (unsigned i = 0; i < Q.white_backward_neighbor[vertex].size(); ++i)
			{
				idx = Q.white_backward_neighbor[vertex][i];
				len = embedding[idx][embedding[idx][0]];
				embedding[idx][0] -= (len + 1);
			}
		}

		cur_vertex = Q.match_order[depth]; 
		for (unsigned i = 0; i < Q.children[cur_vertex].size(); ++i)
			Q.CEB_valid[Q.children[cur_vertex][i]] = false;
		if (embedding[depth][0] > 0)
		{
			depth++;
			if (depth == Q.match_order.size())
				continue;
			next_vertex = Q.match_order[depth];
			black_size = Q.black_backward_neighbor[next_vertex].size(); 
			white_size = Q.white_backward_neighbor[next_vertex].size(); 
			if ((Q.CEB_flag[next_vertex])&&(Q.CEB_valid[next_vertex]))
			{
				if (Q.CEB_iter[next_vertex] == 0) 
				{
					idx = Q.backward_neighbor_set[next_vertex].back();
					while(true)
					{
						depth--;
						if (depth > idx)
							embedding[depth][0] = 0;
						else
						{
							vertex = Q.match_order[depth];
							if (Q.encoding[vertex]) 
							{
								embedding[depth][0]--;
							}
							else 
							{
								len = embedding[depth][embedding[depth][0]];
								embedding[depth][0] -= (len + 1);
							}
							break;
						}
					}
					continue;	
				}
				if (Q.encoding[next_vertex])
				{
					embedding[depth][0] += Q.CEB_iter[next_vertex];				
					if (white_size > 0)
					{
						for (unsigned i = 0; i < white_size; ++i)
						{
							idx = Q.white_backward_neighbor[next_vertex][i];
							for (unsigned j = Q.CEB[next_vertex][i]; j < Q.CEB[next_vertex][i+1]; ++j)
							{
								embedding[idx][0]++;
								embedding[idx][embedding[idx][0]] = Q.CEB[next_vertex][j];
							}
						}	
					}
				}
				else 
				{
					if (white_size == 0)
					{
						embedding[depth][0] += Q.CEB_iter[next_vertex];	
					}
					else
					{
						unsigned i;
						for (i = 0; i < white_size; ++i)
						{
							idx = Q.white_backward_neighbor[next_vertex][i];
							for (unsigned j = Q.CEB[next_vertex][i]; j < Q.CEB[next_vertex][i+1]; ++j)
							{
								embedding[idx][0]++;
								embedding[idx][embedding[idx][0]] = Q.CEB[next_vertex][j];
							}
						}				
						for (unsigned j = Q.CEB[next_vertex][i]; j < Q.CEB[next_vertex][i+1]; ++j)
						{
							embedding[depth][0]++;
							embedding[depth][embedding[depth][0]] = Q.CEB[next_vertex][j];
						}						
					}
				}
				continue;
			}


			if (Q.encoding[next_vertex]) 
			{
				temp_size = 0;
				unordered_set<unsigned> union_candidate_set;
				if (black_size > 0) 
				{
					if (black_size == 1) 
					{
						idx = Q.black_backward_neighbor[next_vertex][0];
						v = embedding[idx][embedding[idx][0]];
						vertex = Q.match_order[idx];
						iter = qcsr.offset_list[vertex][next_vertex].find(v); 
						if (iter == qcsr.offset_list[vertex][next_vertex].end())
						{
							while(true)
							{
								depth--;
								if (depth > idx)
									embedding[depth][0] = 0;
								else
								{
									embedding[depth][0]--;
									break;
								}
							}
							continue;	
						}
						begin_loc[depth] = qcsr.offset_list[vertex][next_vertex][v].first;
						end_loc[depth] = begin_loc[depth] + qcsr.offset_list[vertex][next_vertex][v].second;
						if (white_size == 0) 
						{
							if (Q.CEB_flag[next_vertex])
							{
								Q.CEB_iter[next_vertex] = 0;
								Q.CEB_valid[next_vertex] = true;
							}
							for (unsigned i = begin_loc[depth]; i < end_loc[depth]; ++i)
							{
								flag = true;
								for (unsigned j = 0; j < Q.repeated_vertex_set[next_vertex].size(); ++j) 
								{
									if (embedding[Q.repeated_vertex_set[next_vertex][j]][embedding[Q.repeated_vertex_set[next_vertex][j]][0]] == qcsr.adjacency_list[i])
									{
										flag = false;
										break;
									}
								}
								if (flag)
								{
									embedding[depth][0]++;
									embedding[depth][embedding[depth][0]] = qcsr.adjacency_list[i];
									if (Q.CEB_flag[next_vertex])
										Q.CEB_iter[next_vertex]++;
								}
							}
						}
						else 
						{
							for (unsigned i = begin_loc[depth]; i < end_loc[depth]; ++i)
							{
								flag = true;
								for (unsigned j = 0; j < Q.repeated_vertex_set[next_vertex].size(); ++j) 
								{
									if (embedding[Q.repeated_vertex_set[next_vertex][j]][embedding[Q.repeated_vertex_set[next_vertex][j]][0]] == qcsr.adjacency_list[i])
									{
										flag = false;
										break;
									}
								}
								if (flag)
								{
									temp_candidate_set[temp_size] = qcsr.adjacency_list[i];
									temp_size++;
								}
							}
						}
					}
					else 
					{
						min_len = 100000;
						total_len = 0;
						for (unsigned i = 0; i < Q.black_backward_neighbor[next_vertex].size(); ++i)
						{
							idx = Q.black_backward_neighbor[next_vertex][i];
							v = embedding[idx][embedding[idx][0]];
							vertex = Q.match_order[idx];
							iter = qcsr.offset_list[vertex][next_vertex].find(v); 
							if (iter == qcsr.offset_list[vertex][next_vertex].end())
							{
								min_len = 0;
								break;
							}
							total_len += qcsr.offset_list[vertex][next_vertex][v].second;
							if (qcsr.offset_list[vertex][next_vertex][v].second < min_len)
							{
								min_len = qcsr.offset_list[vertex][next_vertex][v].second;
								min_idx = i;
							}
							temp_begin_loc[i] = qcsr.offset_list[vertex][next_vertex][v].first;
							temp_end_loc[i] = temp_begin_loc[i] + qcsr.offset_list[vertex][next_vertex][v].second;
							temp_address[i] = qcsr.adjacency_list.data();
						}
						if (min_len == 0)
						{
							while(true)
							{
								depth--;
								if (depth > idx)
									embedding[depth][0] = 0;
								else
								{
									embedding[depth][0]--;
									break;
								}
							}
							continue;								
						}
						if (total_len <= (min_len << 3)*black_size)
							ret = set_intersection(Q, depth, black_size, temp_address, temp_begin_loc, temp_end_loc, embedding, temp_candidate_set, temp_size, true);
						else
							ret = edge_check(G, Q, depth, black_size, temp_address, temp_begin_loc, temp_end_loc, embedding, temp_candidate_set, temp_size, min_idx, true);
						if (!ret)
						{
							while(true)
							{
								depth--;
								if (depth > idx)
									embedding[depth][0] = 0;
								else
								{
									embedding[depth][0]--;
									break;
								}
							}
							continue;								
						}
					}
				}
				else 
				{
					min_len = 100000;
					for (unsigned i = 0; i < white_size; ++i)
					{
						idx = Q.white_backward_neighbor[next_vertex][i];
						temp_iter = embedding[idx][0];
						len = embedding[idx][temp_iter];
						if (len < min_len)
						{
							min_len = len;
							min_idx = idx;
						}						
					}
					temp_iter = embedding[min_idx][0];
					vertex = Q.match_order[min_idx];
					for (unsigned i = temp_iter - min_len; i < temp_iter; ++i)
					{
						v = embedding[min_idx][i];
						iter = qcsr.offset_list[vertex][next_vertex].find(v); 
						if (iter == qcsr.offset_list[vertex][next_vertex].end())
							continue;
						begin_loc[depth] = qcsr.offset_list[vertex][next_vertex][v].first;
						end_loc[depth] = begin_loc[depth] + qcsr.offset_list[vertex][next_vertex][v].second;
						for (unsigned i = begin_loc[depth]; i < end_loc[depth]; ++i)
							union_candidate_set.insert(qcsr.adjacency_list[i]);		
					}
					if (union_candidate_set.size() == 0)
					{
						while(true)
						{
							depth--;
							if (depth > min_idx)
								embedding[depth][0] = 0;
							else
							{
								embedding[depth][0] -= (min_len + 1);
								break;
							}
						}
						continue;								
					}
				}

				if (black_size > 0)
				{
					if (temp_size ==  0)
						continue;
					if (Q.CEB_flag[next_vertex])
					{
						Q.CEB_iter[next_vertex] = 0;
						Q.CEB_valid[next_vertex] = true;
						for (unsigned j = 0; j < white_size; ++j)
						{
							idx = Q.white_backward_neighbor[next_vertex][j];
							iter_before_split[idx] = embedding[idx][0];
						}
					}
					for (unsigned i = 0; i < temp_size; ++i)
					{
						v = temp_candidate_set[i];
						flag = true;
						unsigned j;
						for (j = 0; j < white_size; ++j)
						{
							idx = Q.white_backward_neighbor[next_vertex][j];
							temp_iter = embedding[idx][0];
							len = embedding[idx][temp_iter];	
							vertex = Q.match_order[idx];			
							new_len = 0;
							iter = qcsr.offset_list[next_vertex][vertex].find(v);
							if (iter == qcsr.offset_list[next_vertex][vertex].end())
							{
								flag = false;
								break;
							}
							unsigned left1 = temp_iter - len;
							unsigned right1 = temp_iter;
							unsigned left2 = qcsr.offset_list[next_vertex][vertex][v].first;
							unsigned right2 = left2 + qcsr.offset_list[next_vertex][vertex][v].second; 
							unsigned x, y;
							while (true)
							{
								if (left1 == right1)
									break;
								if (left2 == right2)
								 	break;
								x = embedding[idx][left1];
								y = qcsr.adjacency_list[left2]; 	
								if (x < y)
									left1++;
								else if (x > y)
									left2++;		
								else
								{
									new_len++;
									embedding[idx][temp_iter+new_len] = x;
									left1++;
									left2++;
								}												
							}	
							if (new_len == 0)
							{
								flag = false;
								break;	
							}
							else
							{
								embedding[idx][0] += (new_len+1);
								embedding[idx][embedding[idx][0]] = new_len;
							}	
						}
						if (flag)
						{
							embedding[depth][0]++;
							embedding[depth][embedding[depth][0]] = v;
							if (Q.CEB_flag[next_vertex])
								Q.CEB_iter[next_vertex]++;
						}
						else
						{
							for (unsigned k = 0; k < j; ++k)
							{
								idx = Q.white_backward_neighbor[next_vertex][k];
								len = embedding[idx][embedding[idx][0]];
								embedding[idx][0] -= (len + 1);
							}
						}
					}
					if ((Q.CEB_flag[next_vertex])&&(Q.CEB_iter[next_vertex] != 0))
					{
						Q.CEB[next_vertex][0] = white_size+1;
						for (unsigned j = 0; j < white_size; ++j)
						{
							idx = Q.white_backward_neighbor[next_vertex][j];
							len = embedding[idx][0]-iter_before_split[idx];
							Q.CEB[next_vertex][j+1] = Q.CEB[next_vertex][j]+len;
							for (unsigned k = 0; k < len; ++k)
								Q.CEB[next_vertex][Q.CEB[next_vertex][j]+k] = embedding[idx][iter_before_split[idx]+1+k];
						}
					}
				}
				else
				{
					if (Q.CEB_flag[next_vertex])
					{
						Q.CEB_iter[next_vertex] = 0;
						Q.CEB_valid[next_vertex] = true;
						for (unsigned j = 0; j < white_size; ++j)
						{
							idx = Q.white_backward_neighbor[next_vertex][j];
							iter_before_split[idx] = embedding[idx][0];
						}
					}
					for (unordered_set<unsigned>::iterator i = union_candidate_set.begin(); i != union_candidate_set.end(); ++i)
					{
						v = *i;
						flag = true;
						for (unsigned j = 0; j < Q.repeated_vertex_set[next_vertex].size(); ++j) 
						{
							if (embedding[Q.repeated_vertex_set[next_vertex][j]][embedding[Q.repeated_vertex_set[next_vertex][j]][0]] == v)
							{
								flag = false;
								break;
							}
						}
						if (!flag)
							continue;
						unsigned j;
						for (j = 0; j < white_size; ++j)
						{
							idx = Q.white_backward_neighbor[next_vertex][j];
							temp_iter = embedding[idx][0];
							len = embedding[idx][temp_iter];	
							vertex = Q.match_order[idx];			
							new_len = 0;
							iter = qcsr.offset_list[next_vertex][vertex].find(v);
							if (iter == qcsr.offset_list[next_vertex][vertex].end())
							{
								flag = false;
								break;
							}
							unsigned left1 = temp_iter - len;
							unsigned right1 = temp_iter;
							unsigned left2 = qcsr.offset_list[next_vertex][vertex][v].first;
							unsigned right2 = left2 + qcsr.offset_list[next_vertex][vertex][v].second; 
							unsigned x, y;
							while (true)
							{
								if (left1 == right1)
									break;
								if (left2 == right2)
								 	break;
								x = embedding[idx][left1];
								y = qcsr.adjacency_list[left2]; 	
								if (x < y)
									left1++;
								else if (x > y)
									left2++;		
								else
								{
									new_len++;
									embedding[idx][temp_iter+new_len] = x;
									left1++;
									left2++;
								}												
							}	
							if (new_len == 0)
							{
								flag = false;
								break;	
							}
							else
							{
								embedding[idx][0] += (new_len+1);
								embedding[idx][embedding[idx][0]] = new_len;
							}	
						}
						if (flag)
						{
							embedding[depth][0]++;
							embedding[depth][embedding[depth][0]] = v;
							if (Q.CEB_flag[next_vertex])
								Q.CEB_iter[next_vertex]++;
						}
						else
						{
							for (unsigned k = 0; k < j; ++k)
							{
								idx = Q.white_backward_neighbor[next_vertex][k];
								len = embedding[idx][embedding[idx][0]];
								embedding[idx][0] -= (len + 1);
							}
						}
					}		
					if ((Q.CEB_flag[next_vertex])&&(Q.CEB_iter[next_vertex] != 0))
					{
						Q.CEB[next_vertex][0] = white_size+1;
						for (unsigned j = 0; j < white_size; ++j)
						{
							idx = Q.white_backward_neighbor[next_vertex][j];
							len = embedding[idx][0]-iter_before_split[idx];
							Q.CEB[next_vertex][j+1] = Q.CEB[next_vertex][j]+len;
							for (unsigned k = 0; k < len; ++k)
								Q.CEB[next_vertex][Q.CEB[next_vertex][j]+k] = embedding[idx][iter_before_split[idx]+1+k];
						}
					}			
				}
			}
			else 
			{
				black_size = Q.black_backward_neighbor[next_vertex].size(); 
				white_size = Q.white_backward_neighbor[next_vertex].size(); 
				temp_size = 0;
				if (black_size > 0)
				{
					if (black_size == 1) 
					{
						idx = Q.black_backward_neighbor[next_vertex][0];
						v = embedding[idx][embedding[idx][0]];
						vertex = Q.match_order[idx];
						iter = qcsr.offset_list[vertex][next_vertex].find(v); 
						if (iter == qcsr.offset_list[vertex][next_vertex].end())
						{
							while(true)
							{
								depth--;
								if (depth > idx)
									embedding[depth][0] = 0;
								else
								{
									embedding[depth][0]--;
									break;
								}
							}
							continue;	
						}
						begin_loc[depth] = qcsr.offset_list[vertex][next_vertex][v].first;
						end_loc[depth] = begin_loc[depth] + qcsr.offset_list[vertex][next_vertex][v].second;
						if (white_size == 0) 
						{
							if (Q.CEB_flag[next_vertex])
							{
								Q.CEB_iter[next_vertex] = 0;
								Q.CEB_valid[next_vertex] = true;
							}
							for (unsigned i = begin_loc[depth]; i < end_loc[depth]; ++i)
							{
								flag = true;
								for (unsigned j = 0; j < Q.repeated_vertex_set[next_vertex].size(); ++j) 
								{
									if (embedding[Q.repeated_vertex_set[next_vertex][j]][embedding[Q.repeated_vertex_set[next_vertex][j]][0]] == qcsr.adjacency_list[i])
									{
										flag = false;
										break;
									}
								}
								if (flag)
								{
									embedding[depth][0]++;
									embedding[depth][embedding[depth][0]] = qcsr.adjacency_list[i];
									if (Q.CEB_flag[next_vertex])
										Q.CEB_iter[next_vertex]++;
								}
							}
							embedding[depth][0]++;
							embedding[depth][embedding[depth][0]] = qcsr.offset_list[vertex][next_vertex][v].second;
							if (Q.CEB_flag[next_vertex])
								Q.CEB_iter[next_vertex]++;
						}
						else
						{
							for (unsigned i = begin_loc[depth]; i < end_loc[depth]; ++i)
							{
								flag = true;
								for (unsigned j = 0; j < Q.repeated_vertex_set[next_vertex].size(); ++j) 
								{
									if (embedding[Q.repeated_vertex_set[next_vertex][j]][embedding[Q.repeated_vertex_set[next_vertex][j]][0]] == qcsr.adjacency_list[i])
									{
										flag = false;
										break;
									}
								}
								if (flag)
								{
									temp_candidate_set[temp_size] = qcsr.adjacency_list[i];
									temp_size++;
								}
							}
						}
					}
					else 
					{
						min_len = 100000;
						total_len = 0;
						for (unsigned i = 0; i < black_size; ++i)
						{
							idx = Q.black_backward_neighbor[next_vertex][i];
							v = embedding[idx][embedding[idx][0]];
							vertex = Q.match_order[idx];
							iter = qcsr.offset_list[vertex][next_vertex].find(v); 
							if (iter == qcsr.offset_list[vertex][next_vertex].end())
							{
								min_len = 0;
								break;
							}
							total_len += qcsr.offset_list[vertex][next_vertex][v].second;
							if (qcsr.offset_list[vertex][next_vertex][v].second < min_len)
							{
								min_len = qcsr.offset_list[vertex][next_vertex][v].second;
								min_idx = i;
							}
							temp_begin_loc[i] = qcsr.offset_list[vertex][next_vertex][v].first;
							temp_end_loc[i] = temp_begin_loc[i] + qcsr.offset_list[vertex][next_vertex][v].second;
							temp_address[i] = qcsr.adjacency_list.data();
						}
						if (min_len == 0)
						{
							while(true)
							{
								depth--;
								if (depth > idx)
									embedding[depth][0] = 0;
								else
								{
									embedding[depth][0]--;
									break;
								}
							}
							continue;								
						}
						if (total_len <= (min_len << 3)*black_size)
							ret = set_intersection(Q, depth, black_size, temp_address, temp_begin_loc, temp_end_loc, embedding, temp_candidate_set, temp_size, true);
						else
							ret = edge_check(G, Q, depth, black_size, temp_address, temp_begin_loc, temp_end_loc, embedding, temp_candidate_set, temp_size, min_idx, true);
						if (!ret)
						{
							while(true)
							{
								depth--;
								if (depth > idx)
									embedding[depth][0] = 0;
								else
								{
									embedding[depth][0]--;
									break;
								}
							}
							continue;								
						}
					}
				}
				if (white_size > 0) 
				{
					if (Q.CEB_flag[next_vertex])
					{
						Q.CEB_iter[next_vertex] = 0;
						Q.CEB_valid[next_vertex] = true;
						for (unsigned j = 0; j < white_size; ++j)
						{
							idx = Q.white_backward_neighbor[next_vertex][j];
							iter_before_split[idx] = embedding[idx][0];
						}
						iter_before_split[depth] = embedding[depth][0];
					}
					int level = 0;
					for (unsigned i = 0; i < white_size; ++i)
					{
						idx = Q.white_backward_neighbor[next_vertex][i];
						temp_iter = embedding[idx][0];
						white_len[i] = embedding[idx][temp_iter];
						white_end_loc[i] = temp_iter;	
					}			
					white_begin_loc[0] = white_end_loc[0] - white_len[0];
					while (level != -1) 
					{
						if (white_begin_loc[level] < white_end_loc[level])
						{
							idx = Q.white_backward_neighbor[next_vertex][level];
							white_mapping[idx] = embedding[idx][white_begin_loc[level]]; 
							white_begin_loc[level]++;
							level++;
							if (level == white_size)
							{
								min_len = 100000;
								total_len = 0;
								unsigned i;
								for (i = 0; i < white_size; ++i)
								{
									idx = Q.white_backward_neighbor[next_vertex][i];
									v = white_mapping[idx];
									vertex = Q.match_order[idx];
									iter = qcsr.offset_list[vertex][next_vertex].find(v);
									if (iter == qcsr.offset_list[vertex][next_vertex].end())
									{
										min_len = 0;
										break;
									}
									total_len += qcsr.offset_list[vertex][next_vertex][v].second;
									if (qcsr.offset_list[vertex][next_vertex][v].second < min_len)
									{
										min_len = qcsr.offset_list[vertex][next_vertex][v].second;
										min_idx = i;
									}
									temp_begin_loc[i] = qcsr.offset_list[vertex][next_vertex][v].first;
									temp_end_loc[i] = temp_begin_loc[i] + qcsr.offset_list[vertex][next_vertex][v].second;
									temp_address[i] = qcsr.adjacency_list.data();

								}
								if (min_len == 0)
								{
									while(true)
									{
										level--;
										if (level == i)
											break;
									}
									continue;								
								}
								if (temp_size > 0)
								{
									total_len += temp_size;
									if (temp_size < min_len)
									{
										min_len = temp_size;
										min_idx = i;
									}
									temp_begin_loc[i] = 0;
									temp_end_loc[i] = temp_size;
									temp_address[i] = temp_candidate_set;
									i++;
								}
								ret = set_intersection(Q, depth, i, temp_address, temp_begin_loc, temp_end_loc, embedding, temp_candidate_set, temp_size, false);
								if (ret)
								{
									for (unsigned i = 0; i < white_size; ++i)
									{
										idx = Q.white_backward_neighbor[next_vertex][i];
										embedding[idx][0]++;
										embedding[idx][embedding[idx][0]] = white_mapping[idx];
										embedding[idx][0]++;
										embedding[idx][embedding[idx][0]] = 1;
									}
									if (Q.CEB_flag[next_vertex])
										Q.CEB_iter[next_vertex]++;
								}
								level--;
								continue;
							}
							white_begin_loc[level] = white_end_loc[level] - white_len[level];							
						}
						else
						{
							level--;
						}
					}
					if ((Q.CEB_flag[next_vertex])&&(Q.CEB_iter[next_vertex] != 0))
					{
						Q.CEB[next_vertex][0] = white_size+2;
						for (unsigned j = 0; j < white_size; ++j)
						{
							idx = Q.white_backward_neighbor[next_vertex][j];
							len = embedding[idx][0]-iter_before_split[idx];
							Q.CEB[next_vertex][j+1] = Q.CEB[next_vertex][j]+len;
							for (unsigned k = 0; k < len; ++k)
								Q.CEB[next_vertex][Q.CEB[next_vertex][j]+k] = embedding[idx][iter_before_split[idx]+1+k];
						}
						len = embedding[depth][0]-iter_before_split[depth];
						Q.CEB[next_vertex][white_size+1] = Q.CEB[next_vertex][white_size]+len;
						for (unsigned k = 0; k < len; ++k)
							Q.CEB[next_vertex][Q.CEB[next_vertex][white_size]+k] = embedding[depth][iter_before_split[depth]+1+k];
					}	
				}
			}
		}
		else
		{
			depth--;	
			vertex = Q.match_order[depth];
			if (Q.encoding[vertex]) 
			{
				embedding[depth][0]--;
			}
			else 
			{
				len = embedding[depth][embedding[depth][0]];
				embedding[depth][0] -= (len + 1);
			}
			for (unsigned i = 0; i < Q.white_backward_neighbor[vertex].size(); ++i)
			{
				idx = Q.white_backward_neighbor[vertex][i];
				len = embedding[idx][embedding[idx][0]];
				embedding[idx][0] -= (len + 1);
			}
		}
	}

}

int main(int argc, char* argv[])
{
	string data_filename = "";
	string query_filename = ""; 
	string ECS_filename = "";
	unsigned limited_output_num = 100000;

	if (argc <= 2)
	{
		cout << endl;
		cout << "Usage:\t./match -d [data_graph] -q [query_graph] -t [limited_output_num]" << endl;
		cout << endl;
		cout << "Options:" << endl;
		cout << "\t-h,\t\t the help message." << endl;
		cout << "\t-d,\t\t the filename of the data graph (cannot be empty)." << endl;
		cout << "\t-q,\t\t the filename of the query graph (cannot be empty)." << endl;
		cout << "\t-t,\t\t the maximum number of the output results (can be omitted and default 100000)." << endl;
		cout << endl;
		return 0;
	}
	else
	{
		data_filename = getArgValue(argc, argv, "-d", "");
		query_filename = getArgValue(argc, argv, "-q", "");
		ECS_filename = data_filename + "-ECS";
		limited_output_num = atoi(getArgValue(argc, argv, "-t", "100000").c_str());
	}	

    Graph data_graph;
	gettimeofday(&begin_tv, NULL);
	data_graph.read(data_filename);
#ifdef hybrid_opt
		data_graph.setBloomFilter();
#endif
    //data_graph.count_edge();
	gettimeofday(&end_tv, NULL);
	double read_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
    cout << "data_graph = " << data_filename << endl;
    cout << "label_num = " << data_graph.label_num << endl;
    cout << "vertex_num = " << data_graph.vertex_num << endl;
    cout << "edge_num = " << data_graph.edge_num << endl;
    cout << "max_degree = " << data_graph.max_degree << endl;
    cout << "average_degree = " << data_graph.average_degree << endl;
	cout << "read_time = " << read_time << endl;

	Query query_graph;
	gettimeofday(&begin_tv, NULL);
	query_graph.read(query_filename);
    query_graph.CFL_decomposition();
	gettimeofday(&end_tv, NULL);
	read_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
    cout << "query_graph = " << query_filename << endl;
    cout << "vertex_num = " << query_graph.vertex_num << endl;
    cout << "edge_num = " << query_graph.edge_num << endl;
	cout << "read_time = " << read_time << endl;	

	gettimeofday(&begin_tv, NULL);
	getStartVertex(data_graph, query_graph);
	gettimeofday(&end_tv, NULL);
	double get_start_vertex_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
	cout << "start_vertex = " << query_graph.start_vertex << endl;
	cout << "get_start_vertex_time = " << get_start_vertex_time << endl;

	QCSR qcsr(data_graph, query_graph);
	gettimeofday(&begin_tv, NULL);
	qcsr.build(data_graph, query_graph);
	gettimeofday(&end_tv, NULL);
	double build_qcsr_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
	cout << "build_qcsr_time = " << build_qcsr_time << endl;

	unsigned** embedding = new unsigned*[query_graph.vertex_num];
	for (unsigned i = 0; i < query_graph.match_order.size(); ++i)
	{
		if (query_graph.encoding[query_graph.match_order[i]])
		{
			embedding[i] = new unsigned[10000];
			embedding[i][0] = 0;
		}
		else
		{
			embedding[i] = new unsigned[1000000];
			embedding[i][0] = 0;			
		}
		if (query_graph.CEB_flag[query_graph.match_order[i]])
		{
			query_graph.CEB[query_graph.match_order[i]] = new unsigned[100000];
		}
	}
	unsigned res_num = 0;
	gettimeofday(&begin_tv, NULL);
	enumerate(data_graph, query_graph, qcsr, embedding, res_num, limited_output_num);
	gettimeofday(&end_tv, NULL);
	double match_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
	cout << "match_time = " << match_time << endl;
	cout << "embeddings = " << res_num << endl;
	double total_time = read_time + get_start_vertex_time + build_qcsr_time + match_time;
	cout << "total_time = " << total_time << endl;

	return 0;
}

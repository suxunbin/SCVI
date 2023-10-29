#include <iostream>
#include <set>
#include <map>
#include <unordered_map>
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

#define CEB_opt
#define order_opt
//#define hybrid_opt

using namespace std;

struct timeval begin_tv, end_tv;
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

void getStartVertex(Graph& G, Query& Q) // get the start vertex
{
	unsigned v;
	unsigned label;
	unsigned neighbor;
	unsigned nlabel;
	unsigned min_size;
	unsigned min_nlabel;
	double min_score= 1000000000.0;

	if (Q.query_state == 2)
	{
    	for (unsigned i = 0; i < Q.tree.size(); ++i)
    	{
			v = Q.tree[i];
			label = Q.label_list[v];
			min_size = 1000000000;
			for (unsigned j = 0; j < Q.neighbor_list[v].size(); ++j)
			{
				neighbor = Q.neighbor_list[v][j];
				nlabel = Q.label_list[neighbor];
				if (G.partitioned_vertex_list[nlabel][label].size() < min_size)
				{
					min_size = G.partitioned_vertex_list[nlabel][label].size();
					min_nlabel = nlabel; 
				}
			}
			Q.score[v] = (double)min_size / (double)Q.neighbor_list[v].size();
			if (Q.score[v] < min_score)
			{
				min_score = Q.score[v];
				Q.start_vertex = v;
				Q.start_vertex_nlabel = min_nlabel;
			}
    	}
	}
	else
	{
    	for (unsigned i = 0; i < Q.core.size(); ++i)
    	{
			v = Q.core[i];
			label = Q.label_list[v];
			min_size = 1000000000;
			for (unsigned j = 0; j < Q.neighbor_list[v].size(); ++j)
			{
				neighbor = Q.neighbor_list[v][j];
				nlabel = Q.label_list[neighbor];
				if (G.partitioned_vertex_list[nlabel][label].size() < min_size)
				{
					min_size = G.partitioned_vertex_list[nlabel][label].size();
					min_nlabel = nlabel; 
				}
			}
			Q.score[v] = (double)min_size / (double)Q.neighbor_list[v].size();
			if (Q.score[v] < min_score)
			{
				min_score = Q.score[v];
				Q.start_vertex = v;
				Q.start_vertex_nlabel = min_nlabel;
			}
    	}
	}
}

void edge_check(Graph& G, Query& Q, QCSR& qcsr, unsigned& vertex, unsigned* &embedding, unsigned* &begin, unsigned* &end, unsigned* &id, unsigned &min_idx)
{
	unsigned x, x1, x2;
	unsigned idx;
	uint64_t edge;
	unsigned mask;
	bool flag;
	unsigned cnt = 0;
	for (unsigned i = 0; i < Q.backward_neighbor[vertex].size(); ++i)
	{
		if (i == min_idx)
			continue;
		idx = Q.backward_neighbor[vertex][i];
		id[cnt++] = embedding[idx];
	}
	unsigned h1[cnt];
	unsigned h2[cnt];
	for (unsigned i = 0; i < cnt; ++i)
	{
		h1[i] = Graph::BKDR_hash(id[i], G.range);
		h2[i] = Graph::AP_hash(id[i], G.range);
	}
	for (unsigned i = begin[min_idx]; i < end[min_idx]; ++i)
	{
		x = qcsr.adjacency_list[i];
		x1 = Graph::BKDR_hash(x, G.range);
		x2 = Graph::AP_hash(x, G.range);
		flag = true;	
		for (unsigned j = 0; j < cnt; ++j)
		{
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
			qcsr.candidate_vertex_set[vertex][qcsr.candidate_size[vertex]] = x;
			qcsr.candidate_size[vertex]++;
		}
	}
}

void set_intersection(Query& Q, QCSR& qcsr, unsigned& vertex, unsigned* &embedding, unsigned* &begin, unsigned* &end)
{ 
	unsigned cnt = Q.backward_neighbor[vertex].size();
	unsigned cmn = 1;
	bool bflag;
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
		x = qcsr.adjacency_list[begin[cur]];
		y = qcsr.adjacency_list[begin[pre]]; 
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

				qcsr.candidate_vertex_set[vertex][qcsr.candidate_size[vertex]] = x;
				qcsr.candidate_size[vertex]++;
			}		
		}
	}
}

void core_match(Graph& G, Query& Q, QCSR& qcsr, unsigned* &res, unsigned& res_iter, unsigned& res_num, unsigned& limited_num) // core-match
{
	int depth = 0;
	unsigned* begin_loc;
	unsigned* end_loc;
	bool* change;
	unsigned* temp_begin_loc = new unsigned[Q.max_degree];
	unsigned* temp_end_loc = new unsigned[Q.max_degree];
	unsigned* temp_idx = new unsigned[Q.max_degree];
	unsigned v;
	unsigned vertex;
	unsigned next_vertex;
	unsigned temp_vertex;
	unsigned idx;
	unsigned id;
	unsigned label;
	bool flag;
	unsigned CEB_expand_flag;
	begin_loc = new unsigned[Q.vertex_num];
	end_loc = new unsigned[Q.vertex_num];
	change = new bool[Q.vertex_num];
	unsigned* embedding = new unsigned[Q.vertex_num];
	unsigned min_len;
	unsigned total_len;
	unsigned min_idx; 

	begin_loc[0] = 0;
	end_loc[0] = qcsr.candidate_size[Q.start_vertex];
	while (depth != -1) //DFS match
	{
		if (depth == 0)
		{
			if (begin_loc[depth] < end_loc[depth])
			{
				v = qcsr.candidate_vertex_set[Q.start_vertex][begin_loc[depth]];
				embedding[depth] = v;
				if (Q.repeat_state[Q.start_vertex])
					Q.repeat[v] = 0;
				begin_loc[depth]++;
				depth++;
				next_vertex = Q.core_match_order[depth];
				id = v*Q.vertex_num*2+next_vertex*2;
				begin_loc[depth] = qcsr.offset_list[id];
				end_loc[depth] = begin_loc[depth] + qcsr.offset_list[id+1];
			}
			else
				depth--;	
		}
		else if ((depth > 0) && (depth < Q.core.size()))
		{
			vertex = Q.core_match_order[depth];
			if (begin_loc[depth] < end_loc[depth])
			{
				if ((Q.CEB_update[vertex])&&(Q.CEB_width[vertex] > 1))
				{
					if (CEB_expand_flag < 32)
					{
						if (embedding[CEB_expand_flag] == Q.CEB[vertex][begin_loc[depth]+CEB_expand_flag-depth])
						{
							begin_loc[depth] += Q.CEB_width[vertex];
							continue;
						}
						else
						{
							unsigned i = 0;
							while (i < Q.CEB_width[vertex])
							{
								embedding[depth+i] = Q.CEB[vertex][begin_loc[depth]];	
								begin_loc[depth]++;	
								if (Q.repeat_state[Q.core_match_order[depth+i]])
									Q.repeat[embedding[depth+i]] = depth+i;
								++i;						
							}		
							depth += Q.CEB_width[vertex];	
							continue;							
						}
					}
					unsigned i = 0;
					while (i < Q.CEB_width[vertex])
					{
						embedding[depth+i] = Q.CEB[vertex][begin_loc[depth]];	
						begin_loc[depth]++;	
						if (Q.repeat_state[Q.core_match_order[depth+i]])
							Q.repeat[embedding[depth+i]] = depth+i;
						++i;			
					}
					for (i = 0; i < Q.CEB_width[vertex]; ++i)
					{
						if (begin_loc[depth] == end_loc[depth])
							change[depth+i] = true;
						else
						{
							if (embedding[depth+i] == Q.CEB[vertex][begin_loc[depth]+i])
								change[depth+i] = false;
							else
							{
								change[depth+i] = true;
								for (unsigned j = i+1; j < Q.CEB_width[vertex]; ++j)
									change[depth+j] = true;
								break;
							}
						}
					}	
					depth += Q.CEB_width[vertex];					
				}
				else
				{
					if (Q.backward_neighbor[vertex].size() == 1)
						v = qcsr.adjacency_list[begin_loc[depth]];
					else
						v = qcsr.candidate_vertex_set[vertex][begin_loc[depth]];

					if (Q.repeat_state[vertex])
					{
						if (Q.repeat[v] < 32)
						{
							begin_loc[depth]++;
							continue;
						}
						Q.repeat[v] = depth;
					}

					embedding[depth] = v;
					begin_loc[depth]++;			
					depth++;
#ifdef CEB_opt
					temp_vertex = Q.CEB_write[vertex];
					if (temp_vertex < 32)
					{
						for (unsigned i = depth-Q.CEB_width[temp_vertex]; i < depth; ++i)
							Q.CEB[temp_vertex][Q.CEB_iter[temp_vertex]++] = embedding[i];
					}	
					else
					{
						if (Q.CEB_width[vertex] == 1)
							Q.CEB_iter[vertex]++;
					}
#endif
				}
				if (depth == Q.core.size())
				{
					for (unsigned i = 0; i < depth; ++i)
						res[res_iter++] = embedding[i];	
					res_num++;
					if (res_num >= limited_num)
						return;
					depth--;
					vertex = Q.core_match_order[depth];
					temp_vertex = Q.CEB_write[vertex];
					if (temp_vertex < 32)
					{
						if (Q.CEB_update[temp_vertex])
						{
							for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
							{
								vertex = Q.core_match_order[depth-i];
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth-i]] = 32;										
							}
							depth -= (Q.CEB_width[temp_vertex]-1);
							CEB_expand_flag = 32;
							continue;							
						}
					}
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;
					continue;	
				}
				next_vertex = Q.core_match_order[depth];
				if (Q.backward_neighbor[next_vertex].size() == 1)
				{
					if (Q.CEB_update[next_vertex])
					{
						if (Q.CEB_iter[next_vertex] == 0)
						{
							while(depth > Q.CEB_father_idx[next_vertex])
							{
								depth--;
								vertex = Q.core_match_order[depth];
								temp_vertex = Q.CEB_write[vertex];
								if (temp_vertex < 32)
								{
									if (Q.CEB_update[temp_vertex])
									{
										for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
										{
											vertex = Q.core_match_order[depth-i];
											if (Q.repeat_state[vertex])
												Q.repeat[embedding[depth-i]] = 32;		
											for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
												Q.CEB_update[Q.parent[vertex][j]] = false;								
										}
										depth -= (Q.CEB_width[temp_vertex]-1);						
										continue;							
									}
								}
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth]] = 32;
								for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
									Q.CEB_update[Q.parent[vertex][i]] = false;
							}
							CEB_expand_flag = Q.CEB_father_idx[next_vertex];
							continue;							
						}
						if (Q.CEB_width[next_vertex] > 1)
						{
							begin_loc[depth] = 0;		
							end_loc[depth] = Q.CEB_iter[next_vertex];
							CEB_expand_flag = 32;								
						}
						else
						{
							idx = Q.backward_neighbor[next_vertex][0];
							v = embedding[idx];
							id = v*Q.vertex_num*2+next_vertex*2;	
							begin_loc[depth] = qcsr.offset_list[id];
							end_loc[depth] = begin_loc[depth] + qcsr.offset_list[id+1];							
						}		
						continue;				
					}
					idx = Q.backward_neighbor[next_vertex][0];
					v = embedding[idx];
					id = v*Q.vertex_num*2+next_vertex*2;
					if (qcsr.offset_list[id+1] == 0)
					{
						while(depth > idx)
						{
							depth--;
							vertex = Q.core_match_order[depth];
							temp_vertex = Q.CEB_write[vertex];
							if (temp_vertex < 32)
							{
								if (Q.CEB_update[temp_vertex])
								{
									for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
									{
										vertex = Q.core_match_order[depth-i];
										if (Q.repeat_state[vertex])
											Q.repeat[embedding[depth-i]] = 32;		
										for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
											Q.CEB_update[Q.parent[vertex][j]] = false;								
									}
									depth -= (Q.CEB_width[temp_vertex]-1);						
									continue;							
								}
							}
							if (Q.repeat_state[vertex])
								Q.repeat[embedding[depth]] = 32;
							for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
								Q.CEB_update[Q.parent[vertex][i]] = false;	
						}
						CEB_expand_flag = idx;
						continue;
					}
					begin_loc[depth] = qcsr.offset_list[id];
					end_loc[depth] = begin_loc[depth] + qcsr.offset_list[id+1];	
					if (Q.CEB_flag[next_vertex])
						Q.CEB_iter[next_vertex] = 0;						
				}
				else
				{
					if (Q.CEB_update[next_vertex])
					{
						if (Q.CEB_iter[next_vertex] == 0)
						{
							while(depth > Q.CEB_father_idx[next_vertex])
							{
								depth--;
								vertex = Q.core_match_order[depth];
								temp_vertex = Q.CEB_write[vertex];
								if (temp_vertex < 32)
								{
									if (Q.CEB_update[temp_vertex])
									{
										for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
										{
											vertex = Q.core_match_order[depth-i];
											if (Q.repeat_state[vertex])
												Q.repeat[embedding[depth-i]] = 32;		
											for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
												Q.CEB_update[Q.parent[vertex][j]] = false;								
										}
										depth -= (Q.CEB_width[temp_vertex]-1);						
										continue;							
									}
								}
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth]] = 32;
								for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
									Q.CEB_update[Q.parent[vertex][i]] = false;
							}
							CEB_expand_flag = Q.CEB_father_idx[next_vertex];
							continue;							
						}
						if (Q.CEB_width[next_vertex] > 1)
						{
							begin_loc[depth] = 0;		
							end_loc[depth] = Q.CEB_iter[next_vertex];	
							CEB_expand_flag = 32;
						}	
						else	
							begin_loc[depth] = 0;		
						continue;				
					}
					if (Q.NEC[next_vertex] != 0)
					{
						if (Q.NEC[vertex] != Q.NEC[next_vertex])
						{
							min_len = 100000;
							total_len = 0;
							for (unsigned i = 0; i < Q.backward_neighbor[next_vertex].size(); ++i)
							{
								idx = Q.backward_neighbor[next_vertex][i];
								v = embedding[idx];
								id = v*Q.vertex_num*2+next_vertex*2;
								if (qcsr.offset_list[id+1] <= 1)
								{
									min_len = 0;
									break;
								}
								else
								{
									total_len += qcsr.offset_list[id+1];
									if (qcsr.offset_list[id+1] < min_len)
									{
										min_len = qcsr.offset_list[id+1];
										min_idx = i;
									}
									temp_begin_loc[i] = qcsr.offset_list[id];
									temp_end_loc[i] = temp_begin_loc[i] + qcsr.offset_list[id+1];	
								}					
							}
							qcsr.candidate_size[next_vertex] = 0;
							if (min_len > 0)
							{
								if (total_len <= (min_len << 3)*(Q.backward_neighbor[next_vertex].size()))
									set_intersection(Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc);
								else
#ifdef hybrid_opt
									edge_check(G, Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc, temp_idx, min_idx);
#else
									set_intersection(Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc);
#endif
							}
							if (qcsr.candidate_size[next_vertex] <= 1)
							{
								while(depth > idx)
								{
									depth--;
									vertex = Q.core_match_order[depth];
									temp_vertex = Q.CEB_write[vertex];
									if (temp_vertex < 32)
									{
										if (Q.CEB_update[temp_vertex])
										{
											for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
											{
												vertex = Q.core_match_order[depth-i];
												if (Q.repeat_state[vertex])
													Q.repeat[embedding[depth-i]] = 32;		
												for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
													Q.CEB_update[Q.parent[vertex][j]] = false;								
											}
											depth -= (Q.CEB_width[temp_vertex]-1);						
											continue;							
										}
									}
									if (Q.repeat_state[vertex])
										Q.repeat[embedding[depth]] = 32;
									for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
										Q.CEB_update[Q.parent[vertex][i]] = false;	
								}
								CEB_expand_flag = idx;
								continue;
							}
							begin_loc[depth] = 0;
							end_loc[depth] = qcsr.candidate_size[next_vertex];
							if (Q.CEB_flag[next_vertex])
								Q.CEB_iter[next_vertex] = 0;					
							continue;
						}
						else
						{
							begin_loc[depth] = 0;
							end_loc[depth] = end_loc[depth-1];
							for (unsigned i = 0; i < end_loc[depth]; ++i)
								qcsr.candidate_vertex_set[next_vertex][i] = qcsr.candidate_vertex_set[vertex][i];
							if (Q.CEB_flag[next_vertex])
								Q.CEB_iter[next_vertex] = 0;					
							continue;
						}
					}		
					min_len = 100000;
					total_len = 0;
					for (unsigned i = 0; i < Q.backward_neighbor[next_vertex].size(); ++i)
					{
						idx = Q.backward_neighbor[next_vertex][i];
						v = embedding[idx];
						id = v*Q.vertex_num*2+next_vertex*2;
						if (qcsr.offset_list[id+1] == 0)
						{
							min_len = 0;
							break;
						}
						else
						{
							total_len += qcsr.offset_list[id+1];
							if (qcsr.offset_list[id+1] < min_len)
							{
								min_len = qcsr.offset_list[id+1];
								min_idx = i;
							}
							temp_begin_loc[i] = qcsr.offset_list[id];
							temp_end_loc[i] = temp_begin_loc[i] + qcsr.offset_list[id+1];	
						}				
					}					
					qcsr.candidate_size[next_vertex] = 0;
					if (min_len > 0)
					{
						if (total_len <= (min_len << 3)*(Q.backward_neighbor[next_vertex].size()))
							set_intersection(Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc);
						else
#ifdef hybrid_opt
							edge_check(G, Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc, temp_idx, min_idx);
#else
							set_intersection(Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc);
#endif
					}
					if (qcsr.candidate_size[next_vertex] == 0)
					{
						while(depth > idx)
						{
							depth--;
							vertex = Q.core_match_order[depth];
							temp_vertex = Q.CEB_write[vertex];
							if (temp_vertex < 32)
							{
								if (Q.CEB_update[temp_vertex])
								{
									for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
									{
										vertex = Q.core_match_order[depth-i];
										if (Q.repeat_state[vertex])
											Q.repeat[embedding[depth-i]] = 32;		
										for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
											Q.CEB_update[Q.parent[vertex][j]] = false;								
									}
									depth -= (Q.CEB_width[temp_vertex]-1);						
									continue;							
								}
							}
							if (Q.repeat_state[vertex])
								Q.repeat[embedding[depth]] = 32;
							for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
								Q.CEB_update[Q.parent[vertex][i]] = false;
						}
						CEB_expand_flag = idx;
						continue;
					}
					begin_loc[depth] = 0;
					end_loc[depth] = qcsr.candidate_size[next_vertex];
					if (Q.CEB_flag[next_vertex])
						Q.CEB_iter[next_vertex] = 0;					
				}
			}
			else
			{
#ifdef CEB_opt
				if (Q.CEB_flag[vertex])
					Q.CEB_update[vertex] = true;
#endif				
				depth--;
				vertex = Q.core_match_order[depth];
				temp_vertex = Q.CEB_write[vertex];
				if (temp_vertex < 32)
				{
					if (Q.CEB_update[temp_vertex])
					{
						for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
						{
							vertex = Q.core_match_order[depth-i];
							if (Q.repeat_state[vertex])
								Q.repeat[embedding[depth-i]] = 32;		
							if (change[depth-i])
							{
								for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)
									Q.CEB_update[Q.parent[vertex][j]] = false;
							}								
						}
						depth -= (Q.CEB_width[temp_vertex]-1);
						CEB_expand_flag = 32;
						continue;
					}					
				}
				if (Q.repeat_state[vertex])
					Q.repeat[embedding[depth]] = 32;
				for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
					Q.CEB_update[Q.parent[vertex][i]] = false;					
			}
		}
	}
}

void tl_match(Graph& G, Query& Q, QCSR& qcsr, unsigned* &res, unsigned& res_iter, unsigned** &index, vector<unsigned>* &leaf_res, vector<unsigned> &common_leaf_res, unsigned& res_num, unsigned& limited_num) //tree-match and leaf-match
{
	int depth = 0;
	unsigned* begin_loc;
	unsigned* end_loc;
	unsigned v; 
	unsigned vertex; 
	unsigned next_vertex; 
	unsigned idx; 
	unsigned label;
	bool flag;
	unsigned width, width1, width2;	
	unsigned level;	
	unsigned leaf_begin_loc, leaf_end_loc;
	unsigned id; 
	unsigned *temp_id = new unsigned[Q.leaf.size()]; 
	unsigned cnt;
	unsigned iter1, iter2, end1, end2;
	unsigned* common_leaf;

	begin_loc = new unsigned[Q.vertex_num];
	end_loc = new unsigned[Q.vertex_num];
	width = Q.vertex_num + Q.repeat_two_leaf_begin_loc;
	width1 = Q.tree.size() + Q.repeat_leaf_begin_loc*2;
	width2 = Q.tree.size() + Q.repeat_two_leaf_begin_loc*2;
	if (Q.repeat_two_leaf_begin_loc != 0)
	{
		index = new unsigned*[Q.repeat_two_leaf_begin_loc]; 
		for (unsigned i = 0; i < Q.repeat_two_leaf_begin_loc; ++i)
		{
			vertex = Q.leaf[i];
			if (Q.repeated_vertex_set[vertex].size() == 0)
			{
				idx = Q.backward_neighbor[vertex][0];
				label = Q.label_list[Q.tree_match_order[idx]];
				cnt = G.label_frequency[label]*2;
				index[i] = new unsigned[cnt];
				for (unsigned j = 0; j < cnt; ++j)
					index[i][j] = 0;
			}
		}
		leaf_res = new vector<unsigned>[Q.repeat_two_leaf_begin_loc]; 
		for (unsigned i = 0; i < Q.repeat_two_leaf_begin_loc; ++i)
			leaf_res[i].push_back(1); 	
		common_leaf_res.push_back(1); 
		common_leaf = new unsigned[Q.leaf.size()];		
	}
	unsigned* embedding = new unsigned[width];

	begin_loc[0] = 0;
	end_loc[0] = qcsr.candidate_size[Q.start_vertex];
	while (depth != -1) //DFS match
	{
		if (depth == 0)
		{
			if (begin_loc[depth] < end_loc[depth])
			{
				v = qcsr.candidate_vertex_set[Q.start_vertex][begin_loc[depth]];
				embedding[depth] = v;
				if (Q.repeat_state[Q.start_vertex])
					Q.repeat[v] = 0;
				begin_loc[depth]++;
				depth++;
				next_vertex = Q.tree_match_order[depth];
				id = v*Q.vertex_num*2+next_vertex*2;
				begin_loc[depth] = qcsr.offset_list[id];
				end_loc[depth] = begin_loc[depth] + qcsr.offset_list[id+1];
			}
			else
				depth--;
		}
		else if ((depth > 0) && (depth < Q.tree.size()-1))
		{
			if (begin_loc[depth] < end_loc[depth])
			{
				vertex = Q.tree_match_order[depth];
				v = qcsr.adjacency_list[begin_loc[depth]];

				if (Q.repeat_state[vertex])
				{
					if (Q.repeat[v] < 32)
					{
						begin_loc[depth]++;
						continue;
					}
					Q.repeat[v] = depth;
				}

				embedding[depth] = v;
				begin_loc[depth]++;
				depth++;
				next_vertex = Q.tree_match_order[depth];
				idx = Q.backward_neighbor[next_vertex][0];
				v = embedding[idx];
				id = v*Q.vertex_num*2+next_vertex*2;
				if (qcsr.offset_list[id+1] == 0)
				{
					depth--;
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;
#ifdef CEB_opt
					while(depth > idx)
					{
						depth--;
						vertex = Q.tree_match_order[depth];
						if (Q.repeat_state[vertex])
							Q.repeat[embedding[depth]] = 32;
					}
#endif
					continue;
				}
				begin_loc[depth] = qcsr.offset_list[id];
				end_loc[depth] = begin_loc[depth] + qcsr.offset_list[id+1];				
			}
			else
			{
				depth--;
				vertex = Q.tree_match_order[depth];
				if (Q.repeat_state[vertex])
					Q.repeat[embedding[depth]] = 32;
			}
		}
		else if (depth == Q.tree.size()-1)
		{
			if (begin_loc[depth] < end_loc[depth])
			{
				vertex = Q.tree_match_order[depth];
				v = qcsr.adjacency_list[begin_loc[depth]];
				
				if (Q.repeat_state[vertex])
				{
					if (Q.repeat[v] < 32)
					{
						begin_loc[depth]++;
						continue;
					}
					Q.repeat[v] = depth;
				}				

				embedding[depth] = v;
				begin_loc[depth]++;

				cnt = 1;
				flag = true;
				for (unsigned i = 0; i < Q.repeat_two_leaf_begin_loc; ++i)
				{
					next_vertex = Q.leaf[i];
					label = Q.label_list[next_vertex];
					idx = Q.backward_neighbor[next_vertex][0];
					v = embedding[idx];
					if (Q.repeated_vertex_set[next_vertex].size() == 0)
					{
						id = (v-G.label_index_list[G.label_list[v]])*2;
						if (index[i][id] == 0)
						{
							leaf_begin_loc = G.offset_list[v*G.label_num+label];
							leaf_end_loc =  G.offset_list[v*G.label_num+label+1];
							if ((leaf_end_loc-leaf_begin_loc) == 0)
							{
								flag = false;
								break;
							}
							index[i][id] = leaf_res[i][0];
							for (unsigned j = leaf_begin_loc; j < leaf_end_loc; ++j)
							{
								leaf_res[i].push_back(G.adjacency_list[j]);
								index[i][id+1]++;
							}
							leaf_res[i][0] += index[i][id+1];								
						}
						embedding[depth+2*i+1] = index[i][id]; 
						embedding[depth+2*i+2] = index[i][id+1];
					}
					else
					{
						leaf_begin_loc = G.offset_list[v*G.label_num+label];
						leaf_end_loc =  G.offset_list[v*G.label_num+label+1];
						if ((leaf_end_loc-leaf_begin_loc) == 0)
						{
							flag = false;
							break;
						}
						for (unsigned j = leaf_begin_loc; j < leaf_end_loc; ++j)
						{
							if (Q.repeat[G.adjacency_list[j]] < 32)
								continue;
							leaf_res[i].push_back(G.adjacency_list[j]);
						}
						if ((leaf_res[i].size() - leaf_res[i][0]) == 0)
						{
							flag = false;
							idx = Q.CEB_father_idx[next_vertex];
							break;							
						}
						embedding[depth+2*i+1] = leaf_res[i][0]; 
						embedding[depth+2*i+2] = leaf_res[i].size() - leaf_res[i][0];
						leaf_res[i][0] = leaf_res[i].size();
					}
				}
				if (!flag)
				{
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;
#ifdef CEB_opt
					while(depth > idx)
					{
						depth--;
						vertex = Q.tree_match_order[depth];
						if (Q.repeat_state[vertex])
							Q.repeat[embedding[depth]] = 32;
					}
#endif
					continue;
				}		
				for (unsigned i = 0; i < Q.repeat_leaf_begin_loc; ++i)
					cnt *= embedding[depth+2*i+2];			
				if (Q.repeat_leaf_begin_loc == Q.leaf.size())
				{
					for (unsigned i = 0; i < width1; ++i)
						res[res_iter++] = embedding[i];
					res_num += cnt; 
					if (res_num >= limited_num)
						return;
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;
					continue;
				}

				for (unsigned i = Q.repeat_leaf_begin_loc; i < Q.repeat_two_leaf_begin_loc; i = i+2)
				{
					common_leaf[i] = common_leaf_res[0]; 
					iter1 = embedding[depth+2*i+1];
					end1 = iter1 + embedding[depth+2*i+2];
					iter2 = embedding[depth+2*i+3];
					end2 = iter2 + embedding[depth+2*i+4];
					while (true)
					{
						if (iter1 == end1)
							break;
						if (iter2 == end2)
							break;
						if (leaf_res[i][iter1] < leaf_res[i+1][iter2])
							iter1++;
						else if (leaf_res[i][iter1] > leaf_res[i+1][iter2])
							iter2++;
						else
						{
							common_leaf_res.push_back(leaf_res[i][iter1]);
							iter1++;
							iter2++;
						}
					}
					common_leaf[i+1] = common_leaf_res.size() - common_leaf_res[0];
					common_leaf_res[0] = common_leaf_res.size();
					cnt *= (embedding[depth+2*i+2]*embedding[depth+2*i+4]-common_leaf[i+1]);
					if (cnt == 0)
					{
						if (Q.repeat_state[vertex])
							Q.repeat[embedding[depth]] = 32;
#ifdef CEB_opt
						idx = max(Q.backward_neighbor[Q.leaf[i]][0], Q.backward_neighbor[Q.leaf[i+1]][0]);
						while(depth > idx)
						{
							depth--;
							vertex = Q.tree_match_order[depth];
							if (Q.repeat_state[vertex])
								Q.repeat[embedding[depth]] = 32;
						}
#endif
						continue;		
					}

				}
				if (Q.repeat_two_leaf_begin_loc == Q.leaf.size())
				{
					width = depth + 1 + Q.repeat_leaf_begin_loc*2;
					for (unsigned i = 0; i < width1; ++i)
						res[res_iter++] = embedding[i];
					for (unsigned i = Q.repeat_leaf_begin_loc; i < Q.repeat_two_leaf_begin_loc; i = i+2)
					{
						res[res_iter++] = embedding[depth + 1 + 2*i];	
						res[res_iter++] = embedding[depth + 2 + 2*i];	
						res[res_iter++] = embedding[depth + 3 + 2*i];	
						res[res_iter++] = embedding[depth + 4 + 2*i];	
						res[res_iter++] = common_leaf[i];
						res[res_iter++] = common_leaf[i+1];
					}
					res_num += cnt; 
					if (res_num >= limited_num)
						return;
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;
					continue;
				}

				for (unsigned i = Q.repeat_two_leaf_begin_loc; i < Q.leaf.size(); ++i)
				{
					next_vertex = Q.leaf[i];
					label = Q.label_list[next_vertex];
					idx = Q.backward_neighbor[next_vertex][0];
					v = embedding[idx];
					temp_id[i] = v*G.label_num+label;
					if ((G.offset_list[temp_id[i]+1]-G.offset_list[temp_id[i]]) == 0)
					{
						flag = false;
						break;
					}						
					begin_loc[depth+1+i] = G.offset_list[temp_id[i]];
					end_loc[depth+1+i] = G.offset_list[temp_id[i]+1];
				}
				if (!flag)
				{
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;
#ifdef CEB_opt
					while(depth > idx)
					{
						depth--;
						vertex = Q.tree_match_order[depth];
						if (Q.repeat_state[vertex])
							Q.repeat[embedding[depth]] = 32;
					}
#endif
					continue;
				}		
				level = depth + 1 + Q.repeat_two_leaf_begin_loc;		
				while (true)
				{
					if (level == Q.vertex_num-1)
					{
						flag = false;
						while (begin_loc[level] < end_loc[level])
						{
							v = G.adjacency_list[begin_loc[level]];
							if (Q.repeat[v] < 32)
							{
								begin_loc[level]++;
								continue;
							}
							if (!flag)
							{
								for (unsigned i = 0; i < width1 ; ++i)
									res[res_iter++] = embedding[i];	
								for (unsigned i = Q.repeat_leaf_begin_loc; i < Q.repeat_two_leaf_begin_loc; i = i+2)
								{
									res[res_iter++] = embedding[depth + 1 + 2*i];	
									res[res_iter++] = embedding[depth + 2 + 2*i];	
									res[res_iter++] = embedding[depth + 3 + 2*i];	
									res[res_iter++] = embedding[depth + 4 + 2*i];	
									res[res_iter++] = common_leaf[i];
									res[res_iter++] = common_leaf[i+1];
								}
								for (unsigned i = width2; i < width-1 ; ++i)
									res[res_iter++] = embedding[i];	
								res[res_iter++] = v;
								res_num += cnt;
								flag = true;	
								begin_loc[level]++;				
							}
							else
							{
								res_iter += (Q.repeat_two_leaf_begin_loc-Q.repeat_leaf_begin_loc+width-1);
								res[res_iter++] = v;
								res_num += cnt;
								begin_loc[level]++;	
							}
						}
						if (res_num >= limited_num)
							return;
						level--;
						Q.repeat[embedding[level+Q.repeat_two_leaf_begin_loc]] = 32;	
						continue;						
					}
					if (begin_loc[level] < end_loc[level])
					{
						v = G.adjacency_list[begin_loc[level]];
						if (Q.repeat[v] < 32)
						{
							begin_loc[level]++;
							continue;
						}
						embedding[level+Q.repeat_two_leaf_begin_loc] = v;
						Q.repeat[v] = level;
						begin_loc[level]++;
						level++;
						begin_loc[level] = G.offset_list[temp_id[level-Q.tree.size()]];	
					}
					else
					{
						level--;
						if (level == depth + Q.repeat_two_leaf_begin_loc)
							break;
						Q.repeat[embedding[level+Q.repeat_two_leaf_begin_loc]] = 32;							
					}
				}
				if (Q.repeat_state[vertex])
					Q.repeat[embedding[depth]] = 32;
			}
			else
			{
				depth--;
				vertex = Q.tree_match_order[depth];
				if (Q.repeat_state[vertex])
					Q.repeat[embedding[depth]] = 32;
			}
		}
	}
}

void cfl_match(Graph& G, Query& Q, QCSR& qcsr, unsigned* &res, unsigned& res_iter, unsigned** &index, vector<unsigned>* &leaf_res, vector<unsigned> &common_leaf_res, unsigned& res_num, unsigned& limited_num) // core-match, forest-match and leaf-match
{
	int depth = 0;
	unsigned* begin_loc; 
	unsigned* end_loc; 
	bool* change;
	unsigned* temp_begin_loc = new unsigned[Q.max_degree]; 
	unsigned* temp_end_loc = new unsigned[Q.max_degree]; 
	unsigned* temp_idx = new unsigned[Q.max_degree]; 
	unsigned v; 
	unsigned vertex; 
	unsigned temp_vertex; 
	unsigned next_vertex; 
	unsigned idx; 
	unsigned label;
	bool flag;
	unsigned CEB_expand_flag;
	unsigned width, width1, width2;
	unsigned level;	
	unsigned level_end_loc;
	unsigned leaf_begin_loc, leaf_end_loc;
	unsigned id; 
	unsigned *temp_id = new unsigned[Q.leaf.size()]; 
	unsigned cnt;
	unsigned min_len;
	unsigned total_len;
	unsigned min_idx; 
	unsigned iter1, iter2, end1, end2;
	unsigned* common_leaf;

	vector<unsigned> match_order;
	for (unsigned i = 0; i < Q.core_match_order.size(); ++i)
		match_order.push_back(Q.core_match_order[i]);
	unsigned root;
	for (unsigned i = 0; i < Q.forest.size(); ++i)
	{
		root = Q.forest[i];
		for (unsigned j = 0; j < Q.forest_match_order[root].size(); ++j)
			match_order.push_back(Q.forest_match_order[root][j]);
	}

	begin_loc = new unsigned[Q.vertex_num];
	end_loc = new unsigned[Q.vertex_num];
	change = new bool[Q.vertex_num];
	width = Q.vertex_num + Q.repeat_two_leaf_begin_loc;
	width1 = match_order.size() + Q.repeat_leaf_begin_loc*2;
	width2 = match_order.size() + Q.repeat_two_leaf_begin_loc*2;
	if (Q.repeat_two_leaf_begin_loc != 0)
	{
		index = new unsigned*[Q.repeat_two_leaf_begin_loc];
		for (unsigned i = 0; i < Q.repeat_two_leaf_begin_loc; ++i)
		{
			vertex = Q.leaf[i];
			if (Q.repeated_vertex_set[vertex].size() == 0)
			{
				idx = Q.backward_neighbor[vertex][0];
				label = Q.label_list[match_order[idx]];
				cnt = G.label_frequency[label]*2;
				index[i] = new unsigned[cnt];
				for (unsigned j = 0; j < cnt; ++j)
					index[i][j] = 0;
			}
		}
		leaf_res = new vector<unsigned>[Q.repeat_two_leaf_begin_loc];
		for (unsigned i = 0; i < Q.repeat_two_leaf_begin_loc; ++i)
			leaf_res[i].push_back(1);		
		common_leaf_res.push_back(1); 
		common_leaf = new unsigned[Q.leaf.size()];	
	}
	unsigned* embedding = new unsigned[width];

	begin_loc[0] = 0;
	end_loc[0] = qcsr.candidate_size[Q.start_vertex];
	while (depth != -1) //DFS match
	{
		if (depth == 0)
		{
			if (begin_loc[depth] < end_loc[depth])
			{
				v = qcsr.candidate_vertex_set[Q.start_vertex][begin_loc[depth]];
				embedding[depth] = v;
				if (Q.repeat_state[Q.start_vertex])
					Q.repeat[v] = 0;
				begin_loc[depth]++;
				depth++;
				next_vertex = match_order[depth];
				id = v*Q.vertex_num*2+next_vertex*2;
				begin_loc[depth] = qcsr.offset_list[id];
				end_loc[depth] = begin_loc[depth] + qcsr.offset_list[id+1];
			}
			else
				depth--;	
		}
		else if ((depth > 0) && (depth < match_order.size()))
		{ 
			vertex = match_order[depth];
			if (begin_loc[depth] < end_loc[depth])
			{
				if ((Q.CEB_update[vertex])&&(Q.CEB_width[vertex] > 1))
				{
					if (CEB_expand_flag < 32)
					{
						if (embedding[CEB_expand_flag] == Q.CEB[vertex][begin_loc[depth]+CEB_expand_flag-depth])
						{
							begin_loc[depth] += Q.CEB_width[vertex];
							continue;
						}
						else
						{
							unsigned i = 0;
							while (i < Q.CEB_width[vertex])
							{
								embedding[depth+i] = Q.CEB[vertex][begin_loc[depth]];	
								begin_loc[depth]++;	
								if (Q.repeat_state[match_order[depth+i]])
									Q.repeat[embedding[depth+i]] = depth+i;
								++i;						
							}		
							depth += Q.CEB_width[vertex];	
							continue;							
						}
					}
					unsigned i = 0;
					while (i < Q.CEB_width[vertex])
					{
						embedding[depth+i] = Q.CEB[vertex][begin_loc[depth]];	
						begin_loc[depth]++;	
						if (Q.repeat_state[match_order[depth+i]])
							Q.repeat[embedding[depth+i]] = depth+i;
						++i;			
					}
					for (i = 0; i < Q.CEB_width[vertex]; ++i)
					{
						if (begin_loc[depth] == end_loc[depth])
							change[depth+i] = true;
						else
						{
							if (embedding[depth+i] == Q.CEB[vertex][begin_loc[depth]+i])
								change[depth+i] = false;
							else
							{
								change[depth+i] = true;
								for (unsigned j = i+1; j < Q.CEB_width[vertex]; ++j)
									change[depth+j] = true;
								break;
							}
						}
					}	
					depth += Q.CEB_width[vertex];	
				}
				else
				{
					if (Q.backward_neighbor[vertex].size() == 1)
						v = qcsr.adjacency_list[begin_loc[depth]];
					else
						v = qcsr.candidate_vertex_set[vertex][begin_loc[depth]];

					if (Q.repeat_state[vertex])
					{
						if (Q.repeat[v] < 32)
						{
							begin_loc[depth]++;
							Q.CEB_father_idx[vertex] = max(Q.CEB_father_idx[vertex], Q.repeat[v]);
							continue;
						}
						Q.repeat[v] = depth;
					}

					embedding[depth] = v;
					begin_loc[depth]++;			
					depth++;
#ifdef CEB_opt
					temp_vertex = Q.CEB_write[vertex];
					if (temp_vertex < 32)
					{
						for (unsigned i = depth-Q.CEB_width[temp_vertex]; i < depth; ++i)
						{
							Q.CEB[temp_vertex][Q.CEB_iter[temp_vertex]++] = embedding[i];
						}
					}
					else
					{
						if (Q.CEB_width[vertex] == 1)
							Q.CEB_iter[vertex]++;
					}
#endif
				}
				if (depth == match_order.size())
					continue;
				next_vertex = match_order[depth];

				if (Q.backward_neighbor[next_vertex].size() == 1) 
				{
					if (Q.CEB_update[next_vertex])
					{
						if (Q.CEB_iter[next_vertex] == 0)
						{
							while(depth > Q.CEB_father_idx[next_vertex])
							{
								depth--;
								vertex = match_order[depth];
								temp_vertex = Q.CEB_write[vertex];
								if (temp_vertex < 32)
								{
									if (Q.CEB_update[temp_vertex])
									{
										for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
										{
											vertex = match_order[depth-i];
											if (Q.repeat_state[vertex])
												Q.repeat[embedding[depth-i]] = 32;		
											for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
												Q.CEB_update[Q.parent[vertex][j]] = false;								
										}
										depth -= (Q.CEB_width[temp_vertex]-1);						
										continue;							
									}
								}
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth]] = 32;
								for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
									Q.CEB_update[Q.parent[vertex][i]] = false;
							}
							CEB_expand_flag = Q.CEB_father_idx[next_vertex];
							continue;							
						}
						if (Q.CEB_width[next_vertex] > 1)
						{						
							begin_loc[depth] = 0;		
							end_loc[depth] = Q.CEB_iter[next_vertex];	
							CEB_expand_flag = 32;			
						}	
						else
						{
							idx = Q.backward_neighbor[next_vertex][0];
							v = embedding[idx];
							id = v*Q.vertex_num*2+next_vertex*2;
							begin_loc[depth] = qcsr.offset_list[id];
							end_loc[depth] = begin_loc[depth] + qcsr.offset_list[id+1];	
						}	
						continue;	
					}
					idx = Q.backward_neighbor[next_vertex][0];
					v = embedding[idx];
					id = v*Q.vertex_num*2+next_vertex*2;
					if (qcsr.offset_list[id+1] == 0)
					{
						while(depth > idx)
						{
							depth--;
							vertex = match_order[depth];
							temp_vertex = Q.CEB_write[vertex];
							if (temp_vertex < 32)
							{
								if (Q.CEB_update[temp_vertex])
								{
									for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
									{
										vertex = match_order[depth-i];
										if (Q.repeat_state[vertex])
											Q.repeat[embedding[depth-i]] = 32;		
										for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
											Q.CEB_update[Q.parent[vertex][j]] = false;								
									}
									depth -= (Q.CEB_width[temp_vertex]-1);						
									continue;							
								}
							}
							if (Q.repeat_state[vertex])
								Q.repeat[embedding[depth]] = 32;
							for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
								Q.CEB_update[Q.parent[vertex][i]] = false;								
						}
						CEB_expand_flag = idx;
						continue;					
					}
					begin_loc[depth] = qcsr.offset_list[id];
					end_loc[depth] = begin_loc[depth] + qcsr.offset_list[id+1];	
					Q.CEB_father_idx[next_vertex] = idx;

					if (qcsr.offset_list[id+1] == 1)
					{
						bool test_repeat = true;
						v = qcsr.adjacency_list[begin_loc[depth]];
						if (Q.repeat_state[next_vertex])
						{
							if (Q.repeat[v] == 32)
								test_repeat = false;
						}
						if (test_repeat)
						{
							Q.CEB_father_idx[next_vertex] = max(Q.CEB_father_idx[next_vertex], Q.repeat[v]);
							while(depth > Q.CEB_father_idx[next_vertex])
							{
								depth--;
								vertex = match_order[depth];
								temp_vertex = Q.CEB_write[vertex];
								if (temp_vertex < 32)
								{
									if (Q.CEB_update[temp_vertex])
									{
										for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
										{
											vertex = match_order[depth-i];
											if (Q.repeat_state[vertex])
												Q.repeat[embedding[depth-i]] = 32;		
											for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
												Q.CEB_update[Q.parent[vertex][j]] = false;								
										}
										depth -= (Q.CEB_width[temp_vertex]-1);						
										continue;							
									}
								}
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth]] = 32;
								for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
									Q.CEB_update[Q.parent[vertex][i]] = false;	
							}
							CEB_expand_flag = Q.CEB_father_idx[next_vertex];
							continue;
						}
					}

					if (Q.CEB_flag[next_vertex])
						Q.CEB_iter[next_vertex] = 0;													
				}
				else
				{
					if (Q.CEB_update[next_vertex])
					{
						if (Q.CEB_iter[next_vertex] == 0)
						{
							while(depth > Q.CEB_father_idx[next_vertex])
							{
								depth--;
								vertex = match_order[depth];
								temp_vertex = Q.CEB_write[vertex];
								if (temp_vertex < 32)
								{
									if (Q.CEB_update[temp_vertex])
									{
										for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
										{
											vertex = match_order[depth-i];
											if (Q.repeat_state[vertex])
												Q.repeat[embedding[depth-i]] = 32;		
											for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
												Q.CEB_update[Q.parent[vertex][j]] = false;								
										}
										depth -= (Q.CEB_width[temp_vertex]-1);						
										continue;							
									}
								}
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth]] = 32;
								for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
									Q.CEB_update[Q.parent[vertex][i]] = false;	
							}
							CEB_expand_flag = Q.CEB_father_idx[next_vertex];
							continue;							
						}
						if (Q.CEB_width[next_vertex] > 1)
						{
							begin_loc[depth] = 0;		
							end_loc[depth] = Q.CEB_iter[next_vertex];	
							CEB_expand_flag = 32;
						}	
						else	
							begin_loc[depth] = 0;		
						continue;				
					}
					if (Q.NEC[next_vertex] != 0)
					{
						if (Q.NEC[vertex] != Q.NEC[next_vertex])
						{
							min_len = 100000;
							total_len = 0;
							for (unsigned i = 0; i < Q.backward_neighbor[next_vertex].size(); ++i)
							{
								idx = Q.backward_neighbor[next_vertex][i];
								v = embedding[idx];
								id = v*Q.vertex_num*2+next_vertex*2;
								if (qcsr.offset_list[id+1] <= 1)
								{
									min_len = 0;
									break;
								}
								else
								{
									total_len += qcsr.offset_list[id+1];
									if (qcsr.offset_list[id+1] < min_len)
									{
										min_len = qcsr.offset_list[id+1];
										min_idx = i;
									}
									temp_begin_loc[i] = qcsr.offset_list[id];
									temp_end_loc[i] = temp_begin_loc[i] + qcsr.offset_list[id+1];	
								}					
							}
							qcsr.candidate_size[next_vertex] = 0;
							if (min_len > 0)
							{
								if (total_len <= (min_len << 3)*(Q.backward_neighbor[next_vertex].size()))
									set_intersection(Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc);
								else
#ifdef hybrid_opt
									edge_check(G, Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc, temp_idx, min_idx);
#else
									set_intersection(Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc);
#endif
							}							
							if (qcsr.candidate_size[next_vertex] <= 1)
							{
								while(depth > idx)
								{
									depth--;
									vertex = match_order[depth];
									temp_vertex = Q.CEB_write[vertex];
									if (temp_vertex < 32)
									{
										if (Q.CEB_update[temp_vertex])
										{
											for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
											{
												vertex = match_order[depth-i];
												if (Q.repeat_state[vertex])
													Q.repeat[embedding[depth-i]] = 32;		
												for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
													Q.CEB_update[Q.parent[vertex][j]] = false;								
											}
											depth -= (Q.CEB_width[temp_vertex]-1);						
											continue;							
										}
									}
									if (Q.repeat_state[vertex])
										Q.repeat[embedding[depth]] = 32;
									for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
										Q.CEB_update[Q.parent[vertex][i]] = false;	
								}
								CEB_expand_flag = idx;
								continue;
							}
							begin_loc[depth] = 0;
							end_loc[depth] = qcsr.candidate_size[next_vertex];
							Q.CEB_father_idx[next_vertex] = idx;
							if (Q.CEB_flag[next_vertex])
								Q.CEB_iter[next_vertex] = 0;					
							continue;
						}
						else
						{
							begin_loc[depth] = 0;
							end_loc[depth] = end_loc[depth-1];
							for (unsigned i = 0; i < end_loc[depth]; ++i)
								qcsr.candidate_vertex_set[next_vertex][i] = qcsr.candidate_vertex_set[vertex][i];
							Q.CEB_father_idx[next_vertex] = Q.CEB_father_idx[vertex];
							if (Q.CEB_flag[next_vertex])
								Q.CEB_iter[next_vertex] = 0;					
							continue;
						}
					}
					min_len = 100000;
					total_len = 0;
					for (unsigned i = 0; i < Q.backward_neighbor[next_vertex].size(); ++i)
					{
						idx = Q.backward_neighbor[next_vertex][i];
						v = embedding[idx];
						id = v*Q.vertex_num*2+next_vertex*2;
						if (qcsr.offset_list[id+1] == 0)
						{
							min_len = 0;
							break;
						}
						else
						{
							total_len += qcsr.offset_list[id+1];
							if (qcsr.offset_list[id+1] < min_len)
							{
								min_len = qcsr.offset_list[id+1];
								min_idx = i;
							}
							temp_begin_loc[i] = qcsr.offset_list[id];
							temp_end_loc[i] = temp_begin_loc[i] + qcsr.offset_list[id+1];	
						}				
					}
					qcsr.candidate_size[next_vertex] = 0;
					if (min_len > 0)
					{
						if (total_len <= (min_len << 3)*(Q.backward_neighbor[next_vertex].size()))
							set_intersection(Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc);
						else
#ifdef hybrid_opt
							edge_check(G, Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc, temp_idx, min_idx);
#else
							set_intersection(Q, qcsr, next_vertex, embedding, temp_begin_loc, temp_end_loc);
#endif
					}
					if (qcsr.candidate_size[next_vertex] == 0)
					{
						while(depth > idx)
						{
							depth--;
							vertex = match_order[depth];
							temp_vertex = Q.CEB_write[vertex];
							if (temp_vertex < 32)
							{
								if (Q.CEB_update[temp_vertex])
								{
									for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
									{
										vertex = match_order[depth-i];
										if (Q.repeat_state[vertex])
											Q.repeat[embedding[depth-i]] = 32;		
										for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
											Q.CEB_update[Q.parent[vertex][j]] = false;								
									}
									depth -= (Q.CEB_width[temp_vertex]-1);						
									continue;							
								}
							}
							if (Q.repeat_state[vertex])
								Q.repeat[embedding[depth]] = 32;
							for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
								Q.CEB_update[Q.parent[vertex][i]] = false;	
						}
						CEB_expand_flag = idx;
						continue;
					}
					begin_loc[depth] = 0;
					end_loc[depth] = qcsr.candidate_size[next_vertex];
					Q.CEB_father_idx[next_vertex] = idx;

					if (end_loc[depth] == 1)
					{
						bool test_repeat = true;
						v = qcsr.candidate_vertex_set[next_vertex][0];
						if (Q.repeat_state[next_vertex])
						{
							if (Q.repeat[v] == 32)
								test_repeat = false;
						}
						if (test_repeat)
						{
							Q.CEB_father_idx[next_vertex] = max(Q.CEB_father_idx[next_vertex], Q.repeat[v]);
							while(depth > Q.CEB_father_idx[next_vertex])
							{
								depth--;
								vertex = match_order[depth];
								temp_vertex = Q.CEB_write[vertex];
								if (temp_vertex < 32)
								{
									if (Q.CEB_update[temp_vertex])
									{
										for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
										{
											vertex = match_order[depth-i];
											if (Q.repeat_state[vertex])
												Q.repeat[embedding[depth-i]] = 32;		
											for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
												Q.CEB_update[Q.parent[vertex][j]] = false;								
										}
										depth -= (Q.CEB_width[temp_vertex]-1);						
										continue;							
									}
								}
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth]] = 32;
								for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
									Q.CEB_update[Q.parent[vertex][i]] = false;	
							}
							CEB_expand_flag = Q.CEB_father_idx[next_vertex];
							continue;
						}
					}

					if (Q.CEB_flag[next_vertex])
						Q.CEB_iter[next_vertex] = 0;					
				}
						
			}
			else
			{
#ifdef CEB_opt
				if (Q.CEB_flag[vertex])
					Q.CEB_update[vertex] = true;
#endif			
					depth--;
					vertex = match_order[depth];
					temp_vertex = Q.CEB_write[vertex];
					if (temp_vertex < 32)
					{
						if (Q.CEB_update[temp_vertex])
						{
							for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
							{
								vertex = match_order[depth-i];
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth-i]] = 32;	
								if (change[depth-i])
								{
									for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)
										Q.CEB_update[Q.parent[vertex][j]] = false;
								}										
							}
							depth -= (Q.CEB_width[temp_vertex]-1);
							CEB_expand_flag = 32;
							continue;
						}					
					}
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;
					for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
						Q.CEB_update[Q.parent[vertex][i]] = false;
			}	
				
		}
		else if (depth == match_order.size())
		{				
				cnt = 1;
				flag = true;

				for (unsigned i = 0; i < Q.repeat_two_leaf_begin_loc; ++i)
				{
					next_vertex = Q.leaf[i];
					label = Q.label_list[next_vertex];
					idx = Q.backward_neighbor[next_vertex][0];
					v = embedding[idx];
					if (Q.repeated_vertex_set[next_vertex].size() == 0)
					{
						id = (v-G.label_index_list[G.label_list[v]])*2;
						if (index[i][id] == 0) 
						{
							leaf_begin_loc = G.offset_list[v*G.label_num+label];
							leaf_end_loc =  G.offset_list[v*G.label_num+label+1];
							if ((leaf_end_loc-leaf_begin_loc) == 0)
							{
								flag = false;
								break;
							}
							index[i][id] = leaf_res[i][0];
							for (unsigned j = leaf_begin_loc; j < leaf_end_loc; ++j)
							{
								leaf_res[i].push_back(G.adjacency_list[j]);
								index[i][id+1]++;
							}
							leaf_res[i][0] += index[i][id+1];								
						}
						embedding[depth+2*i] = index[i][id]; 
						embedding[depth+2*i+1] = index[i][id+1];
					}
					else
					{
						leaf_begin_loc = G.offset_list[v*G.label_num+label];
						leaf_end_loc =  G.offset_list[v*G.label_num+label+1];
						if ((leaf_end_loc-leaf_begin_loc) == 0)
						{
							flag = false;
							break;
						}
						for (unsigned j = leaf_begin_loc; j < leaf_end_loc; ++j)
						{
							if (Q.repeat[G.adjacency_list[j]] < 32)
								continue;
							leaf_res[i].push_back(G.adjacency_list[j]);
						}
						if ((leaf_res[i].size() - leaf_res[i][0]) == 0)
						{
							flag = false;
							idx = Q.CEB_father_idx[next_vertex];
							break;							
						}
						embedding[depth+2*i] = leaf_res[i][0]; 
						embedding[depth+2*i+1] = leaf_res[i].size() - leaf_res[i][0];
						leaf_res[i][0] = leaf_res[i].size();
					}
				}
				if (!flag)
				{
					while(depth > idx)
					{
						depth--;
						vertex = match_order[depth];
						temp_vertex = Q.CEB_write[vertex];
						if (temp_vertex < 32)
						{
							if (Q.CEB_update[temp_vertex])
							{
								for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
								{
									vertex = match_order[depth-i];
									if (Q.repeat_state[vertex])
										Q.repeat[embedding[depth-i]] = 32;		
									for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
										Q.CEB_update[Q.parent[vertex][j]] = false;								
								}
								depth -= (Q.CEB_width[temp_vertex]-1);						
								continue;							
							}
						}
						if (Q.repeat_state[vertex])
							Q.repeat[embedding[depth]] = 32;
						for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
							Q.CEB_update[Q.parent[vertex][i]] = false;
					}
					CEB_expand_flag = idx;
					continue;				
				}		
				for (unsigned i = 0; i < Q.repeat_leaf_begin_loc; ++i)
					cnt *= embedding[depth+2*i+1];			
				if (Q.repeat_leaf_begin_loc == Q.leaf.size())
				{
					for (unsigned i = 0; i < width1; ++i)
						res[res_iter++] = embedding[i];
					res_num += cnt; 

					if (res_num >= limited_num)
						return;
					depth--;
					vertex = match_order[depth];
					temp_vertex = Q.CEB_write[vertex];
					if (temp_vertex < 32)
					{
						if (Q.CEB_update[temp_vertex])
						{
							for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
							{
								vertex = match_order[depth-i];
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth-i]] = 32;										
							}
							depth -= (Q.CEB_width[temp_vertex]-1);
							CEB_expand_flag = 32;
							continue;							
						}
					}
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;	

					continue;				
				}

				for (unsigned i = Q.repeat_leaf_begin_loc; i < Q.repeat_two_leaf_begin_loc; i = i+2)
				{
					common_leaf[i] = common_leaf_res[0]; 
					iter1 = embedding[depth+2*i];
					end1 = iter1 + embedding[depth+2*i+1];
					iter2 = embedding[depth+2*i+2];
					end2 = iter2 + embedding[depth+2*i+3];
					while (true)
					{
						if (iter1 == end1)
							break;
						if (iter2 == end2)
							break;
						if (leaf_res[i][iter1] < leaf_res[i+1][iter2])
							iter1++;
						else if (leaf_res[i][iter1] > leaf_res[i+1][iter2])
							iter2++;
						else
						{
							common_leaf_res.push_back(leaf_res[i][iter1]);
							iter1++;
							iter2++;
						}
					}
					common_leaf[i+1] = common_leaf_res.size() - common_leaf_res[0];
					common_leaf_res[0] = common_leaf_res.size();
					cnt *= (embedding[depth+2*i+1]*embedding[depth+2*i+3]-common_leaf[i+1]);
					if (cnt == 0)
					{
						idx = max(Q.backward_neighbor[Q.leaf[i]][0], Q.backward_neighbor[Q.leaf[i+1]][0]);			
						while(depth > idx)
						{
							depth--;
							vertex = match_order[depth];
							temp_vertex = Q.CEB_write[vertex];
							if (temp_vertex < 32)
							{
								if (Q.CEB_update[temp_vertex])
								{
									for (unsigned ii = 0; ii < Q.CEB_width[temp_vertex]; ++ii)
									{
										vertex = match_order[depth-ii];
										if (Q.repeat_state[vertex])
											Q.repeat[embedding[depth-ii]] = 32;		
										for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
											Q.CEB_update[Q.parent[vertex][j]] = false;								
									}
									depth -= (Q.CEB_width[temp_vertex]-1);						
									continue;							
								}
							}
							if (Q.repeat_state[vertex])
								Q.repeat[embedding[depth]] = 32;
							for (unsigned ii = 0; ii < Q.parent[vertex].size(); ++ii)
								Q.CEB_update[Q.parent[vertex][ii]] = false;
						}
						CEB_expand_flag = idx;
						continue;			
					}
				}
				if (Q.repeat_two_leaf_begin_loc == Q.leaf.size())
				{
					width = depth + Q.repeat_leaf_begin_loc*2;
					for (unsigned i = 0; i < width1; ++i)
						res[res_iter++] = embedding[i];
					for (unsigned i = Q.repeat_leaf_begin_loc; i < Q.repeat_two_leaf_begin_loc; i = i+2)
					{
						res[res_iter++] = embedding[depth + 2*i];	
						res[res_iter++] = embedding[depth + 1 + 2*i];	
						res[res_iter++] = embedding[depth + 2 + 2*i];	
						res[res_iter++] = embedding[depth + 3 + 2*i];	
						res[res_iter++] = common_leaf[i];
						res[res_iter++] = common_leaf[i+1];
					}
					res_num += cnt; 

					if (res_num >= limited_num)
						return;
					depth--;
					vertex = match_order[depth];
					temp_vertex = Q.CEB_write[vertex];
					if (temp_vertex < 32)
					{
						if (Q.CEB_update[temp_vertex])
						{
							for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
							{
								vertex = match_order[depth-i];
								if (Q.repeat_state[vertex])
									Q.repeat[embedding[depth-i]] = 32;										
							}
							depth -= (Q.CEB_width[temp_vertex]-1);
							CEB_expand_flag = 32;
							continue;							
						}
					}
					if (Q.repeat_state[vertex])
						Q.repeat[embedding[depth]] = 32;
						
					continue;
				}

				for (unsigned i = Q.repeat_two_leaf_begin_loc; i < Q.leaf.size(); ++i)
				{
					next_vertex = Q.leaf[i];
					label = Q.label_list[next_vertex];
					idx = Q.backward_neighbor[next_vertex][0];
					v = embedding[idx];
					temp_id[i] = v*G.label_num+label;
					if ((G.offset_list[temp_id[i]+1]-G.offset_list[temp_id[i]]) == 0)
					{
						flag = false;
						break;
					}						
					begin_loc[depth+i] = G.offset_list[temp_id[i]];
					end_loc[depth+i] = G.offset_list[temp_id[i]+1];
				}
				if (!flag)
				{
					while(depth > idx)
					{
						depth--;
						vertex = match_order[depth];
						temp_vertex = Q.CEB_write[vertex];
						if (temp_vertex < 32)
						{
							if (Q.CEB_update[temp_vertex])
							{
								for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
								{
									vertex = match_order[depth-i];
									if (Q.repeat_state[vertex])
										Q.repeat[embedding[depth-i]] = 32;		
									for (unsigned j = 0; j < Q.parent[vertex].size(); ++j)	
										Q.CEB_update[Q.parent[vertex][j]] = false;								
								}
								depth -= (Q.CEB_width[temp_vertex]-1);						
								continue;							
							}
						}
						if (Q.repeat_state[vertex])
							Q.repeat[embedding[depth]] = 32;
						for (unsigned i = 0; i < Q.parent[vertex].size(); ++i)
							Q.CEB_update[Q.parent[vertex][i]] = false;
					}
					CEB_expand_flag = idx;
					continue;						
				}		
				level = depth + Q.repeat_two_leaf_begin_loc;		
				while (true)
				{
					if (level == Q.vertex_num-1)
					{
						flag = false;
						while (begin_loc[level] < end_loc[level])
						{
							v = G.adjacency_list[begin_loc[level]];
							if (Q.repeat[v] < 32)
							{
								begin_loc[level]++;
								continue;
							}
							if (!flag)
							{
								for (unsigned i = 0; i < width1 ; ++i)
									res[res_iter++] = embedding[i];	
								for (unsigned i = Q.repeat_leaf_begin_loc; i < Q.repeat_two_leaf_begin_loc; i = i+2)
								{
									res[res_iter++] = embedding[depth + 2*i];	
									res[res_iter++] = embedding[depth + 1 + 2*i];	
									res[res_iter++] = embedding[depth + 2 + 2*i];	
									res[res_iter++] = embedding[depth + 3 + 2*i];	
									res[res_iter++] = common_leaf[i];
									res[res_iter++] = common_leaf[i+1];
								}
								for (unsigned i = width2; i < width-1 ; ++i)
									res[res_iter++] = embedding[i];	
								res[res_iter++] = v;
								res_num += cnt;
								flag = true;	
								begin_loc[level]++;				
							}
							else
							{
								res_iter += (Q.repeat_two_leaf_begin_loc-Q.repeat_leaf_begin_loc+width-1);
								res[res_iter++] = v;
								res_num += cnt;
								begin_loc[level]++;	
							}
						}
						if (res_num >= limited_num)
							return;
						level--;
						Q.repeat[embedding[level+Q.repeat_two_leaf_begin_loc]] = 32;	
						continue;						
					}
					if (begin_loc[level] < end_loc[level])
					{
						v = G.adjacency_list[begin_loc[level]];
						if (Q.repeat[v] < 32)
						{
							begin_loc[level]++;
							continue;
						}
						embedding[level+Q.repeat_two_leaf_begin_loc] = v;
						Q.repeat[v] = level;
						begin_loc[level]++;
						level++;
						begin_loc[level] = G.offset_list[temp_id[level-match_order.size()]];	
					}
					else
					{
						level--;
						if (level == depth + Q.repeat_two_leaf_begin_loc - 1)
							break;
						Q.repeat[embedding[level+Q.repeat_two_leaf_begin_loc]] = 32;							
					}
				}
				depth--;
				vertex = match_order[depth];
				temp_vertex = Q.CEB_write[vertex];
				if (temp_vertex < 32)
				{
					if (Q.CEB_update[temp_vertex])
					{
						for (unsigned i = 0; i < Q.CEB_width[temp_vertex]; ++i)
						{
							vertex = match_order[depth-i];
							if (Q.repeat_state[vertex])
								Q.repeat[embedding[depth-i]] = 32;										
						}
						depth -= (Q.CEB_width[temp_vertex]-1);
						CEB_expand_flag = 32;
						continue;							
					}
				}
				if (Q.repeat_state[vertex])
					Q.repeat[embedding[depth]] = 32;		
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

	unsigned* res = new unsigned[limited_output_num*96];
	for (unsigned i =0; i < limited_output_num*96; ++i)
		res[i] = 0xffffffff;
	unsigned** index;
	vector<unsigned> *leaf_res;
	vector<unsigned> common_leaf_res;

    //data preprocessing
    Graph data_graph;
	gettimeofday(&begin_tv, NULL);
	data_graph.read(data_filename);
#ifdef hybrid_opt
		data_graph.setBloomFilter();
#endif
	gettimeofday(&end_tv, NULL);
	double read_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
    cout << "data_graph = " << data_filename << endl;
    cout << "label_num = " << data_graph.label_num << endl;
    cout << "vertex_num = " << data_graph.vertex_num << endl;
    cout << "edge_num = " << data_graph.edge_num << endl;
    cout << "max_degree = " << data_graph.max_degree << endl;
    cout << "average_degree = " << data_graph.average_degree << endl;
	cout << "read_time = " << read_time << endl;

    //query preprocessing
	Query query_graph;
	gettimeofday(&begin_tv, NULL);
	query_graph.read(query_filename);
    query_graph.CFL_decomposition();
	gettimeofday(&end_tv, NULL);
	read_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
    cout << "query_graph = " << query_filename << endl;
	cout << "query_state = " << query_graph.query_state << endl;
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
#ifdef order_opt
	qcsr.build(data_graph, query_graph, true);
#else
	qcsr.build(data_graph, query_graph, false);
#endif
	gettimeofday(&end_tv, NULL);
	double build_qcsr_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
	cout << "build_qcsr_time = " << build_qcsr_time << endl;

	//match
	unsigned res_iter = 0;
	unsigned res_num = 0;
	unsigned v, vv;
	unsigned idx;
	unsigned temp_idx;
	unsigned depth;
	unsigned father; //parent vertex
	unsigned CEB_size; //the size of CEB
	if (query_graph.query_state != 2)
	{
		for (unsigned i = 1; i < query_graph.match_order.size(); ++i)
		{
			v = query_graph.match_order[i];
			idx = query_graph.backward_neighbor[v].back();
			if (idx != (i-1))
			{
				depth = i;
				while (true)
				{
					depth++;
					if (depth == query_graph.match_order.size())
					{
						query_graph.CEB_width[v] = depth - i;
						break;
					}
					vv = query_graph.match_order[depth];
					if ((query_graph.backward_neighbor[vv][0] < i)||(query_graph.backward_neighbor[vv].back() != depth-1))
					{
						query_graph.CEB_width[v] = depth-i;
						break;
					}
				}
				query_graph.CEB_flag[v] = true;
				if (query_graph.CEB_width[v] > 1)
				{
					query_graph.CEB_write[query_graph.match_order[depth-1]] = v;
					CEB_size = 1;
					for (unsigned j = i; j < depth; ++j)
					{
						vv = query_graph.match_order[j];
						CEB_size *= qcsr.max_mapping_width[vv];
						for (unsigned k = 0; k < query_graph.repeated_vertex_set[vv].size(); ++k)
						{
							temp_idx = query_graph.repeated_vertex_set[vv][k];
							if ((temp_idx > idx)&&(temp_idx < i))
								idx = temp_idx;
						}
					}
					CEB_size *= query_graph.CEB_width[v];
					query_graph.CEB[v] = new unsigned[CEB_size];
				}
				else
				{
					for (unsigned k = 0; k < query_graph.repeated_vertex_set[v].size(); ++k)
					{
						temp_idx = query_graph.repeated_vertex_set[v][k];
						if ((temp_idx > idx)&&(temp_idx < i))
							idx = temp_idx;
					}				
				}
				query_graph.CEB_father_idx[v] = idx;
				father = query_graph.match_order[idx];
				query_graph.parent[father].push_back(v);
			}
			else
				query_graph.CEB_father_idx[v] = idx;
		}
	}

	gettimeofday(&begin_tv, NULL);
    if (query_graph.query_state == 0) // core
    {
		core_match(data_graph, query_graph, qcsr, res, res_iter, res_num, limited_output_num);
    }
    else if (query_graph.query_state == 2) // tree + leaf
    {
		for (unsigned i = 0; i < query_graph.repeat_two_leaf_begin_loc; ++i)
		{
			v = query_graph.leaf[i];
			idx = query_graph.backward_neighbor[v][0];
			for (unsigned k = 0; k < query_graph.repeated_vertex_set[v].size(); ++k)
			{
				temp_idx = query_graph.repeated_vertex_set[v][k];
				if ((temp_idx > idx)&&(temp_idx < query_graph.tree_match_order.size()))
					idx = temp_idx;
			}
			query_graph.CEB_father_idx[v] = idx;
		}
        tl_match(data_graph, query_graph, qcsr, res, res_iter, index, leaf_res, common_leaf_res, res_num, limited_output_num);
    }
    else // core + leaf, core + forest + leaf
    {
		for (unsigned i = 0; i < query_graph.repeat_two_leaf_begin_loc; ++i)
		{
			v = query_graph.leaf[i];
			idx = query_graph.backward_neighbor[v][0];
			for (unsigned k = 0; k < query_graph.repeated_vertex_set[v].size(); ++k)
			{
				temp_idx = query_graph.repeated_vertex_set[v][k];
				if ((temp_idx > idx)&&(temp_idx < query_graph.match_order.size()))
					idx = temp_idx;
			}
			query_graph.CEB_father_idx[v] = idx;
		}
		cfl_match(data_graph, query_graph, qcsr, res, res_iter, index, leaf_res, common_leaf_res, res_num, limited_output_num);
    }
	gettimeofday(&end_tv, NULL);
	double match_time = (double)(end_tv.tv_sec - begin_tv.tv_sec) * 1000.0 + (double)(end_tv.tv_usec - begin_tv.tv_usec) / 1000.0;
	cout << "match_time = " << match_time << endl;
	cout << "embeddings = " << res_num << endl;
	double total_time = read_time + get_start_vertex_time + build_qcsr_time + match_time;
	cout << "total_time = " << total_time << endl;
	return 0;
}
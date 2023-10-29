#include "Query.h"

Query::Query()
{
}

Query::~Query()
{
	delete[] this->label_list;
	delete[] this->edge_relation;
    delete[] this->neighbor_list;
	delete[] this->neighbor_label_frequency;
	delete[] this->backward_neighbor;
	delete[] this->forward_neighbor; 
	delete[] this->repeated_list; 
	delete[] this->repeated_vertex_set; 
	delete[] this->vertex_state; 
	delete[] this->forest_match_order;
	delete[] this->leaf_num;
	delete[] this->CEB_flag;
	delete[] this->CEB_update; 
	delete[] this->CEB_width; 
	delete[] this->CEB_father_idx; 
	delete[] this->CEB_write; 
	delete[] this->CEB_iter; 
	delete[] this->parent; 
	delete[] this->repeat;
	delete[] this->repeat_state;
	delete[] this->NEC;
	delete[] this->score;
}

void Query::read(const string& filename) // read query graph
{
	char type;
	unsigned vid;
	unsigned src_vid;
	unsigned dst_vid;
	unsigned label;
	ifstream fin(filename.c_str());
	while (fin>>type)
	{
		if (type == 't')
		{
			fin >> this->vertex_num;
			fin >> this->edge_num;
			this->label_list = new unsigned[this->vertex_num];
			this->edge_relation = new unsigned[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
				this->edge_relation[i] = 0;
			this->neighbor_list = new vector<unsigned>[this->vertex_num];
			this->neighbor_label_frequency = new unordered_map<unsigned, unsigned>[this->vertex_num];
			this->backward_neighbor = new vector<unsigned>[this->vertex_num];
			this->forward_neighbor = new vector<unsigned>[this->vertex_num];
			this->vertex_state = new uint16_t[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
				this->vertex_state[i] = 0;
			this->forest_match_order = new vector<unsigned>[this->vertex_num];
			this->leaf_num = new unsigned[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
				this->leaf_num[i] = 0;
			this->score = new double[this->vertex_num];
			this->CEB_flag = new bool[this->vertex_num];
			this->CEB_update = new bool[this->vertex_num];
			this->CEB_width = new unsigned[this->vertex_num];
			this->CEB_father_idx = new unsigned[this->vertex_num];
			this->CEB_write = new unsigned[this->vertex_num];
			this->CEB = new unsigned*[this->vertex_num];
			this->CEB_iter = new unsigned[this->vertex_num];
			this->repeat_state = new bool[this->vertex_num];
			this->NEC = new unsigned[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
			{
				this->CEB_flag[i] = false;
				this->CEB_update[i] = false;
				this->CEB_width[i] = 0;
				this->CEB_write[i] = 32;
				this->CEB_iter[i] = 0;
				this->NEC[i] = 0;
			}			
			this->parent = new vector<unsigned>[this->vertex_num];
		}
		else if (type == 'v')
		{
			fin >> vid >> label;
			this->label_list[vid] = label;
		}
		else if (type == 'e')
		{
			fin >> src_vid >> dst_vid >> label;
			this->neighbor_list[src_vid].push_back(dst_vid); 
			if (this->neighbor_label_frequency[src_vid].find(this->label_list[dst_vid]) == this->neighbor_label_frequency[src_vid].end())
				this->neighbor_label_frequency[src_vid][this->label_list[dst_vid]] = 1;
			else
				this->neighbor_label_frequency[src_vid][this->label_list[dst_vid]]++;
			this->neighbor_list[dst_vid].push_back(src_vid);
			if (this->neighbor_label_frequency[dst_vid].find(this->label_list[src_vid]) == this->neighbor_label_frequency[dst_vid].end())
				this->neighbor_label_frequency[dst_vid][this->label_list[src_vid]] = 1;
			else
				this->neighbor_label_frequency[dst_vid][this->label_list[src_vid]]++;
			this->edge_relation[src_vid] = this->edge_relation[src_vid] | (1 << dst_vid);
			this->edge_relation[dst_vid] = this->edge_relation[dst_vid] | (1 << src_vid);
		}
	}
	fin.close();
}

void Query::CFL_decomposition() // CFL_decomposition
{
	queue<unsigned> q;
	unsigned* degree = new unsigned[this->vertex_num];
	for (unsigned i = 0; i < this->vertex_num; ++i)
	{
		degree[i] = this->neighbor_list[i].size();
		if (degree[i] == 1)
		{
			q.push(i);
			this->vertex_state[i] = 2;
		}
	}

	unsigned top;
	unsigned father;
	while (true)
	{
		if (q.empty())
			break;
		top = q.front();
		if (this->vertex_state[top] == 2)
		{
			father = this->neighbor_list[top][0];
			this->leaf_num[father]++;
			degree[father]--;
			if (degree[father] == 1)
			{
				this->vertex_state[father] = 1;
				q.push(father);
			}
			q.pop();
		}
		else if(this->vertex_state[top] == 1)
		{
			for (unsigned i = 0; i < this->neighbor_list[top].size(); ++i)
			{
				father = this->neighbor_list[top][i];
				if (this->vertex_state[father] == 0)
				{
					degree[father]--;
					if (degree[father] == 1)
					{
						this->vertex_state[father] = 1;
						q.push(father);
					}
					break;
				}
			}
			q.pop();
		}
	}
	for (unsigned i = 0; i < this->vertex_num; ++i)
	{
		if (this->vertex_state[i] == 0)
			this->core.push_back(i);
		else if (this->vertex_state[i] == 2)
			this->leaf.push_back(i);
	}

	if (this->core.size() == this->vertex_num) //only core
	{
		this->query_state = 0;
		return;
	}
	
	if ((this->core.size() + this->leaf.size()) == this->vertex_num) //core + leaf
	{
		this->query_state = 1;
		return;
	}

	if (this->core.size() == 0) //tree + leaf
	{
		for (unsigned i = 0; i < this->vertex_num; ++i)
		{
			if (this->vertex_state[i] == 1)
				this->tree.push_back(i);
		}
		this->query_state = 2;
		return;
	}

	// core + forest + leaf
	this->query_state = 3;
	return;
}

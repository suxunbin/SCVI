#include "Query.h"

Query::Query()
{
}

Query::~Query()
{
	delete[] this->label_list;
	for (unsigned i = 0; i < this->vertex_num; ++i)
		delete[] this->edge_relation[i];
    delete[] this->neighbor_list;
	delete[] this->neighbor_label_frequency;
	delete[] this->backward_neighbor_set; 
	delete[] this->forward_neighbor_set; 
	delete[] this->repeated_list; 
	delete[] this->repeated_vertex_set;
	delete[] this->vertex_state; 
	delete[] this->leaf_num; 
	delete[] this->CEB_flag; 
	delete[] this->CEB_valid; 
	delete[] this->CEB_iter; 
	delete[] this->children; 
	delete[] this->encoding; 
	delete[] this->score;
}

void Query::read(const string& filename)
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
			this->edge_relation = new bool*[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
			{
				this->edge_relation[i] = new bool[this->vertex_num];
				for (unsigned j = 0; j < this->vertex_num; ++j)
					this->edge_relation[i][j] = false;
			}
			this->neighbor_list = new vector<unsigned>[this->vertex_num];
			this->neighbor_label_frequency = new unordered_map<unsigned, unsigned>[this->vertex_num];
			this->backward_neighbor_set = new vector<unsigned>[this->vertex_num];
			this->forward_neighbor_set = new vector<unsigned>[this->vertex_num];
			this->black_backward_neighbor = new vector<unsigned>[this->vertex_num]; 
			this->white_backward_neighbor = new vector<unsigned>[this->vertex_num]; 
			this->vertex_state = new uint16_t[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
				this->vertex_state[i] = 0;
			this->leaf_num = new unsigned[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
				this->leaf_num[i] = 0;
			this->CEB_flag = new bool[this->vertex_num];
			this->CEB_valid = new bool[this->vertex_num];
			this->CEB = new unsigned*[this->vertex_num];
			this->CEB_iter = new unsigned[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
			{
				this->CEB_flag[i] = false;
				this->CEB_valid[i] = false;
				this->CEB_iter[i] = 0;
			}			
			this->children = new vector<unsigned>[this->vertex_num];
			this->encoding = new bool[this->vertex_num];
			for (unsigned i = 0; i < this->vertex_num; ++i)
				this->encoding[i] = true;
			this->score = new double[this->vertex_num];
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
			this->edge_relation[src_vid][dst_vid] = true;
			this->edge_relation[dst_vid][src_vid] = true;
		}
	}
	fin.close();
}

void Query::CFL_decomposition()
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
		if (this->vertex_state[i] == 2)
			this->leaf.push_back(i);
	}
	return;
}

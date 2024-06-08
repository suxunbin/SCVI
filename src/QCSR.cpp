#include "QCSR.h"

QCSR::QCSR(Graph& G, Query& Q)
{
    this->query_vertex_num = Q.vertex_num;
    this->data_vertex_num = G.vertex_num;
    this->offset_list = new unordered_map<unsigned, pair<unsigned, unsigned>>*[Q.vertex_num];
    for (unsigned i = 0; i < Q.vertex_num; ++i)
        this->offset_list[i] = new unordered_map<unsigned, pair<unsigned, unsigned>>[Q.vertex_num];

	this->candidate_set = new unsigned*[Q.vertex_num];
    this->candidate_size = new unsigned[Q.vertex_num];
    this->candidate_flag = new unsigned*[Q.vertex_num];
    //this->candidate_frequency = new unsigned*[Q.vertex_num];
    //this->candidate_workload = new double*[Q.vertex_num];
    //this->max_mapping_width = new unsigned[Q.vertex_num];
    unsigned cnt;
    for (unsigned i = 0; i < Q.vertex_num; ++i)
    {
        cnt = G.label_frequency[Q.label_list[i]];
        this->candidate_set[i] = new unsigned[cnt];
        this->candidate_size[i] = 0;
        this->candidate_flag[i] = new unsigned[cnt];
        //this->candidate_frequency[i] = new unsigned[cnt];
        //this->candidate_workload[i] = new double[cnt];
        for (unsigned j = 0; j < cnt; ++j)
        {
            this->candidate_flag[i][j] = 0;
            //this->candidate_frequency[i][j] = 0;
            //this->candidate_workload[i][j] = 1.0;
        }
        //this->max_mapping_width[i] = 0;
    }
    Q.repeated_list = new vector<unsigned>[G.label_num];
    Q.repeated_vertex_set = new vector<unsigned>[Q.vertex_num];
}

QCSR::~QCSR()
{
    for (unsigned i = 0; i < this->query_vertex_num; ++i)
    {
        delete[] this->offset_list[i];
        delete[] this->candidate_set[i];
	    delete[] this->candidate_flag[i];
        //delete[] this->candidate_frequency[i];
        //delete[] this->candidate_workload[i];
    }
	delete[] this->candidate_size;
    //delete[] this->max_mapping_width;
}

void QCSR::build(Graph& G, Query& Q)
{
    this->start_vertex = Q.start_vertex;
    this->start_vertex_label = Q.label_list[Q.start_vertex];
    this->start_vertex_offset = G.label_index_list[this->start_vertex_label];
    unsigned start_vertex_nlabel = Q.start_vertex_nlabel;
    unsigned v, neighbor_v;
    unsigned label, nlabel;
    unsigned neighbor, forward_neighbor, backward_neighbor;
    bool sflag, flag;
    unsigned begin_loc, end_loc;
    unsigned len;
    unsigned offset = 0;
    double min_score;
    vector<bool> vis(Q.vertex_num); 
    vector<bool> vcm(Q.vertex_num); 
    vector<bool> vce(Q.vertex_num); 
    set<unsigned> next_match_list; 
    double *new_score = new double[Q.vertex_num];   
    for (unsigned i = 0; i < Q.vertex_num; ++i)
        new_score[i] = 0;
    unsigned vertex, next_vertex, pre_vertex; 
    unsigned next_vertex_label, pre_vertex_label; 
    unsigned next_vertex_offset, pre_vertex_offset, vertex_offset; 
    unsigned size; 
    unsigned idx; 
    unsigned state;
    unsigned backward_num; 

    Q.match_order.push_back(Q.start_vertex); 
    Q.repeated_list[start_vertex_label].push_back(0);
    vis[Q.start_vertex] = true; 

    for (unsigned i = 0; i < Q.neighbor_list[Q.start_vertex].size(); ++i)
    {
        neighbor = Q.neighbor_list[Q.start_vertex][i];
        Q.backward_neighbor_set[neighbor].push_back(0); 
        if (Q.vertex_state[neighbor] != 2) 
        {
            Q.forward_neighbor_set[Q.start_vertex].push_back(neighbor); 
            if ((Q.backward_neighbor_set[neighbor].size() + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size()) 
                vce[neighbor] = true;
            next_match_list.insert(neighbor); 
            vcm[neighbor] = true;
        }
    }

    for (unsigned i = 0; i < G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label].size(); ++i)
    {
        v = G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label][i];

        sflag = true;
        for (unordered_map<unsigned, unsigned>::iterator j = Q.neighbor_label_frequency[Q.start_vertex].begin(); j != Q.neighbor_label_frequency[Q.start_vertex].end(); ++j)
        {  
            label = (*j).first;
            if (G.offset_list[v*G.label_num+label+1]-G.offset_list[v*G.label_num+label] < (*j).second)
            {
                sflag = false;
                break;
            }
        }
        if (!sflag)  
            continue;

        for (unsigned j = 0; j < Q.forward_neighbor_set[Q.start_vertex].size(); ++j)
        {
            forward_neighbor = Q.forward_neighbor_set[Q.start_vertex][j]; 
            label = Q.label_list[forward_neighbor]; 
            vertex_offset = G.label_index_list[label]; 
            len = 0; 
            begin_loc = G.offset_list[v*G.label_num+label];
            end_loc = G.offset_list[v*G.label_num+label+1];
            for (unsigned k = begin_loc; k < end_loc; ++k)
            {
                neighbor_v = G.adjacency_list[k]; 
                
                if (this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] == -1)
                    continue;
                if (this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] == 0)
                {
                    
                    flag = true;
                    for (unordered_map<unsigned, unsigned>::iterator t = Q.neighbor_label_frequency[forward_neighbor].begin(); t != Q.neighbor_label_frequency[forward_neighbor].end(); ++t)
                    {  
                        nlabel = (*t).first;
                        if (G.offset_list[neighbor_v*G.label_num+nlabel+1]-G.offset_list[neighbor_v*G.label_num+nlabel] < (*t).second)
                        {
                            flag = false;
                            break;
                        }
                    }
                    if (!flag)
                    {
                        this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] = -1;
                        continue; 
                    }
                }

               
                if (this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] == 0) 
                {
                    if (!vce[forward_neighbor]) 
                    {
                        this->candidate_set[forward_neighbor][this->candidate_size[forward_neighbor]] = neighbor_v;
                        this->candidate_size[forward_neighbor]++;
                    }
                    this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset]++;
                }
                len++;
                //this->candidate_frequency[neighbor][neighbor_v-vertex_offset]++;
            }
            if (len != 0)
            {
                new_score[forward_neighbor] += len;
                //this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
            }
            else
            {
                sflag = false;
                break;
            }
        }

        if (sflag)  
        {
            this->candidate_set[Q.start_vertex][this->candidate_size[Q.start_vertex]] = v;
            //this->candidate_frequency[Q.start_vertex][v-this->start_vertex_offset]++;
            this->candidate_size[Q.start_vertex]++;
        }        
    }

    for (unsigned i = 0; i < Q.forward_neighbor_set[Q.start_vertex].size(); ++i)
    {
        forward_neighbor = Q.forward_neighbor_set[Q.start_vertex][i];
        Q.score[forward_neighbor] = min(Q.score[forward_neighbor], new_score[forward_neighbor] / (double)Q.neighbor_list[forward_neighbor].size()); 
    }

  
    while (true)
    {
        if (next_match_list.empty())
            break; 

       
        min_score = 2.0E+300;
        pre_vertex = Q.match_order.back();
        state = 2;
        for (set<unsigned>::iterator i = next_match_list.begin(); i != next_match_list.end(); ++i)
        {
            if (Q.vertex_state[*i] < state)
            {
                min_score = Q.score[*i];
                next_vertex = *i;
                state = Q.vertex_state[*i];
            }
            else if (Q.vertex_state[*i] == state)
            {
                if (Q.score[*i] < min_score)
                {
                    min_score = Q.score[*i];
                    next_vertex = *i;
                }
            }
        }
        next_vertex_label = Q.label_list[next_vertex];
        next_vertex_offset = G.label_index_list[next_vertex_label];
       
        if (Q.edge_relation[pre_vertex][next_vertex])
            Q.encoding[pre_vertex] = true;  
        else
        {
            Q.encoding[pre_vertex] = false; 
            Q.CEB_flag[next_vertex] = true; 
        }

        Q.match_order.push_back(next_vertex); 
        for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
            Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
        Q.repeated_list[next_vertex_label].push_back(Q.match_order.size()-1);
        next_match_list.erase(next_vertex); 
        vis[next_vertex] = true;     

        
        for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
        {
            neighbor = Q.neighbor_list[next_vertex][i];
            if (!vis[neighbor])
            {
                Q.backward_neighbor_set[neighbor].push_back(Q.match_order.size()-1); 
                if (Q.vertex_state[neighbor] != 2)
                {
                    Q.forward_neighbor_set[next_vertex].push_back(neighbor); 
                    if ((Q.backward_neighbor_set[neighbor].size() + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size()) 
                        vce[neighbor] = true;
                    if (vcm[neighbor])
                    {
                        this->candidate_size[neighbor] = 0;
                        new_score[neighbor] = 0;
                    }
                    else
                    {
                        next_match_list.insert(neighbor); 
                        vcm[neighbor] = true;   
                    }     
                }
            }
        }

        if (vce[next_vertex]) 
            continue;        

        size = this->candidate_size[next_vertex];
        this->candidate_size[next_vertex] = 0;
        
        for (unsigned i = 0; i < size; ++i)
        {
            v = this->candidate_set[next_vertex][i]; 
            sflag = true;

            
            for (unsigned j = 0; j < Q.forward_neighbor_set[next_vertex].size(); ++j)
            {
                forward_neighbor = Q.forward_neighbor_set[next_vertex][j]; 
                label = Q.label_list[forward_neighbor]; 
                vertex_offset = G.label_index_list[label]; 
                backward_num = Q.backward_neighbor_set[forward_neighbor].size()-1;
                len = 0; 
                begin_loc = G.offset_list[v*G.label_num+label];
                end_loc = G.offset_list[v*G.label_num+label+1];
                for (unsigned k = begin_loc; k < end_loc; ++k)
                {
                    neighbor_v = G.adjacency_list[k]; 
                    
                    if (this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] < backward_num)
                        continue;
                    if (this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] == backward_num)
                    {
                        if (backward_num == 0)
                        {
                            flag = true;
                            for (unordered_map<unsigned, unsigned>::iterator t = Q.neighbor_label_frequency[forward_neighbor].begin(); t != Q.neighbor_label_frequency[forward_neighbor].end(); ++t)
                            {
                                nlabel = (*t).first;
                                if (G.offset_list[neighbor_v*G.label_num+nlabel+1]-G.offset_list[neighbor_v*G.label_num+nlabel] < (*t).second)
                                {
                                    flag = false;
                                    break;
                                }
                            }
                            if (!flag)
                            {
                                this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] = -1;
                                continue; 
                            }
                        }
                    }

                   
                    if (this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] == backward_num) 
                    {
                        if (!vce[forward_neighbor]) 
                        {
                            this->candidate_set[forward_neighbor][this->candidate_size[forward_neighbor]] = neighbor_v;
                            this->candidate_size[forward_neighbor]++;                               
                        }
                        this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset]++;
                    }
                    len++;
                }
                if (len != 0)
                {
                    new_score[forward_neighbor] += len;
                    //this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                }
                else
                {
                    sflag = false;
                    break;
                }                  
            }
            if (sflag)  
            {
                this->candidate_set[next_vertex][this->candidate_size[next_vertex]] = v;
                this->candidate_size[next_vertex]++;
            }
            else
                this->candidate_flag[next_vertex][v-next_vertex_offset] = -1;                
        } 

        for (unsigned i = 0; i < Q.forward_neighbor_set[next_vertex].size(); ++i)
        {
            forward_neighbor = Q.forward_neighbor_set[next_vertex][i];
            Q.score[forward_neighbor] = min(Q.score[forward_neighbor], new_score[forward_neighbor] / (double)Q.neighbor_list[forward_neighbor].size()); 
        }
    }
    Q.encoding[Q.match_order.back()] = false; 

    
    for (unsigned it = Q.match_order.size(); it > 0;)
    {
        --it;
        pre_vertex = Q.match_order[it];
        if (Q.forward_neighbor_set[pre_vertex].size() == 0)
            continue; 
        else
        {
            pre_vertex_label = Q.label_list[pre_vertex];
            pre_vertex_offset = G.label_index_list[pre_vertex_label];
            for (unsigned i = 0; i < this->candidate_size[pre_vertex]; ++i)
            {
                v = this->candidate_set[pre_vertex][i];
                sflag = true;
                for (unsigned j = 0; j < Q.forward_neighbor_set[pre_vertex].size(); ++j)
                {
                    forward_neighbor = Q.forward_neighbor_set[pre_vertex][j];
                    label = Q.label_list[forward_neighbor];    
                    vertex_offset = G.label_index_list[label];
                    backward_num = Q.backward_neighbor_set[forward_neighbor].size();
                    len = 0;
                    begin_loc = G.offset_list[v*G.label_num+label];
                    end_loc = G.offset_list[v*G.label_num+label+1];
                    for (unsigned k = begin_loc; k < end_loc; ++k)
                    {
                        neighbor_v = G.adjacency_list[k]; 
                        if (this->candidate_flag[forward_neighbor][neighbor_v-vertex_offset] == backward_num)
                        {
                            this->adjacency_list.push_back(neighbor_v);
                            len++;
                        }
                    }
                    if (len != 0)
                    {
                        offset_list[pre_vertex][forward_neighbor][v] = pair<unsigned, unsigned>(offset, len);
                        offset += len;
                    }
                    else
                    {
                        sflag = false;
                        break;
                    }
                }
                if (!sflag)  
                {
                    this->candidate_flag[pre_vertex][v-pre_vertex_offset] = -1;
                    continue;
                }
                for (unsigned j = 0; j < Q.backward_neighbor_set[pre_vertex].size(); ++j)
                {
                    backward_neighbor = Q.match_order[Q.backward_neighbor_set[pre_vertex][j]];
                    if (Q.encoding[backward_neighbor] == false)
                    {
                        label = Q.label_list[backward_neighbor];    
                        vertex_offset = G.label_index_list[label];
                        backward_num = Q.backward_neighbor_set[backward_neighbor].size();
                        len = 0;
                        begin_loc = G.offset_list[v*G.label_num+label];
                        end_loc = G.offset_list[v*G.label_num+label+1];      
                        for (unsigned k = begin_loc; k < end_loc; ++k)
                        {
                            neighbor_v = G.adjacency_list[k]; 
                            if (this->candidate_flag[backward_neighbor][neighbor_v-vertex_offset] == backward_num)
                            {
                                this->adjacency_list.push_back(neighbor_v);
                                len++;
                            }
                        }      
                        if (len != 0)
                        {
                            offset_list[pre_vertex][backward_neighbor][v] = pair<unsigned, unsigned>(offset, len);
                            offset += len;
                        }
                        else
                        {
                            sflag = false;
                            break;
                        }            
                    }
                }
                if (!sflag)  
                {
                    this->candidate_flag[pre_vertex][v-pre_vertex_offset] = -1;
                    continue;
                }
            }
        }
    }

    idx = Q.match_order.size();
    
    for (unsigned i = 0; i < Q.leaf.size(); ++i)
    {
        v = Q.leaf[i];
        label = Q.label_list[v];
        for (unsigned j = 0; j < Q.repeated_list[label].size(); ++j)
            Q.repeated_vertex_set[v].push_back(Q.repeated_list[label][j]);
        Q.repeated_list[label].push_back(idx);
        idx++;
    } 

    for (unsigned i = 0; i < Q.match_order.size(); ++i)
    {
        v = Q.match_order[i];
        for (unsigned j = 0; j < Q.backward_neighbor_set[v].size(); ++j)
        {
            backward_neighbor = Q.match_order[Q.backward_neighbor_set[v][j]];
            if (Q.encoding[backward_neighbor])
                Q.black_backward_neighbor[v].push_back(Q.backward_neighbor_set[v][j]);
            else
                Q.white_backward_neighbor[v].push_back(Q.backward_neighbor_set[v][j]);
        }
    }

    for (unsigned i = 0; i < Q.match_order.size(); ++i)
    {
        v = Q.match_order[i];
        if (!Q.encoding[v])
            Q.white_vertex_set.push_back(i);
    }

    for (unsigned i = 0; i < Q.vertex_num; ++i)
    {
        if (Q.CEB_flag[i])
        {
            unsigned parent = Q.match_order[Q.backward_neighbor_set[i].back()];
            Q.children[parent].push_back(i);
        }
    }
}

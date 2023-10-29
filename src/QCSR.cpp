#include "QCSR.h"

QCSR::QCSR(Graph& G, Query& Q)
{
    this->query_vertex_num = Q.vertex_num;
    this->data_vertex_num = G.vertex_num;
    this->vfs = new bool*[G.vertex_num];
    for (unsigned i = 0; i < G.vertex_num; ++i)
    {
        this->vfs[i] = new bool[Q.vertex_num];
        for (unsigned j = 0; j < Q.vertex_num; ++j)
            this->vfs[i][j] = false;
    }
    this->offset_list = new unsigned[G.vertex_num*Q.vertex_num*2];
    for (unsigned i = 0; i < G.vertex_num*Q.vertex_num*2; ++i)
        this->offset_list[i] = 0;

	this->candidate_vertex_set = new unsigned*[Q.vertex_num];
    this->candidate_size = new unsigned[Q.vertex_num];
    this->candidate_valid = new unsigned*[Q.vertex_num];
    this->candidate_frequency = new unsigned*[Q.vertex_num];
    this->candidate_workload = new double*[Q.vertex_num];
    this->max_mapping_width = new unsigned[Q.vertex_num];
    unsigned cnt;
    for (unsigned i = 0; i < Q.vertex_num; ++i)
    {
        cnt = G.label_frequency[Q.label_list[i]];
        this->candidate_vertex_set[i] = new unsigned[cnt];
        this->candidate_size[i] = 0;
        this->candidate_valid[i] = new unsigned[cnt];
        this->candidate_frequency[i] = new unsigned[cnt];
        this->candidate_workload[i] = new double[cnt];
        for (unsigned j = 0; j < cnt; ++j)
        {
            this->candidate_valid[i][j] = 0;
            this->candidate_frequency[i][j] = 0;
            this->candidate_workload[i][j] = 1.0;
        }
        this->max_mapping_width[i] = 0;
    }
    Q.repeated_list = new vector<unsigned>[G.label_num];
    Q.repeated_vertex_set = new vector<unsigned>[Q.vertex_num];
    Q.repeat = new unsigned[G.vertex_num];
    for (unsigned i = 0; i < G.vertex_num; ++i)
        Q.repeat[i] = 32;
}

QCSR::~QCSR()
{
    for (unsigned i = 0; i < this->data_vertex_num; ++i)
	    delete[] this->vfs[i];
	delete[] this->offset_list;
    for (unsigned i = 0; i < this->query_vertex_num; ++i)
    {
        delete[] this->candidate_vertex_set[i];
	    delete[] this->candidate_valid[i];
        delete[] this->candidate_frequency[i];
        delete[] this->candidate_workload[i];
    }
	delete[] this->candidate_size;
    delete[] this->max_mapping_width;
}

void QCSR::build(Graph& G, Query& Q, bool opt_flag)
{
    this->start_vertex = Q.start_vertex;
    this->start_vertex_label = Q.label_list[Q.start_vertex];
    this->start_vertex_offset = G.label_index_list[this->start_vertex_label];
    unsigned start_vertex_nlabel = Q.start_vertex_nlabel;
    unsigned v;
    unsigned label, nlabel;
    unsigned neighbor, neighbor_v;
    bool sflag, flag;
    unsigned begin_loc, end_loc;
    unsigned len;
    unsigned offset = 0;
    double min_score;
    unsigned cur = 0; //the current timestamp
    unsigned cur_NEC = 0; //the current equivalent vertex class
    vector<bool> vis(Q.vertex_num); //the flag denoting whether a vertex is handled
    vector<bool> vcm(Q.vertex_num); //the flag denoting whether a vertex can be matched
    vector<bool> vce(Q.vertex_num); //the flag denoting whether a vertex can be further extended. 0:can be extended，1:cannot be extended
    set<unsigned> next_match_list; 
    unsigned *timestamp = new unsigned[Q.vertex_num]; //record the timestamps of the final updating
    double *new_score = new double[Q.vertex_num];   //record the scores
    for (unsigned i = 0; i < Q.vertex_num; ++i)
    {
        timestamp[i] = 0;
        new_score[i] = 0;
    }
    unsigned next_vertex, pre_vertex;
    unsigned next_vertex_label, pre_vertex_label;
    unsigned next_vertex_offset, pre_vertex_offset, vertex_offset;
    unsigned *forward_neighbor = new unsigned[Q.vertex_num];
    unsigned forward_iter;
    unsigned size;
    unsigned idx;
    unsigned long long temp_workload;
    if (opt_flag)
    {
    if ((Q.query_state == 0) || (Q.query_state == 1)) // core or core + leaf
    {
        Q.core_match_order.push_back(Q.start_vertex);
        Q.repeated_list[start_vertex_label].push_back(0);
        vis[Q.start_vertex] = true;
        forward_iter = 0;
        for (unsigned i = 0; i < Q.neighbor_list[Q.start_vertex].size(); ++i)
        {
            neighbor = Q.neighbor_list[Q.start_vertex][i];
            Q.backward_neighbor[neighbor].push_back(0);
            if (Q.vertex_state[neighbor] == 0)
            {
                forward_neighbor[forward_iter++] = neighbor;
                Q.forward_neighbor[Q.start_vertex].push_back(neighbor);
                next_match_list.insert(neighbor);
                vcm[neighbor] = true;
            }
        }   

        //filter the neighbor list of the initial vertex
        for (unsigned i = 0; i < G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label].size(); ++i)
        {
            v = G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label][i];

            //NLF
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

            for (unsigned j = 0; j < forward_iter; ++j)
            {
                neighbor = forward_neighbor[j];
                label = Q.label_list[neighbor];
                vertex_offset = G.label_index_list[label];
                len = 0;
                begin_loc = G.offset_list[v*G.label_num+label];
                end_loc = G.offset_list[v*G.label_num+label+1];
                for (unsigned k = begin_loc; k < end_loc; ++k)
                {
                    neighbor_v = G.adjacency_list[k];
                    //0: not verified， 1：verified but invalid， 2：verified and valid
                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 1)
                        continue;
                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 0)
                    {
                        flag = true;
                        for (unordered_map<unsigned, unsigned>::iterator t = Q.neighbor_label_frequency[neighbor].begin(); t != Q.neighbor_label_frequency[neighbor].end(); ++t)
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
                            this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 1;
                            continue;
                        }
                    }

                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 0)
                    {
                        this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                        this->candidate_size[neighbor]++;
                        this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 2;
                    }
                    len++;
                    this->candidate_frequency[neighbor][neighbor_v-vertex_offset]++;
                }
                if (len != 0)
                {
                    new_score[neighbor] += len;
                    this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                }
                else
                {
                    sflag = false;
                    break;
                }
            }
            if (sflag)
            {
                this->candidate_vertex_set[Q.start_vertex][this->candidate_size[Q.start_vertex]] = v;
                this->candidate_frequency[Q.start_vertex][v-this->start_vertex_offset]++;
                this->candidate_size[Q.start_vertex]++;
            }
        }
        
        cur += 2;
        for (unsigned i = 0; i < forward_iter; ++i)
        {
            neighbor = forward_neighbor[i];
            timestamp[neighbor] = cur; //update the timestamps
            Q.score[neighbor] = new_score[neighbor] / (double)Q.neighbor_list[neighbor].size(); //update the scores
        }

        //filter the neighbor lists of the other vertices
        while (true)
        {
            if (next_match_list.empty())
                break;
            //find the next query vertex
            min_score = 2.0E+300;
            pre_vertex = Q.core_match_order.back();
            for (set<unsigned>::iterator i = next_match_list.begin(); i != next_match_list.end(); ++i)
            {
                if((Q.edge_relation[pre_vertex] == Q.edge_relation[*i])&&(Q.label_list[pre_vertex] == Q.label_list[*i]))
                {
                    next_vertex = *i;
                    if (Q.NEC[pre_vertex] == 0)
                    {
                        cur_NEC++;
                        Q.NEC[pre_vertex] = cur_NEC;
                    }
                    Q.NEC[*i] = cur_NEC;
                    break;
                }
                if (Q.score[*i] < min_score)
                {
                    min_score = Q.score[*i];
                    next_vertex = *i;
                }
            }
            next_vertex_label = Q.label_list[next_vertex];
            next_vertex_offset = G.label_index_list[next_vertex_label];

            Q.core_match_order.push_back(next_vertex);
            for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
                Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
            Q.repeated_list[next_vertex_label].push_back(Q.core_match_order.size()-1);
            next_match_list.erase(next_vertex);
            vis[next_vertex] = true;

            forward_iter = 0;
            for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
            {
                neighbor = Q.neighbor_list[next_vertex][i];
                if (!vis[neighbor])
                {
                    Q.backward_neighbor[neighbor].push_back(Q.core_match_order.size()-1);
                    if (Q.vertex_state[neighbor] == 0)
                    {
                        forward_neighbor[forward_iter++] = neighbor;
                        Q.forward_neighbor[next_vertex].push_back(neighbor);
                        if ((Q.backward_neighbor[neighbor].size() + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
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

            if (vce[next_vertex]) //If a vertex has no forward neighbor (except for leaf vertices), we do not deal with it.
                continue;      

            size = this->candidate_size[next_vertex];
            this->candidate_size[next_vertex] = 0;
            //filter the neighbor list of the next query vertex
            for (unsigned i = 0; i < size; ++i)
            {
                v = this->candidate_vertex_set[next_vertex][i];
                sflag = true;

                for (unsigned j = 0; j < forward_iter; ++j)
                {
                    neighbor = forward_neighbor[j];
                    label = Q.label_list[neighbor];
                    vertex_offset = G.label_index_list[label];
                    len = 0;
                    begin_loc = G.offset_list[v*G.label_num+label];
                    end_loc = G.offset_list[v*G.label_num+label+1];
                    for (unsigned k = begin_loc; k < end_loc; ++k)
                    {
                        neighbor_v = G.adjacency_list[k];
                        //timestamp[neighbor]: not verified， cur+1：verified but invalid， cur+2：verified and valid
                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] < timestamp[neighbor])
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == (cur+1))
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == timestamp[neighbor])
                        {
                            if (!vcm[neighbor])
                            {
                                flag = true;
                                for (unordered_map<unsigned, unsigned>::iterator t = Q.neighbor_label_frequency[neighbor].begin(); t != Q.neighbor_label_frequency[neighbor].end(); ++t)
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
                                    this->candidate_valid[neighbor][neighbor_v-vertex_offset] = cur+1;
                                    continue;
                                }

                            }
                        }

                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == timestamp[neighbor])
                        {
                            if (!vce[neighbor]) //if the neighbor can be further extended, then record its candidates
                            {
                                this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                                this->candidate_frequency[neighbor][neighbor_v-vertex_offset] = 0;
                                this->candidate_size[neighbor]++;                               
                            }
                            this->candidate_valid[neighbor][neighbor_v-vertex_offset] = cur+2;
                        }
                        len++;
                        if (!vce[neighbor])
                            this->candidate_frequency[neighbor][neighbor_v-vertex_offset] += this->candidate_frequency[next_vertex][v-next_vertex_offset];
                    }

                    if (len != 0)
                    {
                        new_score[neighbor] += (len*this->candidate_frequency[next_vertex][v-next_vertex_offset]);
                        this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                    }
                    else
                    {
                        sflag = false;
                        break;
                    }
                }
                if (sflag)
                {
                    this->candidate_vertex_set[next_vertex][this->candidate_size[next_vertex]] = v;
                    this->candidate_size[next_vertex]++;
                }
                else
                    this->candidate_valid[next_vertex][v-next_vertex_offset] = 1;
            }

            cur += 2;
            for (unsigned i = 0; i < forward_iter; ++i)
            {
                neighbor = forward_neighbor[i];
                timestamp[neighbor] = cur; //update the timestamps
                Q.score[neighbor] = new_score[neighbor] / (double)Q.neighbor_list[neighbor].size(); //update the scores
            }
        }

        for (unsigned it = Q.core_match_order.size(); it > 0;)
        {
            --it;
            pre_vertex = Q.core_match_order[it];
            if (Q.forward_neighbor[pre_vertex].size() == 0)
                continue;
            else
            {
                pre_vertex_label = Q.label_list[pre_vertex];
                pre_vertex_offset = G.label_index_list[pre_vertex_label];
                for (unsigned i = 0; i < this->candidate_size[pre_vertex]; ++i)
                {
                    v = this->candidate_vertex_set[pre_vertex][i];
                    sflag = true;
                    for (unsigned j = 0; j < Q.forward_neighbor[pre_vertex].size(); ++j)
                    {
                        temp_workload = 0;
                        neighbor = Q.forward_neighbor[pre_vertex][j];
                        label = Q.label_list[neighbor];    
                        vertex_offset = G.label_index_list[label];
                        if (!this->vfs[v][neighbor])
                        {
                            this->vfs[v][neighbor] = true;
                            begin_loc = G.offset_list[v*G.label_num+label];
                            end_loc = G.offset_list[v*G.label_num+label+1];
                            for (unsigned k = begin_loc; k < end_loc; ++k)
                            {
                                neighbor_v = G.adjacency_list[k];
                                if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == timestamp[neighbor])
                                {
                                    this->adjacency_list.push_back(neighbor_v);
                                    this->offset_list[v*Q.vertex_num*2+neighbor*2+1]++;
                                    if (Q.backward_neighbor[neighbor].back()==it)
                                        temp_workload += this->candidate_workload[neighbor][neighbor_v-vertex_offset]; //compute the workload from bottom to up
                                }
                            }
                            if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] != 0)
                            {
                                this->offset_list[v*Q.vertex_num*2+neighbor*2] = offset;
                                offset += this->offset_list[v*Q.vertex_num*2+neighbor*2+1];
                            }
                            else
                            {
                                sflag = false;
                                break;
                            }
                        }                    
                        else
                        {
                            if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] == 0)
                            {
                                sflag = false;
                                break;
                            }                              
                        }
                        if (Q.backward_neighbor[neighbor].back()==it)
                            this->candidate_workload[pre_vertex][v-pre_vertex_offset] *= temp_workload;
                    }
                    if (!sflag)  //filter
                    {
                        this->candidate_valid[pre_vertex][v-pre_vertex_offset] = 1;
                        this->candidate_workload[pre_vertex][v-pre_vertex_offset] = 0;
                    }
                }
            }
        }
        sort(this->candidate_vertex_set[Q.start_vertex], this->candidate_vertex_set[Q.start_vertex]+this->candidate_size[Q.start_vertex], 
        [this](const unsigned& x, const unsigned& y)->bool{return this->candidate_workload[this->start_vertex][x-this->start_vertex_offset] < this->candidate_workload[this->start_vertex][y-this->start_vertex_offset];});

        if (Q.leaf.size() == 0)
        {
            for (unsigned i = 0; i < Q.vertex_num; ++i)
            {
                label = Q.label_list[i];
                if (Q.repeated_list[label].size() > 1)
                    Q.repeat_state[i] = true;
                else
                    Q.repeat_state[i] = false;
            }

            for (unsigned i = 0; i < Q.core.size(); ++i)
            {
                v = Q.core[i];
                if (Q.backward_neighbor[v].size() > Q.max_degree)
                    Q.max_degree = Q.backward_neighbor[v].size();
            }   
            return;
        }
        unsigned repeat_flag;
        vector<unsigned> leaf_vec;
        vector<unsigned> rleaf_vec;
        vector<unsigned> rleaf_off;
        unsigned begin_iter, end_iter;
        offset = 0;
        rleaf_off.push_back(0);
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            if (vis[v])
                continue;
            label = Q.label_list[v];
            repeat_flag = false;
            for (unsigned j = i+1; j < Q.leaf.size(); ++j)
            {
                if (!vis[Q.leaf[j]])
                {
                    if (label == Q.label_list[Q.leaf[j]])
                    {
                        repeat_flag = true;
                        rleaf_vec.push_back(Q.leaf[j]);
                        offset++;
                        vis[Q.leaf[j]] = true;
                    }
                }
            }
            if (repeat_flag)
            {
                rleaf_vec.push_back(v);
                offset++;
                rleaf_off.push_back(offset);
            }
            else
                leaf_vec.push_back(v);
            vis[v] = true;                
        }
        for (unsigned i = 0; i < leaf_vec.size(); ++i)
            Q.leaf[i] = leaf_vec[i];
        Q.repeat_leaf_begin_loc = leaf_vec.size();
        begin_iter = Q.repeat_leaf_begin_loc;
        end_iter = Q.leaf.size() - 1;
        for (unsigned i = 0; i < rleaf_off.size()-1; ++i)
        {
            if (rleaf_off[i+1] - rleaf_off[i] > 2)
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[end_iter--] = rleaf_vec[j];
            }
            else
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[begin_iter++] = rleaf_vec[j];
            }
        }
        Q.repeat_two_leaf_begin_loc = begin_iter; 

        idx = Q.core_match_order.size();
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            label = Q.label_list[v];
            for (unsigned j = 0; j < Q.repeated_list[label].size(); ++j)
                Q.repeated_vertex_set[v].push_back(Q.repeated_list[label][j]);
            Q.repeated_list[label].push_back(idx);
            idx++;
        }
    }
    else if (Q.query_state == 2) // tree + leaf
    {
        Q.tree_match_order.push_back(Q.start_vertex);
        Q.repeated_list[start_vertex_label].push_back(0);
        vis[Q.start_vertex] = true;
        forward_iter = 0;
        for (unsigned i = 0; i < Q.neighbor_list[Q.start_vertex].size(); ++i)
        {
            neighbor = Q.neighbor_list[Q.start_vertex][i];
            Q.backward_neighbor[neighbor].push_back(0);
            if (Q.vertex_state[neighbor] == 1)
            {
                forward_neighbor[forward_iter++] = neighbor;
                Q.forward_neighbor[Q.start_vertex].push_back(neighbor);
                if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                    vce[neighbor] = true;
                next_match_list.insert(neighbor);
            }
        }   
                        
        //filter the neighbor list of the initial vertex
        for (unsigned i = 0; i < G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label].size(); ++i)
        {
            v = G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label][i];

            //NLF
            sflag = true;
            for (unsigned j = 0; j < Q.neighbor_list[Q.start_vertex].size(); ++j)
            {  
                label = Q.label_list[Q.neighbor_list[Q.start_vertex][j]];
                if (!G.nsa[v][label])
                {
                    sflag = false;
                    break;
                }
            }
            if (!sflag)
                continue;

            for (unsigned j = 0; j < forward_iter; ++j)
            {
                neighbor = forward_neighbor[j];
                label = Q.label_list[neighbor];
                vertex_offset = G.label_index_list[label];
                len = 0;
                begin_loc = G.offset_list[v*G.label_num+label];
                end_loc = G.offset_list[v*G.label_num+label+1];
                for (unsigned k = begin_loc; k < end_loc; ++k)
                {
                    neighbor_v = G.adjacency_list[k];
                    //0: not verified， 1：verified but invalid， 2：verified and valid
                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 1)
                        continue;
                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 0)
                    {
                        flag = true;
                        for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t)
                        {
                            nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                            if (!G.nsa[neighbor_v][nlabel])
                            {
                                flag = false;
                                break;
                            }
                        }
                        if (!flag)
                        {
                            this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 1;
                            continue;
                        }
                    }

                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 0)
                    {
                        if (!vce[neighbor]) //If the neighbor can be further extended, then record its candidates.
                        {
                            this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                            this->candidate_size[neighbor]++;                               
                        }
                        this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 2;
                    }
                    len++;
                    if (!vce[neighbor])
                        this->candidate_frequency[neighbor][neighbor_v-vertex_offset]++;                       
                }
                if (len != 0)
                    new_score[neighbor] += len;
                else
                {
                    sflag = false;
                    break;
                }                
            }
            if (sflag)
            {
                this->candidate_vertex_set[Q.start_vertex][this->candidate_size[Q.start_vertex]] = v;
                this->candidate_frequency[Q.start_vertex][v-this->start_vertex_offset]++;
                this->candidate_size[Q.start_vertex]++;
            }
        }
        
        for (unsigned i = 0; i < forward_iter; ++i)
        {
            neighbor = forward_neighbor[i];
            Q.score[neighbor] = new_score[neighbor] / (double)Q.neighbor_list[neighbor].size(); //update the scores
        }

        //filter the neighbor lists of the other vertices
        while (true)
        {
            if (next_match_list.empty())
                break;
            //find the next query vertex
            min_score = 2.0E+300;
            pre_vertex = Q.tree_match_order.back();
            for (set<unsigned>::iterator i = next_match_list.begin(); i != next_match_list.end(); ++i)
            {
                if (Q.score[*i] < min_score)
                {
                    min_score = Q.score[*i];
                    next_vertex = *i;
                }
            }
            next_vertex_label = Q.label_list[next_vertex];
            next_vertex_offset = G.label_index_list[next_vertex_label];

            Q.tree_match_order.push_back(next_vertex);
            for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
                Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
            Q.repeated_list[next_vertex_label].push_back(Q.tree_match_order.size()-1);
            next_match_list.erase(next_vertex);
            vis[next_vertex] = true;

            forward_iter = 0;
            for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
            {
                neighbor = Q.neighbor_list[next_vertex][i];
                if (!vis[neighbor])
                {
                    Q.backward_neighbor[neighbor].push_back(Q.tree_match_order.size()-1);
                    if (Q.vertex_state[neighbor] == 1)
                    {
                        forward_neighbor[forward_iter++] = neighbor;
                        Q.forward_neighbor[next_vertex].push_back(neighbor);
                        if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                            vce[neighbor] = true;
                        next_match_list.insert(neighbor);                      
                    }
                }   
            }            
            
            if (vce[next_vertex])
                continue;

            size = this->candidate_size[next_vertex];
            this->candidate_size[next_vertex] = 0;
            //filter the neighbor list of the next vertex
            for (unsigned i = 0; i < size; ++i)
            {
                v = this->candidate_vertex_set[next_vertex][i];
                sflag = true;

                for (unsigned j = 0; j < forward_iter; ++j)
                {
                    neighbor = forward_neighbor[j];
                    label = Q.label_list[neighbor];
                    vertex_offset = G.label_index_list[label];
                    len = 0;
                    begin_loc = G.offset_list[v*G.label_num+label];
                    end_loc = G.offset_list[v*G.label_num+label+1];
                    for (unsigned k = begin_loc; k < end_loc; ++k)
                    {
                        neighbor_v = G.adjacency_list[k];
                        //0: not verified， 1：verified but invalid， 2：verified and valid
                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 1)
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 0)
                        {
                            flag = true;
                            for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t)
                            {
                                nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                                if (!G.nsa[neighbor_v][nlabel])
                                {
                                    flag = false;
                                    break;
                                }
                            }
                            if (!flag)
                            {
                                this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 1;
                                continue;
                            }
                        }

                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 0)
                        {
                            if (!vce[neighbor])
                            {
                                this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                                this->candidate_frequency[neighbor][neighbor_v-vertex_offset] = 0;
                                this->candidate_size[neighbor]++;                               
                            }
                            this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 2;
                        }
                        len++;
                        if (!vce[neighbor])
                            this->candidate_frequency[neighbor][neighbor_v-vertex_offset] += this->candidate_frequency[next_vertex][v-next_vertex_offset];
                    }
                    if (len != 0)
                        new_score[neighbor] += (len*this->candidate_frequency[next_vertex][v-next_vertex_offset]);
                    else
                    {
                        sflag = false;
                        break;
                    }                    
                }
                if (sflag)
                {
                    this->candidate_vertex_set[next_vertex][this->candidate_size[next_vertex]] = v;
                    this->candidate_size[next_vertex]++;
                }
                else
                    this->candidate_valid[next_vertex][v-next_vertex_offset] = 1;                
            }

            for (unsigned i = 0; i < forward_iter; ++i)
            {
                neighbor = forward_neighbor[i];
                Q.score[neighbor] = new_score[neighbor] / (double)Q.neighbor_list[neighbor].size();
            }
        }

        for (unsigned it = Q.tree_match_order.size(); it > 0;)
        {
            --it;
            pre_vertex = Q.tree_match_order[it];
            if (Q.forward_neighbor[pre_vertex].size() == 0)
                continue;
            else
            {
                pre_vertex_label = Q.label_list[pre_vertex];
                pre_vertex_offset = G.label_index_list[pre_vertex_label];
                for (unsigned i = 0; i < this->candidate_size[pre_vertex]; ++i)
                {
                    v = this->candidate_vertex_set[pre_vertex][i];
                    sflag = true;
                    for (unsigned j = 0; j < Q.forward_neighbor[pre_vertex].size(); ++j)
                    {
                        neighbor = Q.forward_neighbor[pre_vertex][j];
                        label = Q.label_list[neighbor];    
                        vertex_offset = G.label_index_list[label];
                        begin_loc = G.offset_list[v*G.label_num+label];
                        end_loc = G.offset_list[v*G.label_num+label+1];
                        for (unsigned k = begin_loc; k < end_loc; ++k)
                        {
                            neighbor_v = G.adjacency_list[k];
                            if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 2)
                            {
                                this->adjacency_list.push_back(neighbor_v);
                                this->offset_list[v*Q.vertex_num*2+neighbor*2+1]++;
                            }
                        }
                        if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] != 0)
                        {
                            this->offset_list[v*Q.vertex_num*2+neighbor*2] = offset;
                            offset += this->offset_list[v*Q.vertex_num*2+neighbor*2+1];
                        }
                        else
                        {
                            sflag = false;
                            break;
                        }                   
                    }
                    if (!sflag)  //filter
                        this->candidate_valid[pre_vertex][v-pre_vertex_offset] = 1;
                }
            }
        }

        unsigned repeat_flag;
        vector<unsigned> leaf_vec;
        vector<unsigned> rleaf_vec;
        vector<unsigned> rleaf_off;
        unsigned begin_iter, end_iter;
        offset = 0;
        rleaf_off.push_back(0);
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            if (vis[v])
                continue;
            label = Q.label_list[v];
            repeat_flag = false;
            for (unsigned j = i+1; j < Q.leaf.size(); ++j)
            {
                if (!vis[Q.leaf[j]])
                {
                    if (label == Q.label_list[Q.leaf[j]])
                    {
                        repeat_flag = true;
                        rleaf_vec.push_back(Q.leaf[j]);
                        offset++;
                        vis[Q.leaf[j]] = true;
                    }
                }
            }
            if (repeat_flag)
            {
                rleaf_vec.push_back(v);
                offset++;
                rleaf_off.push_back(offset);
            }
            else
                leaf_vec.push_back(v);
            vis[v] = true;                
        }
        for (unsigned i = 0; i < leaf_vec.size(); ++i)
            Q.leaf[i] = leaf_vec[i];
        Q.repeat_leaf_begin_loc = leaf_vec.size();
        begin_iter = Q.repeat_leaf_begin_loc;
        end_iter = Q.leaf.size() - 1;
        for (unsigned i = 0; i < rleaf_off.size()-1; ++i)
        {
            if (rleaf_off[i+1] - rleaf_off[i] > 2)
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[end_iter--] = rleaf_vec[j];
            }
            else
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[begin_iter++] = rleaf_vec[j];
            }
        }
        Q.repeat_two_leaf_begin_loc = begin_iter; 

        idx = Q.tree_match_order.size();
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            label = Q.label_list[v];
            for (unsigned j = 0; j < Q.repeated_list[label].size(); ++j)
                Q.repeated_vertex_set[v].push_back(Q.repeated_list[label][j]);
            Q.repeated_list[label].push_back(idx);
            idx++;
        }       
    }
    else if (Q.query_state == 3) // core + tree + leaf
    {      
        set<unsigned> *next_match_tree = new set<unsigned>[Q.vertex_num];
        unsigned root;

        Q.core_match_order.push_back(Q.start_vertex);
        Q.match_order.push_back(Q.start_vertex);
        Q.repeated_list[start_vertex_label].push_back(0);
        vis[Q.start_vertex] = true;
        forward_iter = 0;
        flag = true;
        for (unsigned i = 0; i < Q.neighbor_list[Q.start_vertex].size(); ++i)
        {
            neighbor = Q.neighbor_list[Q.start_vertex][i];
            Q.backward_neighbor[neighbor].push_back(0);
            if (Q.vertex_state[neighbor] == 0)
            {
                forward_neighbor[forward_iter++] = neighbor;
                Q.forward_neighbor[Q.start_vertex].push_back(neighbor);
                next_match_list.insert(neighbor);
                vcm[neighbor] = true;
            }
            else if (Q.vertex_state[neighbor] == 1)
            {
                forward_neighbor[forward_iter++] = neighbor;
                Q.forward_neighbor[Q.start_vertex].push_back(neighbor);
                if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                    vce[neighbor] = true;
                next_match_tree[Q.start_vertex].insert(neighbor);
                if (flag)
                {
                    Q.forest.push_back(Q.start_vertex);
                    flag = false;
                }
            }
        }   

        for (unsigned i = 0; i < G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label].size(); ++i)
        {
            v = G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label][i];

            sflag = true;
            for (unsigned j = 0; j < Q.neighbor_list[Q.start_vertex].size(); ++j)
            {  
                label = Q.label_list[Q.neighbor_list[Q.start_vertex][j]];
                if (!G.nsa[v][label])
                {
                    sflag = false;
                    break;
                }
            }
            if (!sflag) 
                continue;

            for (unsigned j = 0; j < forward_iter; ++j)
            {
                neighbor = forward_neighbor[j];
                label = Q.label_list[neighbor];
                vertex_offset = G.label_index_list[label];
                len = 0;
                begin_loc = G.offset_list[v*G.label_num+label];
                end_loc = G.offset_list[v*G.label_num+label+1];
                for (unsigned k = begin_loc; k < end_loc; ++k)
                {
                    neighbor_v = G.adjacency_list[k];
                    //0: not verified， 1：verified but invalid， 2：verified and valid
                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 1)
                        continue;
                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 0)
                    {
                        flag = true;
                        for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t)
                        {
                            nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                            if (!G.nsa[neighbor_v][nlabel])
                            {
                                flag = false;
                                break;
                            }
                        }
                        if (!flag)
                        {
                            this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 1;
                            continue;
                        }
                    }

                    if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 0)
                    {
                        if (!vce[neighbor])
                        {
                            this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                            this->candidate_size[neighbor]++;                               
                        }
                        this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 2;
                    }
                    len++;
                    if (!vce[neighbor])
                        this->candidate_frequency[neighbor][neighbor_v-vertex_offset]++;
                }
                if (len != 0)
                {
                    new_score[neighbor] += len;
                    this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                }
                else
                {
                    sflag = false;
                    break;
                }                
            }
            if (sflag)
            {
                this->candidate_vertex_set[Q.start_vertex][this->candidate_size[Q.start_vertex]] = v;
                this->candidate_frequency[Q.start_vertex][v-this->start_vertex_offset]++;
                this->candidate_size[Q.start_vertex]++;
            }
        }
        
        cur += 2;
        for (unsigned i = 0; i < forward_iter; ++i)
        {
            neighbor = forward_neighbor[i];
            timestamp[neighbor] = cur; 
            Q.score[neighbor] = new_score[neighbor] / (double)Q.neighbor_list[neighbor].size();
        }

        //filter the neighbor list of the other core vertices
        while (true)
        {
            if (next_match_list.empty())
                break;
            //find the next query vertex
            min_score = 2.0E+300;
            pre_vertex = Q.core_match_order.back();
            for (set<unsigned>::iterator i = next_match_list.begin(); i != next_match_list.end(); ++i)
            {
                if((Q.edge_relation[pre_vertex] == Q.edge_relation[*i])&&(Q.label_list[pre_vertex] == Q.label_list[*i]))
                {
                    next_vertex = *i;
                    if (Q.NEC[pre_vertex] == 0)
                    {
                        cur_NEC++;
                        Q.NEC[pre_vertex] = cur_NEC;
                    }
                    Q.NEC[*i] = cur_NEC;
                    break;
                }
                if (Q.score[*i] < min_score)
                {
                    min_score = Q.score[*i];
                    next_vertex = *i;
                }
            }
            next_vertex_label = Q.label_list[next_vertex];
            next_vertex_offset = G.label_index_list[next_vertex_label];

            Q.core_match_order.push_back(next_vertex);
            Q.match_order.push_back(next_vertex);
            for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
                Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
            Q.repeated_list[next_vertex_label].push_back(Q.core_match_order.size()-1);
            next_match_list.erase(next_vertex);
            vis[next_vertex] = true;

            forward_iter = 0;
            flag = true;
            for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
            {
                neighbor = Q.neighbor_list[next_vertex][i];
                if (!vis[neighbor])
                {
                    Q.backward_neighbor[neighbor].push_back(Q.core_match_order.size()-1);
                    if (Q.vertex_state[neighbor] == 0)
                    {
                        forward_neighbor[forward_iter++] = neighbor;
                        Q.forward_neighbor[next_vertex].push_back(neighbor);
                        if ((Q.backward_neighbor[neighbor].size() + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
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
                    else if (Q.vertex_state[neighbor] == 1)
                    {
                        forward_neighbor[forward_iter++] = neighbor;
                        Q.forward_neighbor[next_vertex].push_back(neighbor);
                        if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                            vce[neighbor] = true;
                        next_match_tree[next_vertex].insert(neighbor);
                        if (flag)
                        {
                            Q.forest.push_back(next_vertex);
                            flag = false;
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
                v = this->candidate_vertex_set[next_vertex][i];
                sflag = true;

                for (unsigned j = 0; j < forward_iter; ++j)
                {
                    neighbor = forward_neighbor[j];
                    label = Q.label_list[neighbor];
                    vertex_offset = G.label_index_list[label];
                    len = 0;
                    begin_loc = G.offset_list[v*G.label_num+label];
                    end_loc = G.offset_list[v*G.label_num+label+1];
                    for (unsigned k = begin_loc; k < end_loc; ++k)
                    {
                        neighbor_v = G.adjacency_list[k];
                        //timestamp[neighbor]: not verified， cur+1：verified but invalid， cur+2：verified and valid
                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] < timestamp[neighbor])
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == (cur+1))
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == timestamp[neighbor])
                        {
                            if (!vcm[neighbor])
                            {
                                flag = true;
                                for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t)
                                {
                                    nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                                    if (!G.nsa[neighbor_v][nlabel])
                                    {
                                        flag = false;
                                        break;
                                    }
                                }
                                if (!flag)
                                {
                                    this->candidate_valid[neighbor][neighbor_v-vertex_offset] = cur+1;
                                    continue;
                                }
                            }
                        }

                        if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == timestamp[neighbor])
                        {
                            if (!vce[neighbor])
                            {
                                this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                                this->candidate_frequency[neighbor][neighbor_v-vertex_offset] = 0;
                                this->candidate_size[neighbor]++;                               
                            }
                            this->candidate_valid[neighbor][neighbor_v-vertex_offset] = cur+2;
                        }
                        len++;
                        if (!vce[neighbor])
                            this->candidate_frequency[neighbor][neighbor_v-vertex_offset] += this->candidate_frequency[next_vertex][v-next_vertex_offset];
                    }
                    if (len != 0)
                    {
                        new_score[neighbor] += (len*this->candidate_frequency[next_vertex][v-next_vertex_offset]);
                        this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                    }
                    else
                    {
                        sflag = false;
                        break;
                    }                  
                }
                if (sflag)
                {
                    this->candidate_vertex_set[next_vertex][this->candidate_size[next_vertex]] = v;
                    this->candidate_size[next_vertex]++;
                }
                else
                    this->candidate_valid[next_vertex][v-next_vertex_offset] = 1;                
            }

            cur += 2;
            for (unsigned i = 0; i < forward_iter; ++i)
            {
                neighbor = forward_neighbor[i];
                timestamp[neighbor] = cur; 
                Q.score[neighbor] = new_score[neighbor] / (double)Q.neighbor_list[neighbor].size();
            }
        }

        idx = Q.core_match_order.size();
        //filter the neighbor list of the other tree vertices
        for (unsigned it = 0; it < Q.forest.size(); ++it)
        {
            root = Q.forest[it];
            while (true)
            {
                if (next_match_tree[root].empty())
                    break;
                //find the next query vertex
                min_score = 2.0E+300;
                for (set<unsigned>::iterator i = next_match_tree[root].begin(); i != next_match_tree[root].end(); ++i)
                {
                    if (Q.score[*i] < min_score)
                    {
                        min_score = Q.score[*i];
                        next_vertex = *i;
                    }
                }
                next_vertex_label = Q.label_list[next_vertex];
                next_vertex_offset = G.label_index_list[next_vertex_label];

                Q.forest_match_order[root].push_back(next_vertex);
                Q.match_order.push_back(next_vertex);
                for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
                    Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
                Q.repeated_list[next_vertex_label].push_back(idx+Q.forest_match_order[root].size()-1);
                next_match_tree[root].erase(next_vertex);
                vis[next_vertex] = true;

                forward_iter = 0;
                for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
                {
                    neighbor = Q.neighbor_list[next_vertex][i];
                    if (!vis[neighbor])
                    {
                        Q.backward_neighbor[neighbor].push_back(idx+Q.forest_match_order[root].size()-1);
                        if (Q.vertex_state[neighbor] == 1)
                        {
                            forward_neighbor[forward_iter++] = neighbor;
                            Q.forward_neighbor[next_vertex].push_back(neighbor);
                            if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                                vce[neighbor] = true;
                            next_match_tree[root].insert(neighbor);                   
                        }
                    }   
                }             

                if (vce[next_vertex]) 
                    continue;

                size = this->candidate_size[next_vertex];
                this->candidate_size[next_vertex] = 0;
                //filter the neighbor list of the next query vertex
                for (unsigned i = 0; i < size; ++i)
                {
                    v = this->candidate_vertex_set[next_vertex][i];
                    sflag = true;

                    for (unsigned j = 0; j < forward_iter; ++j)
                    {
                        neighbor = forward_neighbor[j];
                        label = Q.label_list[neighbor];
                        vertex_offset = G.label_index_list[label];
                        len = 0;
                        begin_loc = G.offset_list[v*G.label_num+label];
                        end_loc = G.offset_list[v*G.label_num+label+1];
                        for (unsigned k = begin_loc; k < end_loc; ++k)
                        {
                            neighbor_v = G.adjacency_list[k];
                            //0: not verified， 1：verified but invalid， 2：verfied and valid
                            if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == 1)
                                continue;
                            if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == timestamp[neighbor])
                            {
                                flag = true;
                                for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t)
                                {
                                    nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                                    if (!G.nsa[neighbor_v][nlabel])
                                    {
                                        flag = false;
                                        break;
                                    }
                                }
                                if (!flag)
                                {
                                    this->candidate_valid[neighbor][neighbor_v-vertex_offset] = 1;
                                    continue; 
                                }
                            }

                            if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == timestamp[neighbor])
                            {
                                if (!vce[neighbor]) 
                                {
                                    this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                                    this->candidate_size[neighbor]++;                               
                                }
                                this->candidate_valid[neighbor][neighbor_v-vertex_offset] = cur+2;
                            }
                            len++;
                            if (!vce[neighbor])
                                this->candidate_frequency[neighbor][neighbor_v-vertex_offset] += this->candidate_frequency[next_vertex][v-next_vertex_offset];
                        }
                        if (len != 0)
                        {
                            new_score[neighbor] += (len*this->candidate_frequency[next_vertex][v-next_vertex_offset]);
                            this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                        }
                        else
                        {
                            sflag = false;
                            break;
                        }                         
                    }
                    if (sflag) 
                    {
                        this->candidate_vertex_set[next_vertex][this->candidate_size[next_vertex]] = v;
                        this->candidate_size[next_vertex]++;
                    }
                    else
                        this->candidate_valid[next_vertex][v-next_vertex_offset] = 1;                  
                }

                cur += 2;
                for (unsigned i = 0; i < forward_iter; ++i)
                {
                    neighbor = forward_neighbor[i];
                    timestamp[neighbor] = cur; 
                    Q.score[neighbor] = new_score[neighbor] / (double)Q.neighbor_list[neighbor].size(); 
                }
            }
            idx += Q.forest_match_order[root].size();
        }

        for (unsigned it = Q.match_order.size(); it > 0;)
        {
            --it;
            pre_vertex = Q.match_order[it];
            if (Q.forward_neighbor[pre_vertex].size() == 0)
                continue; 
            else
            {
                pre_vertex_label = Q.label_list[pre_vertex];
                pre_vertex_offset = G.label_index_list[pre_vertex_label];
                for (unsigned i = 0; i < this->candidate_size[pre_vertex]; ++i)
                {
                    v = this->candidate_vertex_set[pre_vertex][i];
                    sflag = true;
                    for (unsigned j = 0; j < Q.forward_neighbor[pre_vertex].size(); ++j)
                    {
                        temp_workload = 0;
                        neighbor = Q.forward_neighbor[pre_vertex][j];
                        label = Q.label_list[neighbor];    
                        vertex_offset = G.label_index_list[label];
                        if (!this->vfs[v][neighbor])
                        {
                            this->vfs[v][neighbor] = true;
                            begin_loc = G.offset_list[v*G.label_num+label];
                            end_loc = G.offset_list[v*G.label_num+label+1];
                            for (unsigned k = begin_loc; k < end_loc; ++k)
                            {
                                neighbor_v = G.adjacency_list[k];
                                if (this->candidate_valid[neighbor][neighbor_v-vertex_offset] == timestamp[neighbor])
                                {
                                    this->adjacency_list.push_back(neighbor_v);
                                    this->offset_list[v*Q.vertex_num*2+neighbor*2+1]++;
                                    if (Q.backward_neighbor[neighbor].back()==it)
                                        temp_workload += this->candidate_workload[neighbor][neighbor_v-vertex_offset]; //compute the workload from bottom to up
                                }
                            }
                            if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] != 0)
                            {
                                this->offset_list[v*Q.vertex_num*2+neighbor*2] = offset;
                                offset += this->offset_list[v*Q.vertex_num*2+neighbor*2+1];
                            }
                            else
                            {
                                sflag = false;
                                break;
                            }
                        }                    
                        else
                        {
                            if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] == 0)
                            {
                                sflag = false;
                                break;
                            }                        
                        }
                        if (Q.backward_neighbor[neighbor].back()==it)
                            this->candidate_workload[pre_vertex][v-pre_vertex_offset] *= temp_workload;
                    }
                    if (!sflag)  //filter
                    {
                        this->candidate_valid[pre_vertex][v-pre_vertex_offset] = 1;
                        this->candidate_workload[pre_vertex][v-pre_vertex_offset] = 0;
                    }
                }
            }
        }
        sort(this->candidate_vertex_set[Q.start_vertex], this->candidate_vertex_set[Q.start_vertex]+this->candidate_size[Q.start_vertex], 
        [this](const unsigned& x, const unsigned& y)->bool{return this->candidate_workload[this->start_vertex][x-this->start_vertex_offset] > this->candidate_workload[this->start_vertex][y-this->start_vertex_offset];});

        unsigned repeat_flag;
        vector<unsigned> leaf_vec;
        vector<unsigned> rleaf_vec;
        vector<unsigned> rleaf_off;
        unsigned begin_iter, end_iter;
        offset = 0;
        rleaf_off.push_back(0);
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            if (vis[v])
                continue;
            label = Q.label_list[v];
            repeat_flag = false;
            for (unsigned j = i+1; j < Q.leaf.size(); ++j)
            {
                if (!vis[Q.leaf[j]])
                {
                    if (label == Q.label_list[Q.leaf[j]])
                    {
                        repeat_flag = true;
                        rleaf_vec.push_back(Q.leaf[j]);
                        offset++;
                        vis[Q.leaf[j]] = true;
                    }
                }
            }
            if (repeat_flag)
            {
                rleaf_vec.push_back(v);
                offset++;
                rleaf_off.push_back(offset);
            }
            else
                leaf_vec.push_back(v);
            vis[v] = true;                
        }
        for (unsigned i = 0; i < leaf_vec.size(); ++i)
            Q.leaf[i] = leaf_vec[i];
        Q.repeat_leaf_begin_loc = leaf_vec.size();
        begin_iter = Q.repeat_leaf_begin_loc;
        end_iter = Q.leaf.size() - 1;
        for (unsigned i = 0; i < rleaf_off.size()-1; ++i)
        {
            if (rleaf_off[i+1] - rleaf_off[i] > 2)
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[end_iter--] = rleaf_vec[j];
            }
            else
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[begin_iter++] = rleaf_vec[j];
            }
        }
        Q.repeat_two_leaf_begin_loc = begin_iter; 

        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            label = Q.label_list[v];
            for (unsigned j = 0; j < Q.repeated_list[label].size(); ++j)
                Q.repeated_vertex_set[v].push_back(Q.repeated_list[label][j]);
            Q.repeated_list[label].push_back(idx);
            idx++;
        }
    }    

    for (unsigned i = 0; i < Q.vertex_num; ++i)
    {
        label = Q.label_list[i];
        if (Q.repeated_list[label].size() > 1)
            Q.repeat_state[i] = true;
        else
            Q.repeat_state[i] = false;
    }

    for (unsigned i = 0; i < Q.core.size(); ++i)
    {
        v = Q.core[i];
        if (Q.backward_neighbor[v].size() > Q.max_degree)
            Q.max_degree = Q.backward_neighbor[v].size();
    }   

    }
    else
    {

    if ((Q.query_state == 0) || (Q.query_state == 1)) // core or core + leaf
    {
        Q.core_match_order.push_back(Q.start_vertex);
        Q.repeated_list[start_vertex_label].push_back(0);
        vis[Q.start_vertex] = true;
        forward_iter = 0;
        for (unsigned i = 0; i < Q.neighbor_list[Q.start_vertex].size(); ++i)
        {
            neighbor = Q.neighbor_list[Q.start_vertex][i];
            Q.backward_neighbor[neighbor].push_back(0);
            if (Q.vertex_state[neighbor] == 0)
            {
                forward_neighbor[forward_iter++] = neighbor;
                Q.forward_neighbor[Q.start_vertex].push_back(neighbor);
                next_match_list.insert(neighbor);
                vcm[neighbor] = true;
            }
        }   

        for (unsigned i = 0; i < G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label].size(); ++i)
        {
            v = G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label][i];

            sflag = true;
            for (unsigned j = 0; j < Q.neighbor_list[Q.start_vertex].size(); ++j)
            {  
                label = Q.label_list[Q.neighbor_list[Q.start_vertex][j]];
                if (!G.nsa[v][label])
                {
                    sflag = false;
                    break;
                }
            }
            if (!sflag)
                continue;

            for (unsigned j = 0; j < forward_iter; ++j)
            {
                neighbor = forward_neighbor[j];
                label = Q.label_list[neighbor];
                len = 0;
                begin_loc = G.offset_list[v*G.label_num+label];
                end_loc = G.offset_list[v*G.label_num+label+1];
                for (unsigned k = begin_loc; k < end_loc; ++k)
                {
                    neighbor_v = G.adjacency_list[k]; 

                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 1)
                        continue;
                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 0)
                    {
                        flag = true;
                        for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t) 
                        {
                            nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                            if (!G.nsa[neighbor_v][nlabel])
                            {
                                flag = false;
                                break;
                            }
                        }
                        if (!flag)
                        {
                            this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 1;
                            continue; 
                        }
                    }

                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 0)
                    {
                        this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                        this->candidate_size[neighbor]++;
                        this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 2;
                    }
                    len++;
                }
                if (len != 0)
                {
                    new_score[neighbor] += len;
                    this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                }
                else
                {
                    sflag = false;
                    break;
                }
            }
            if (sflag) 
            {
                this->candidate_vertex_set[Q.start_vertex][this->candidate_size[Q.start_vertex]] = v;
                this->candidate_size[Q.start_vertex]++;
            }
        }
        
        cur += 2;
        for (unsigned i = 0; i < forward_iter; ++i)
        {
            neighbor = forward_neighbor[i];
            timestamp[neighbor] = cur; 
            Q.score[neighbor] = min(Q.score[neighbor], new_score[neighbor] / (double)Q.neighbor_list[neighbor].size()); 
        }

        while (true)
        {
            if (next_match_list.empty())
                break;

            min_score = 2.0E+300;
            pre_vertex = Q.core_match_order.back();
            for (set<unsigned>::iterator i = next_match_list.begin(); i != next_match_list.end(); ++i)
            {
                if((Q.edge_relation[pre_vertex] == Q.edge_relation[*i])&&(Q.label_list[pre_vertex] == Q.label_list[*i]))
                {
                    next_vertex = *i;
                    if (Q.NEC[pre_vertex] == 0)
                    {
                        cur_NEC++;
                        Q.NEC[pre_vertex] = cur_NEC;
                    }
                    Q.NEC[*i] = cur_NEC;
                    break;
                }
                if (Q.score[*i] < min_score)
                {
                    min_score = Q.score[*i];
                    next_vertex = *i;
                }
            }
            next_vertex_label = Q.label_list[next_vertex];

            Q.core_match_order.push_back(next_vertex);
            for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
                Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
            Q.repeated_list[next_vertex_label].push_back(Q.core_match_order.size()-1);
            next_match_list.erase(next_vertex);
            vis[next_vertex] = true;

            forward_iter = 0;
            for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
            {
                neighbor = Q.neighbor_list[next_vertex][i];
                if (!vis[neighbor])
                {
                    Q.backward_neighbor[neighbor].push_back(Q.core_match_order.size()-1);
                    if (Q.vertex_state[neighbor] == 0)
                    {
                        forward_neighbor[forward_iter++] = neighbor;
                        Q.forward_neighbor[next_vertex].push_back(neighbor);
                        if ((Q.backward_neighbor[neighbor].size() + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
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
                v = this->candidate_vertex_set[next_vertex][i];
                sflag = true;

                for (unsigned j = 0; j < forward_iter; ++j)
                {
                    neighbor = forward_neighbor[j];
                    label = Q.label_list[neighbor];
                    len = 0;
                    begin_loc = G.offset_list[v*G.label_num+label];
                    end_loc = G.offset_list[v*G.label_num+label+1];
                    for (unsigned k = begin_loc; k < end_loc; ++k)
                    {
                        neighbor_v = G.adjacency_list[k]; 

                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] < timestamp[neighbor])
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == (cur+1))
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == timestamp[neighbor])
                        {
                            if (!vcm[neighbor])
                            {
                                flag = true;
                                for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t) 
                                {
                                    nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                                    if (!G.nsa[neighbor_v][nlabel])
                                    {
                                        flag = false;
                                        break;
                                    }
                                }
                                if (!flag)
                                {
                                    this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = cur+1;
                                    continue; 
                                }
                            }
                        }

                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == timestamp[neighbor])
                        {
                            if (!vce[neighbor]) 
                            {
                                this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                                this->candidate_size[neighbor]++;                               
                            }
                            this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = cur+2;
                        }
                        len++;
                    }

                    if (len != 0)
                    {
                        new_score[neighbor] += len;
                        this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                    }
                    else
                    {
                        sflag = false;
                        break;
                    }
                }
                if (sflag) 
                {
                    this->candidate_vertex_set[next_vertex][this->candidate_size[next_vertex]] = v;
                    this->candidate_size[next_vertex]++;
                }
                else
                    this->candidate_valid[next_vertex][v-G.label_index_list[next_vertex_label]] = 1;
            }

            cur += 2;
            for (unsigned i = 0; i < forward_iter; ++i)
            {
                neighbor = forward_neighbor[i];
                timestamp[neighbor] = cur; 
                Q.score[neighbor] = min(Q.score[neighbor], new_score[neighbor] / (double)Q.neighbor_list[neighbor].size());
            }
        }

        for (unsigned it = Q.core_match_order.size(); it > 0;)
        {
            --it;
            pre_vertex = Q.core_match_order[it];
            if (Q.forward_neighbor[pre_vertex].size() == 0)
                continue;
            else
            {
                pre_vertex_label = Q.label_list[pre_vertex];
                for (unsigned i = 0; i < this->candidate_size[pre_vertex]; ++i)
                {
                    v = this->candidate_vertex_set[pre_vertex][i];
                    sflag = true;
                    for (unsigned j = 0; j < Q.forward_neighbor[pre_vertex].size(); ++j)
                    {
                        neighbor = Q.forward_neighbor[pre_vertex][j];
                        label = Q.label_list[neighbor];    
                        if (!this->vfs[v][neighbor])
                        {
                            this->vfs[v][neighbor] = true;
                            begin_loc = G.offset_list[v*G.label_num+label];
                            end_loc = G.offset_list[v*G.label_num+label+1];
                            for (unsigned k = begin_loc; k < end_loc; ++k)
                            {
                                neighbor_v = G.adjacency_list[k]; 
                                if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == timestamp[neighbor])
                                {
                                    this->adjacency_list.push_back(neighbor_v);
                                    this->offset_list[v*Q.vertex_num*2+neighbor*2+1]++;
                                }
                            }
                            if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] != 0)
                            {
                                this->offset_list[v*Q.vertex_num*2+neighbor*2] = offset;
                                offset += this->offset_list[v*Q.vertex_num*2+neighbor*2+1];
                            }
                            else
                            {
                                sflag = false;
                                break;
                            }
                        }                    
                        else
                        {
                            if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] == 0)
                            {
                                sflag = false;
                                break;
                            }                          
                        }
                    }
                    if (!sflag) 
                        this->candidate_valid[pre_vertex][v-G.label_index_list[pre_vertex_label]] = 1;
                }
            }
        }

        if (Q.leaf.size() == 0)
        {
            for (unsigned i = 0; i < Q.vertex_num; ++i)
            {
                label = Q.label_list[i];
                if (Q.repeated_list[label].size() > 1)
                    Q.repeat_state[i] = true;
                else
                    Q.repeat_state[i] = false;
            }

            for (unsigned i = 0; i < Q.core.size(); ++i)
            {
                v = Q.core[i];
                if (Q.backward_neighbor[v].size() > Q.max_degree)
                    Q.max_degree = Q.backward_neighbor[v].size();
            }   
            return;
        }
        unsigned repeat_flag;
        vector<unsigned> leaf_vec;
        vector<unsigned> rleaf_vec;
        vector<unsigned> rleaf_off;
        unsigned begin_iter, end_iter;
        offset = 0;
        rleaf_off.push_back(0);
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            if (vis[v])
                continue;
            label = Q.label_list[v];
            repeat_flag = false;
            for (unsigned j = i+1; j < Q.leaf.size(); ++j)
            {
                if (!vis[Q.leaf[j]])
                {
                    if (label == Q.label_list[Q.leaf[j]])
                    {
                        repeat_flag = true;
                        rleaf_vec.push_back(Q.leaf[j]);
                        offset++;
                        vis[Q.leaf[j]] = true;
                    }
                }
            }
            if (repeat_flag)
            {
                rleaf_vec.push_back(v);
                offset++;
                rleaf_off.push_back(offset);
            }
            else
                leaf_vec.push_back(v);
            vis[v] = true;                
        }
        for (unsigned i = 0; i < leaf_vec.size(); ++i)
            Q.leaf[i] = leaf_vec[i];
        Q.repeat_leaf_begin_loc = leaf_vec.size();
        begin_iter = Q.repeat_leaf_begin_loc;
        end_iter = Q.leaf.size() - 1;
        for (unsigned i = 0; i < rleaf_off.size()-1; ++i)
        {
            if (rleaf_off[i+1] - rleaf_off[i] > 2)
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[end_iter--] = rleaf_vec[j];
            }
            else
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[begin_iter++] = rleaf_vec[j];
            }
        }
        Q.repeat_two_leaf_begin_loc = begin_iter; 

        idx = Q.core_match_order.size();
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            label = Q.label_list[v];
            for (unsigned j = 0; j < Q.repeated_list[label].size(); ++j)
                Q.repeated_vertex_set[v].push_back(Q.repeated_list[label][j]);
            Q.repeated_list[label].push_back(idx);
            idx++;
        }
    }
    else if (Q.query_state == 2) // tree + leaf
    {
        Q.tree_match_order.push_back(Q.start_vertex);
        Q.repeated_list[start_vertex_label].push_back(0);
        vis[Q.start_vertex] = true;
        forward_iter = 0;
        for (unsigned i = 0; i < Q.neighbor_list[Q.start_vertex].size(); ++i)
        {
            neighbor = Q.neighbor_list[Q.start_vertex][i];
            Q.backward_neighbor[neighbor].push_back(0);
            if (Q.vertex_state[neighbor] == 1)
            {
                forward_neighbor[forward_iter++] = neighbor;
                Q.forward_neighbor[Q.start_vertex].push_back(neighbor);
                if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                    vce[neighbor] = true;
                next_match_list.insert(neighbor);
            }
        }   
                        
        for (unsigned i = 0; i < G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label].size(); ++i)
        {
            v = G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label][i]; 

            sflag = true;
            for (unsigned j = 0; j < Q.neighbor_list[Q.start_vertex].size(); ++j)
            {  
                label = Q.label_list[Q.neighbor_list[Q.start_vertex][j]];
                if (!G.nsa[v][label])
                {
                    sflag = false;
                    break;
                }
            }
            if (!sflag) 
                continue;

            for (unsigned j = 0; j < forward_iter; ++j)
            {
                neighbor = forward_neighbor[j];
                label = Q.label_list[neighbor];
                len = 0;
                begin_loc = G.offset_list[v*G.label_num+label];
                end_loc = G.offset_list[v*G.label_num+label+1];
                for (unsigned k = begin_loc; k < end_loc; ++k)
                {
                    neighbor_v = G.adjacency_list[k];

                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 1)
                        continue;
                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 0)
                    {
                        flag = true;
                        for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t) 
                        {
                            nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                            if (!G.nsa[neighbor_v][nlabel])
                            {
                                flag = false;
                                break;
                            }
                        }
                        if (!flag)
                        {
                            this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 1;
                            continue; 
                        }
                    }

                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 0)
                    {
                        if (!vce[neighbor])
                        {
                            this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                            this->candidate_size[neighbor]++;                               
                        }
                        this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 2;
                    }
                    len++;    
                }
                if (len != 0)
                    new_score[neighbor] += len;
                else
                {
                    sflag = false;
                    break;
                }                
            }
            if (sflag) 
            {
                this->candidate_vertex_set[Q.start_vertex][this->candidate_size[Q.start_vertex]] = v;
                this->candidate_size[Q.start_vertex]++;
            }
        }
        
        for (unsigned i = 0; i < forward_iter; ++i)
        {
            neighbor = forward_neighbor[i];
            Q.score[neighbor] = min(Q.score[neighbor], new_score[neighbor] / (double)Q.neighbor_list[neighbor].size()); 
        }

        while (true)
        {
            if (next_match_list.empty())
                break;

            min_score = 2.0E+300;
            pre_vertex = Q.tree_match_order.back();
            for (set<unsigned>::iterator i = next_match_list.begin(); i != next_match_list.end(); ++i)
            {
                if (Q.score[*i] < min_score)
                {
                    min_score = Q.score[*i];
                    next_vertex = *i;
                }
            }
            next_vertex_label = Q.label_list[next_vertex];

            Q.tree_match_order.push_back(next_vertex);
            for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
                Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
            Q.repeated_list[next_vertex_label].push_back(Q.tree_match_order.size()-1);
            next_match_list.erase(next_vertex);
            vis[next_vertex] = true;

            forward_iter = 0;
            for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
            {
                neighbor = Q.neighbor_list[next_vertex][i];
                if (!vis[neighbor])
                {
                    Q.backward_neighbor[neighbor].push_back(Q.tree_match_order.size()-1);
                    if (Q.vertex_state[neighbor] == 1)
                    {
                        forward_neighbor[forward_iter++] = neighbor;
                        Q.forward_neighbor[next_vertex].push_back(neighbor);
                        if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                            vce[neighbor] = true;
                        next_match_list.insert(neighbor);                      
                    }
                }   
            }            
            
            if (vce[next_vertex]) 
                continue;

            size = this->candidate_size[next_vertex];
            this->candidate_size[next_vertex] = 0;

            for (unsigned i = 0; i < size; ++i)
            {
                v = this->candidate_vertex_set[next_vertex][i]; 
                sflag = true;

                for (unsigned j = 0; j < forward_iter; ++j)
                {
                    neighbor = forward_neighbor[j];
                    label = Q.label_list[neighbor];
                    len = 0;
                    begin_loc = G.offset_list[v*G.label_num+label];
                    end_loc = G.offset_list[v*G.label_num+label+1];
                    for (unsigned k = begin_loc; k < end_loc; ++k)
                    {
                        neighbor_v = G.adjacency_list[k]; 

                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 1)
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 0)
                        {
                            flag = true;
                            for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t) 
                            {
                                nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                                if (!G.nsa[neighbor_v][nlabel])
                                {
                                    flag = false;
                                    break;
                                }
                            }
                            if (!flag)
                            {
                                this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 1;
                                continue; 
                            }
                        }

                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 0)
                        {
                            if (!vce[neighbor]) 
                            {
                                this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                                this->candidate_size[neighbor]++;                               
                            }
                            this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 2;
                        }
                        len++;
                    }
                    if (len != 0)
                        new_score[neighbor] += len;
                    else
                    {
                        sflag = false;
                        break;
                    }                    
                }
                if (sflag) 
                {
                    this->candidate_vertex_set[next_vertex][this->candidate_size[next_vertex]] = v;
                    this->candidate_size[next_vertex]++;
                }
                else
                    this->candidate_valid[next_vertex][v-G.label_index_list[next_vertex_label]] = 1;                
            }

            for (unsigned i = 0; i < forward_iter; ++i)
            {
                neighbor = forward_neighbor[i];
                Q.score[neighbor] = min(Q.score[neighbor], new_score[neighbor] / (double)Q.neighbor_list[neighbor].size());
            }
        }

        for (unsigned it = Q.tree_match_order.size(); it > 0;)
        {
            --it;
            pre_vertex = Q.tree_match_order[it];
            if (Q.forward_neighbor[pre_vertex].size() == 0)
                continue;               
            else
            {
                pre_vertex_label = Q.label_list[pre_vertex];
                for (unsigned i = 0; i < this->candidate_size[pre_vertex]; ++i)
                {
                    v = this->candidate_vertex_set[pre_vertex][i];
                    sflag = true;
                    for (unsigned j = 0; j < Q.forward_neighbor[pre_vertex].size(); ++j)
                    {
                        neighbor = Q.forward_neighbor[pre_vertex][j];
                        label = Q.label_list[neighbor];    

                        begin_loc = G.offset_list[v*G.label_num+label];
                        end_loc = G.offset_list[v*G.label_num+label+1];
                        for (unsigned k = begin_loc; k < end_loc; ++k)
                        {
                            neighbor_v = G.adjacency_list[k]; 
                            if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 2)
                            {
                                this->adjacency_list.push_back(neighbor_v);
                                this->offset_list[v*Q.vertex_num*2+neighbor*2+1]++;
                            }
                        }
                        if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] != 0)
                        {
                            this->offset_list[v*Q.vertex_num*2+neighbor*2] = offset;
                            offset += this->offset_list[v*Q.vertex_num*2+neighbor*2+1];
                        }
                        else
                        {
                            sflag = false;
                            break;
                        }                   
                    }
                    if (!sflag) 
                        this->candidate_valid[pre_vertex][v-G.label_index_list[pre_vertex_label]] = 1;
                }
            }
        }

        unsigned repeat_flag;
        vector<unsigned> leaf_vec;
        vector<unsigned> rleaf_vec;
        vector<unsigned> rleaf_off;
        unsigned begin_iter, end_iter;
        offset = 0;
        rleaf_off.push_back(0);
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            if (vis[v])
                continue;
            label = Q.label_list[v];
            repeat_flag = false;
            for (unsigned j = i+1; j < Q.leaf.size(); ++j)
            {
                if (!vis[Q.leaf[j]])
                {
                    if (label == Q.label_list[Q.leaf[j]])
                    {
                        repeat_flag = true;
                        rleaf_vec.push_back(Q.leaf[j]);
                        offset++;
                        vis[Q.leaf[j]] = true;
                    }
                }
            }
            if (repeat_flag)
            {
                rleaf_vec.push_back(v);
                offset++;
                rleaf_off.push_back(offset);
            }
            else
                leaf_vec.push_back(v);
            vis[v] = true;                
        }
        for (unsigned i = 0; i < leaf_vec.size(); ++i)
            Q.leaf[i] = leaf_vec[i];
        Q.repeat_leaf_begin_loc = leaf_vec.size();
        begin_iter = Q.repeat_leaf_begin_loc;
        end_iter = Q.leaf.size() - 1;
        for (unsigned i = 0; i < rleaf_off.size()-1; ++i)
        {
            if (rleaf_off[i+1] - rleaf_off[i] > 2)
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[end_iter--] = rleaf_vec[j];
            }
            else
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[begin_iter++] = rleaf_vec[j];
            }
        }
        Q.repeat_two_leaf_begin_loc = begin_iter; 

        idx = Q.tree_match_order.size();
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            label = Q.label_list[v];
            for (unsigned j = 0; j < Q.repeated_list[label].size(); ++j)
                Q.repeated_vertex_set[v].push_back(Q.repeated_list[label][j]);
            Q.repeated_list[label].push_back(idx);
            idx++;
        }       
    }
    else if (Q.query_state == 3) // core + tree + leaf
    {      
        set<unsigned> *next_match_tree = new set<unsigned>[Q.vertex_num];
        unsigned root;

        Q.core_match_order.push_back(Q.start_vertex);
        Q.match_order.push_back(Q.start_vertex);
        Q.repeated_list[start_vertex_label].push_back(0);
        vis[Q.start_vertex] = true;
        forward_iter = 0;
        flag = true;
        for (unsigned i = 0; i < Q.neighbor_list[Q.start_vertex].size(); ++i)
        {
            neighbor = Q.neighbor_list[Q.start_vertex][i];
            Q.backward_neighbor[neighbor].push_back(0);
            if (Q.vertex_state[neighbor] == 0)
            {
                forward_neighbor[forward_iter++] = neighbor;
                Q.forward_neighbor[Q.start_vertex].push_back(neighbor);
                next_match_list.insert(neighbor);
                vcm[neighbor] = true;
            }
            else if (Q.vertex_state[neighbor] == 1)
            {
                forward_neighbor[forward_iter++] = neighbor;
                Q.forward_neighbor[Q.start_vertex].push_back(neighbor);
                if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                    vce[neighbor] = true;
                next_match_tree[Q.start_vertex].insert(neighbor);
                if (flag)
                {
                    Q.forest.push_back(Q.start_vertex);
                    flag = false;
                }
            }
        }   

        for (unsigned i = 0; i < G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label].size(); ++i)
        {
            v = G.partitioned_vertex_list[start_vertex_nlabel][start_vertex_label][i]; 

            sflag = true;
            for (unsigned j = 0; j < Q.neighbor_list[Q.start_vertex].size(); ++j)
            {  
                label = Q.label_list[Q.neighbor_list[Q.start_vertex][j]];
                if (!G.nsa[v][label])
                {
                    sflag = false;
                    break;
                }
            }
            if (!sflag)  
                continue;

            for (unsigned j = 0; j < forward_iter; ++j)
            {
                neighbor = forward_neighbor[j];
                label = Q.label_list[neighbor];
                len = 0;
                begin_loc = G.offset_list[v*G.label_num+label];
                end_loc = G.offset_list[v*G.label_num+label+1];
                for (unsigned k = begin_loc; k < end_loc; ++k)
                {
                    neighbor_v = G.adjacency_list[k]; 

                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 1)
                        continue;
                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 0)
                    {
                        flag = true;
                        for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t) 
                        {
                            nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                            if (!G.nsa[neighbor_v][nlabel])
                            {
                                flag = false;
                                break;
                            }
                        }
                        if (!flag)
                        {
                            this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 1;
                            continue; 
                        }
                    }

                    if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 0)
                    {
                        if (!vce[neighbor]) 
                        {
                            this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                            this->candidate_size[neighbor]++;                               
                        }
                        this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 2;
                    }
                    len++;
                }
                if (len != 0)
                {
                    new_score[neighbor] += len;
                    this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                }
                else
                {
                    sflag = false;
                    break;
                }                
            }
            if (sflag) 
            {
                this->candidate_vertex_set[Q.start_vertex][this->candidate_size[Q.start_vertex]] = v;
                this->candidate_size[Q.start_vertex]++;
            }
        }
        
        cur += 2;
        for (unsigned i = 0; i < forward_iter; ++i)
        {
            neighbor = forward_neighbor[i];
            timestamp[neighbor] = cur; 
            Q.score[neighbor] = min(Q.score[neighbor], new_score[neighbor] / (double)Q.neighbor_list[neighbor].size()); 
        }

        while (true)
        {
            if (next_match_list.empty())
                break;

            min_score = 2.0E+300;
            pre_vertex = Q.core_match_order.back();
            for (set<unsigned>::iterator i = next_match_list.begin(); i != next_match_list.end(); ++i)
            {
                if((Q.edge_relation[pre_vertex] == Q.edge_relation[*i])&&(Q.label_list[pre_vertex] == Q.label_list[*i]))
                {
                    next_vertex = *i;
                    if (Q.NEC[pre_vertex] == 0)
                    {
                        cur_NEC++;
                        Q.NEC[pre_vertex] = cur_NEC;
                    }
                    Q.NEC[*i] = cur_NEC;
                    break;
                }
                if (Q.score[*i] < min_score)
                {
                    min_score = Q.score[*i];
                    next_vertex = *i;
                }
            }
            next_vertex_label = Q.label_list[next_vertex];

            Q.core_match_order.push_back(next_vertex);
            Q.match_order.push_back(next_vertex);
            for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
                Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
            Q.repeated_list[next_vertex_label].push_back(Q.core_match_order.size()-1);
            next_match_list.erase(next_vertex);
            vis[next_vertex] = true;

            forward_iter = 0;
            flag = true;
            for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
            {
                neighbor = Q.neighbor_list[next_vertex][i];
                if (!vis[neighbor])
                {
                    Q.backward_neighbor[neighbor].push_back(Q.core_match_order.size()-1);
                    if (Q.vertex_state[neighbor] == 0)
                    {
                        forward_neighbor[forward_iter++] = neighbor;
                        Q.forward_neighbor[next_vertex].push_back(neighbor);
                        if ((Q.backward_neighbor[neighbor].size() + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
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
                    else if (Q.vertex_state[neighbor] == 1)
                    {
                        forward_neighbor[forward_iter++] = neighbor;
                        Q.forward_neighbor[next_vertex].push_back(neighbor);
                        if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                            vce[neighbor] = true;
                        next_match_tree[next_vertex].insert(neighbor);
                        if (flag)
                        {
                            Q.forest.push_back(next_vertex);
                            flag = false;
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
                v = this->candidate_vertex_set[next_vertex][i]; 
                sflag = true;

                for (unsigned j = 0; j < forward_iter; ++j)
                {
                    neighbor = forward_neighbor[j];
                    label = Q.label_list[neighbor];
                    len = 0;
                    begin_loc = G.offset_list[v*G.label_num+label];
                    end_loc = G.offset_list[v*G.label_num+label+1];
                    for (unsigned k = begin_loc; k < end_loc; ++k)
                    {
                        neighbor_v = G.adjacency_list[k]; 

                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] < timestamp[neighbor])
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == (cur+1))
                            continue;
                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == timestamp[neighbor])
                        {
                            if (!vcm[neighbor])
                            {
                                flag = true;
                                for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t)
                                {
                                    nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                                    if (!G.nsa[neighbor_v][nlabel])
                                    {
                                        flag = false;
                                        break;
                                    }
                                }
                                if (!flag)
                                {
                                    this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = cur+1;
                                    continue; 
                                }
                            }
                        }

                        if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == timestamp[neighbor])
                        {
                            if (!vce[neighbor]) 
                            {
                                this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                                this->candidate_size[neighbor]++;                               
                            }
                            this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = cur+2;
                        }
                        len++;
                    }
                    if (len != 0)
                    {
                        new_score[neighbor] += len;
                        this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                    }
                    else
                    {
                        sflag = false;
                        break;
                    }                  
                }
                if (sflag) 
                {
                    this->candidate_vertex_set[next_vertex][this->candidate_size[next_vertex]] = v;
                    this->candidate_size[next_vertex]++;
                }
                else
                    this->candidate_valid[next_vertex][v-G.label_index_list[next_vertex_label]] = 1;                
            }

            cur += 2;
            for (unsigned i = 0; i < forward_iter; ++i)
            {
                neighbor = forward_neighbor[i];
                timestamp[neighbor] = cur; 
                Q.score[neighbor] = min(Q.score[neighbor], new_score[neighbor] / (double)Q.neighbor_list[neighbor].size()); 
            }
        }

        idx = Q.core_match_order.size();
        for (unsigned it = 0; it < Q.forest.size(); ++it)
        {
            root = Q.forest[it]; 
            while (true)
            {
                if (next_match_tree[root].empty())
                    break;

                min_score = 2.0E+300;
                for (set<unsigned>::iterator i = next_match_tree[root].begin(); i != next_match_tree[root].end(); ++i)
                {
                    if (Q.score[*i] < min_score)
                    {
                        min_score = Q.score[*i];
                        next_vertex = *i;
                    }
                }
                next_vertex_label = Q.label_list[next_vertex];

                Q.forest_match_order[root].push_back(next_vertex);
                Q.match_order.push_back(next_vertex);
                for (unsigned i = 0; i < Q.repeated_list[next_vertex_label].size(); ++i)
                    Q.repeated_vertex_set[next_vertex].push_back(Q.repeated_list[next_vertex_label][i]);
                Q.repeated_list[next_vertex_label].push_back(idx+Q.forest_match_order[root].size()-1);
                next_match_tree[root].erase(next_vertex);
                vis[next_vertex] = true;

                forward_iter = 0;
                for (unsigned i = 0; i < Q.neighbor_list[next_vertex].size(); ++i)
                {
                    neighbor = Q.neighbor_list[next_vertex][i];
                    if (!vis[neighbor])
                    {
                        Q.backward_neighbor[neighbor].push_back(idx+Q.forest_match_order[root].size()-1);
                        if (Q.vertex_state[neighbor] == 1)
                        {
                            forward_neighbor[forward_iter++] = neighbor;
                            Q.forward_neighbor[next_vertex].push_back(neighbor);
                            if ((1 + Q.leaf_num[neighbor]) == Q.neighbor_list[neighbor].size())
                                vce[neighbor] = true;
                            next_match_tree[root].insert(neighbor);                   
                        }
                    }   
                }             

                if (vce[next_vertex])
                    continue;

                size = this->candidate_size[next_vertex];
                this->candidate_size[next_vertex] = 0;

                for (unsigned i = 0; i < size; ++i)
                {
                    v = this->candidate_vertex_set[next_vertex][i];
                    sflag = true;

                    for (unsigned j = 0; j < forward_iter; ++j)
                    {
                        neighbor = forward_neighbor[j];
                        label = Q.label_list[neighbor];
                        len = 0;
                        begin_loc = G.offset_list[v*G.label_num+label];
                        end_loc = G.offset_list[v*G.label_num+label+1];
                        for (unsigned k = begin_loc; k < end_loc; ++k)
                        {
                            neighbor_v = G.adjacency_list[k]; 

                            if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == 1)
                                continue;
                            if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == timestamp[neighbor])
                            {
                                flag = true;
                                for (unsigned t = 0; t < Q.neighbor_list[neighbor].size(); ++t) 
                                {
                                    nlabel = Q.label_list[Q.neighbor_list[neighbor][t]];
                                    if (!G.nsa[neighbor_v][nlabel])
                                    {
                                        flag = false;
                                        break;
                                    }
                                }
                                if (!flag)
                                {
                                    this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = 1;
                                    continue; 
                                }
                            }

                            if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == timestamp[neighbor])
                            {
                                if (!vce[neighbor]) 
                                {
                                    this->candidate_vertex_set[neighbor][this->candidate_size[neighbor]] = neighbor_v;
                                    this->candidate_size[neighbor]++;                               
                                }
                                this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] = cur+2;
                            }
                            len++;
                        }
                        if (len != 0)
                        {
                            new_score[neighbor] += len;
                            this->max_mapping_width[neighbor] = max(this->max_mapping_width[neighbor], len);
                        }
                        else
                        {
                            sflag = false;
                            break;
                        }                         
                    }
                    if (sflag) 
                    {
                        this->candidate_vertex_set[next_vertex][this->candidate_size[next_vertex]] = v;
                        this->candidate_size[next_vertex]++;
                    }
                    else
                        this->candidate_valid[next_vertex][v-G.label_index_list[next_vertex_label]] = 1;                  
                }

                cur += 2;
                for (unsigned i = 0; i < forward_iter; ++i)
                {
                    neighbor = forward_neighbor[i];
                    timestamp[neighbor] = cur; 
                    Q.score[neighbor] = min(Q.score[neighbor], new_score[neighbor] / (double)Q.neighbor_list[neighbor].size()); 
                }
            }
            idx += Q.forest_match_order[root].size();
        }

        for (unsigned it = Q.match_order.size(); it > 0;)
        {
            --it;
            pre_vertex = Q.match_order[it];
            if (Q.forward_neighbor[pre_vertex].size() == 0)
                continue; 
            else
            {
                pre_vertex_label = Q.label_list[pre_vertex];
                for (unsigned i = 0; i < this->candidate_size[pre_vertex]; ++i)
                {
                    v = this->candidate_vertex_set[pre_vertex][i];
                    sflag = true;
                    for (unsigned j = 0; j < Q.forward_neighbor[pre_vertex].size(); ++j)
                    {
                        neighbor = Q.forward_neighbor[pre_vertex][j];
                        label = Q.label_list[neighbor];    
                        if (!this->vfs[v][neighbor])
                        {
                            this->vfs[v][neighbor] = true;
                            begin_loc = G.offset_list[v*G.label_num+label];
                            end_loc = G.offset_list[v*G.label_num+label+1];
                            for (unsigned k = begin_loc; k < end_loc; ++k)
                            {
                                neighbor_v = G.adjacency_list[k]; 
                                if (this->candidate_valid[neighbor][neighbor_v-G.label_index_list[label]] == timestamp[neighbor])
                                {
                                    this->adjacency_list.push_back(neighbor_v);
                                    this->offset_list[v*Q.vertex_num*2+neighbor*2+1]++;
                                }
                            }
                            if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] != 0)
                            {
                                this->offset_list[v*Q.vertex_num*2+neighbor*2] = offset;
                                offset += this->offset_list[v*Q.vertex_num*2+neighbor*2+1];
                            }
                            else
                            {
                                sflag = false;
                                break;
                            }
                        }                    
                        else
                        {
                            if (this->offset_list[v*Q.vertex_num*2+neighbor*2+1] == 0)
                            {
                                sflag = false;
                                break;
                            }                           
                        }
                    }
                    if (!sflag) 
                        this->candidate_valid[pre_vertex][v-G.label_index_list[pre_vertex_label]] = 1;
                }
            }
        }

        unsigned repeat_flag;
        vector<unsigned> leaf_vec;
        vector<unsigned> rleaf_vec;
        vector<unsigned> rleaf_off;
        unsigned begin_iter, end_iter;
        offset = 0;
        rleaf_off.push_back(0);
        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            if (vis[v])
                continue;
            label = Q.label_list[v];
            repeat_flag = false;
            for (unsigned j = i+1; j < Q.leaf.size(); ++j)
            {
                if (!vis[Q.leaf[j]])
                {
                    if (label == Q.label_list[Q.leaf[j]])
                    {
                        repeat_flag = true;
                        rleaf_vec.push_back(Q.leaf[j]);
                        offset++;
                        vis[Q.leaf[j]] = true;
                    }
                }
            }
            if (repeat_flag)
            {
                rleaf_vec.push_back(v);
                offset++;
                rleaf_off.push_back(offset);
            }
            else
                leaf_vec.push_back(v);
            vis[v] = true;                
        }
        for (unsigned i = 0; i < leaf_vec.size(); ++i)
            Q.leaf[i] = leaf_vec[i];
        Q.repeat_leaf_begin_loc = leaf_vec.size();
        begin_iter = Q.repeat_leaf_begin_loc;
        end_iter = Q.leaf.size() - 1;
        for (unsigned i = 0; i < rleaf_off.size()-1; ++i)
        {
            if (rleaf_off[i+1] - rleaf_off[i] > 2)
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[end_iter--] = rleaf_vec[j];
            }
            else
            {
                for (unsigned j = rleaf_off[i]; j < rleaf_off[i+1]; ++j)
                    Q.leaf[begin_iter++] = rleaf_vec[j];
            }
        }
        Q.repeat_two_leaf_begin_loc = begin_iter; 

        for (unsigned i = 0; i < Q.leaf.size(); ++i)
        {
            v = Q.leaf[i];
            label = Q.label_list[v];
            for (unsigned j = 0; j < Q.repeated_list[label].size(); ++j)
                Q.repeated_vertex_set[v].push_back(Q.repeated_list[label][j]);
            Q.repeated_list[label].push_back(idx);
            idx++;
        }
    }    

    for (unsigned i = 0; i < Q.vertex_num; ++i)
    {
        label = Q.label_list[i];
        if (Q.repeated_list[label].size() > 1)
            Q.repeat_state[i] = true;
        else
            Q.repeat_state[i] = false;
    }

    for (unsigned i = 0; i < Q.core.size(); ++i)
    {
        v = Q.core[i];
        if (Q.backward_neighbor[v].size() > Q.max_degree)
            Q.max_degree = Q.backward_neighbor[v].size();
    }   

    }
}

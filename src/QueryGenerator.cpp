#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <dirent.h>
#include <stdio.h>
using namespace std;

int main(int argc, char* argv[])
{
	string graph_name = argv[1];
	//int label_num = atoi(argv[2]);
	DIR *dir = opendir((graph_name + "_CFL_query/").c_str());
	if (opendir((graph_name + "_SCVI_query/").c_str()) == NULL)
	{
		string cmd = "mkdir " + graph_name + "_SCVI_query/";
		system(cmd.c_str());
	}
	if (opendir((graph_name + "_DAF_query/").c_str()) == NULL)
	{
		string cmd = "mkdir " + graph_name + "_DAF_query/";
		system(cmd.c_str());
	}
	struct dirent *ptr;
	while ((ptr = readdir(dir)) != NULL)
	{
		string filename = ptr->d_name;
		if ((filename == ".") || (filename == ".."))
			continue;

		ifstream fin(graph_name + "_CFL_query/" + filename);
		ofstream fout1(graph_name + "_SCVI_query/" + filename);
		ofstream fout2(graph_name + "_DAF_query/" + filename);
		char type;
        vector<vector<int> > neighbors;
        vector<int> labels;
		while (fin >> type)
		{
			if (type == 't')
			{
				int node_num, edge_num;
				fin >> node_num >> edge_num;
				fout1 << "t " << node_num << " " << edge_num << endl;
				fout2 << "t 1 " << node_num << " " << edge_num * 2 << endl;
                neighbors.resize(node_num);
				labels.resize(node_num);
			}
			else if (type == 'v')
			{
				int vid, label, neighbor_num;
				fin >> vid >> label >> neighbor_num;
				fout1 << "v " << vid << " " << label << endl;
                labels[vid] = label;
			}
			else if (type == 'e')
			{
				int src_vid, dst_vid, label;
				fin >> src_vid >> dst_vid >> label;
				fout1 << "e " << src_vid << " " << dst_vid << " " << label << endl;
                neighbors[src_vid].push_back(dst_vid);
                neighbors[dst_vid].push_back(src_vid);
			}
		}
        for (int i = 0; i < neighbors.size(); ++i)
        {
            fout2 << i << " " << labels[i] << " " << neighbors[i].size();
 			for (int j = 0; j < neighbors[i].size(); ++j)
			{
				fout2 << " " << neighbors[i][j];
			}
			fout2 << endl;           
        }
		fin.close();
		fout1.close();
		fout2.close();
	}
	closedir(dir);
	return 0;
}

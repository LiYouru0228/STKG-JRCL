import pickle
import numpy as np
import random
from tqdm import tqdm
from itertools import islice,combinations
import networkx as nx
import collections
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import os
import re
import pickle
import datetime
import copy
import torch
from torch.utils import data
import argparse

np.random.seed(20221231)

def build_knowledge_graph(kg, slice_id, isNew = False):
    def _print_graph_statistic(Graph, slice_id):
        print('The slice ' + str(slice_id) + ' knowledge graph has been built completely')
        print('The number of nodes is:  ' + str(len(Graph.nodes()))) 
        print('The number of edges is:  ' + str(len(Graph.edges())))
    
    if isNew:
        pair2relation = {}
        for item in kg:
            nodes = kg[item]
            for node in nodes: 
                pair2relation[item,node[0]] = node[1]
                pair2relation[node[0],item] = node[1]

        kg_nodes = []
        kg_edges = collections.defaultdict(list)

        for item in kg:
            kg_nodes.append(item)
            nodes = [each[0] for each in kg[item]]
            for node in nodes:
                kg_edges[item].append(node)

        knowledge_graph = nx.Graph()
        kg_nodes_list = kg_nodes
        kg_edges_list = kg_edges

        for n in kg_nodes_list: knowledge_graph.add_node(n)
        for start in kg_edges_list:
            for end in kg_edges_list[start]:
                knowledge_graph.add_edge(start,end)
                
        f_knowledge_graph = open('kg-' + str(slice_id) + '.pkl','wb')
        pickle.dump((knowledge_graph, pair2relation), f_knowledge_graph)
        f_knowledge_graph.close()
    else:
        knowledge_graph = pickle.load(open('kg-' + str(slice_id) + '.pkl', "rb"))[0]
        pair2relation = pickle.load(open('kg-' + str(slice_id) + '.pkl', "rb"))[1]
    
    _print_graph_statistic(knowledge_graph, slice_id)
    
    return knowledge_graph, pair2relation

def GenSamples(data, knowledge_graph, dataset_name): 
    def _dataset_split(raw_data):
        rating_np = []
        for each in raw_data:
            if knowledge_graph.has_node(int(each[0])) and knowledge_graph.has_node(int(each[1])):
                rating_np.append([each[0],each[1],each[2]])
        rating_np = np.asarray(rating_np)
        
        eval_ratio = 0.1
        test_ratio = 0.1

        n_ratings = len(rating_np)
        eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
        left = set(range(n_ratings)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))

        train_data = np.array(rating_np)[train_indices]
        eval_data = np.array(rating_np)[eval_indices]
        test_data = np.array(rating_np)[test_indices]
        
        print ("Dataset is splitted as:")
        print(train_data.shape, eval_data.shape, test_data.shape)

        return train_data, eval_data, test_data
    
    raw_data = data['sample_dict'][dataset_name]
    return _dataset_split(raw_data), knowledge_graph

def join_base_info(data):
    X,y = [],[]
    for each in data:
        if int(each[0]) in entityid2base_info and int(each[0]) in entityid2base_info:
            X.append(entityid2base_info[int(each[0])]+entityid2base_info[int(each[1])])
            y.append(int(each[2]))
    return np.asarray(X),y

def MinePathsMetaTag(graph, entityid2type, entityid2lbs_info, pair2relation, node1, node2, k):
    node2path, node2lbs = [],[]
    
    # extract paths from short to longer:
    def _k_shortest_paths(G, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
        )
    
    if nx.has_path(graph, node1, node2):
        paths = _k_shortest_paths(graph, node1, node2, k)

        # extract paths and lbs:
        for i in range(len(paths)):
            t_path, t_lbs = [],[]
            for j in range(len(paths[i])):
                if j != len(paths[i]) - 1:
                    t_path.append(paths[i][j]) # add entity
                    if paths[i][j] in entityid2lbs_info:
                        t_lbs.append(entityid2lbs_info[paths[i][j]]) # add lbs
                    if (paths[i][j], paths[i][j + 1]) in pair2relation:
                        t_path.append('r' + str(pair2relation[paths[i][j], paths[i][j + 1]])) # add relation
                else:
                    t_path.append(paths[i][j])
                    if paths[i][j] in entityid2lbs_info:
                        t_lbs.append(entityid2lbs_info[paths[i][j]]) # add lbs
                    
            node2path.append(t_path)
            node2lbs.append(t_lbs)
    else:
        return [str(node1) + str('->NaP->') + str(node2)], ['B->NaP->B']
            
    return node2path, node2lbs

def sequenceEncoding(sample2paths, sampleSize):
    documents = []
    for sample in sample2paths:
        for index, items in enumerate(sample2paths[sample]):
            documents.append(TaggedDocument([str(item) for item in items], [str(sample) + ':' + str(index)]))
    path_model = Doc2Vec(documents, dm=1, vector_size=64, window=5, min_count=2, epochs=10, workers=8)

    path2vec = {}
    for sample in sample2paths:
        if len(sample2paths[sample]) == sampleSize:
            for index, items in enumerate(sample2paths[sample]):
                path2vec[str(sample) + ':' + str(index)] = path_model.docvecs[str(sample) + ':' + str(index)]
        else:
            for index, items in enumerate(sample2paths[sample]):
                path2vec[str(sample) + ':' + str(index)] = path_model.docvecs[str(sample) + ':' + str(index)]
            tmp = path_model.docvecs[str(sample) + ':' + str(index)]
            for cnt in range(sampleSize - index -1):
                path2vec[str(sample) + ':' + str(cnt+len(sample2paths[sample]))] = tmp
    return path2vec

class Dataset(data.Dataset):
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.name[index]

    def __len__(self):
        return len(self.x)
    
def DataPreprocessing(links):            
    feas, label, names = [],[],[]
    for link in links:
        feas.append(entityid2base_info[link[0]] + entityid2base_info[link[1]])
        label.append(link[2])
        names.append(str(link[0]) + '|' + str(link[1]))
    
    feas = np.asarray(feas)
    print(feas.shape)
    print(np.mean(label))
    
    eval_ratio = 0.1
    test_ratio = 0.1

    n_ratings = len(label)
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    data_raw1,data_raw2,data_raw3 = {},{},{}
    data_raw1['data'] = (np.array(feas)[train_indices], list(np.array(label)[train_indices]))
    data_raw1['names'] = list(np.array(names)[train_indices])
    
    data_raw2['data'] = (np.array(feas)[eval_indices], list(np.array(label)[eval_indices]))
    data_raw2['names'] = list(np.array(names)[eval_indices])
    
    data_raw3['data'] = (np.array(feas)[test_indices], list(np.array(label)[test_indices]))
    data_raw3['names'] = list(np.array(names)[test_indices])
    
    print(data_raw1['data'][0].shape,data_raw2['data'][0].shape,data_raw3['data'][0].shape)
    print(np.mean(data_raw1['data'][1]),np.mean(data_raw2['data'][1]),np.mean(data_raw3['data'][1]))
    
    return data_raw1, data_raw2, data_raw3
    
def pipline(args):
    print('load data...')
    f_smc_dict = open("./smc_dict-kdd.pkl", "rb")
    smc_dict = pickle.load(f_smc_dict)
    
    # build stkg
    if args.is_rebuild_stkg:
        raw_dkg = smc_dict['dkg']
        dkg = {}
        for each in tqdm(raw_dkg):
            knowledge_graph, pair2relation = build_knowledge_graph(raw_dkg[each], each, isNew = True)
            dkg[each] = (knowledge_graph, pair2relation)
        f_dkg = open('./stkg.pkl','wb')
        pickle.dump(dkg, f_dkg)
        f_dkg.close()
    else:
        f_dkg = open('./stkg.pkl', 'rb')
        dkg = pickle.load(f_dkg)
    
    # bulid raw data
    raw_data, new_kg = GenSamples(smc_dict, dkg[11][0], args.dataset)
    train_data, eval_data, test_data = raw_data[0], raw_data[1], raw_data[2]
    
    # join base info.
    entityid2base_info = smc_dict['entityid2base_info']
    train_set, eval_set, test_set = join_base_info(train_data),join_base_info(eval_data),join_base_info(test_data)
    
    # build links
    links = raw_data[0].tolist() + raw_data[1].tolist() + raw_data[2].tolist()
    pairs = [link for link in links]
    
    # relational sequence mining
    if arg.is_rebuild_rs:
        node2dualPath = {}
        for each in dkg:
            node2path, node2lbs = {}, {}
            for pair in tqdm(pairs):
                node1, node2 = pair[0], pair[1]
                key, label = str(node1) + '|' + str(node2), pair[2]
                if key not in node2path:
                    node2path[key], node2lbs[key] = MinePathsMetaTag(dkg[each][0], smc_dict['entityid2type'], smc_dict['entityid2lbs_info'], dkg[each][1], node1, node2, args.K)
                else:
                    continue
        node2dualPath[each] = (node2path, node2lbs)
        f_node2dualPath = open('./node2dualPath.pkl', 'wb')
        pickle.dump(node2dualPath, f_node2dualPath)
        f_node2dualPath.close()
    else:
        f_node2dualPath = open('./node2dualPath.pkl', 'rb')
        node2dualPath = pickle.load(f_node2dualPath)
        
    # seqs encoding
    if args.is_encoding_again:
        dkg_path2vec = {}
        for each in tqdm(dkg):
            node2path, node2lbs = node2dualPath[each]
            path2vec = sequenceEncoding(node2path, args.K)
            lbs2vec = sequenceEncoding(node2lbs, args.K)
            dkg_path2vec[each] = (path2vec,lbs2vec)
        f_dkg_path2vec = open('./path2vec.pkl','wb')
        pickle.dump(dkg_path2vec, f_dkg_path2vec)
        f_dkg_path2vec.close()
    else:
        f_dkg_path2vec = open('./path2vec.pkl', 'rb')
        dkg_path2vec = pickle.load(f_dkg_path2vec)
    
    # build dataset
    batch_size = args.bs
    sampleSize = args.K
    train_raw, val_raw, test_raw = DataPreprocessing(links)

    train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    f_sample_dict= open('./dataset.pkl','wb')
    sample_dict = {'train_raw':train_loader, 'val_raw':valid_loader, 'test_raw':test_loader}
    pickle.dump(sample_dict, f_sample_dict)
    f_sample_dict.close()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='专用设备制造业', help='dataset name')
parser.add_argument('--is_rebuild_stkg', type=bool, default=True, help='is_rebuild_stkg')
parser.add_argument('--is_rebuild_rs', type=bool, default=True, help='is_rebuild_rs')
parser.add_argument('--is_encoding_again', type=bool, default=True, help='is_encoding_again')
parser.add_argument('--K', type=int, default=20, help='size of the sampled path subset')
parser.add_argument('--bs', type=int, default=128, help='batch_size')

def main():
    args = parser.parse_args()
    pipline(args)

if __name__ == '__main__':
    main()

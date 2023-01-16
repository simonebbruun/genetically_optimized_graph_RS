import pandas as pd
import numpy as np
import networkx as nx
import json
from operator import itemgetter
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import Parallel, delayed
import multiprocessing
import pickle

num_cores = multiprocessing.cpu_count()


''' Import. '''

ecommerce_items_b = pd.concat([pd.read_csv('ecommerce_items_b.csv'), pd.read_csv('ecommerce_items_t.csv')])
ecommerce_services_b = pd.concat([pd.read_csv('ecommerce_services_b.csv'),  pd.read_csv('ecommerce_services_t.csv')])
claim_items_b = pd.concat([pd.read_csv('claim_items_b.csv'), pd.read_csv('claim_items_t.csv')])
claim_services_b = pd.concat([pd.read_csv('claim_services_b.csv'), pd.read_csv('claim_services_t.csv')])
info_items_b = pd.concat([pd.read_csv('info_items_b.csv'), pd.read_csv('info_items_t.csv')])
info_services_b = pd.concat([pd.read_csv('info_services_b.csv'), pd.read_csv('info_services_t.csv')])
account_items_b = pd.concat([pd.read_csv('account_items_b.csv'), pd.read_csv('account_items_t.csv')])
account_services_b = pd.concat([pd.read_csv('account_services_b.csv'), pd.read_csv('account_services_t.csv')])
purchase_items_b = pd.concat([pd.read_csv('purchase_items_b.csv'), pd.read_csv('purchase_items_tr.csv')])

purchase_items_t = pd.read_csv('purchase_items_t.csv')

purchase_items_f = pd.read_csv('purchase_items_f_t.csv')



''' Preprocessing. '''
# Add letter to ids that signify their type.
ecommerce_items_b['item_id'] = ecommerce_items_b['item_id'].map(lambda x: 'i' + str(x))
ecommerce_items_b['user_id'] = ecommerce_items_b['user_id'].map(lambda x: 'u' + str(x))
ecommerce_services_b['service_id'] = ecommerce_services_b['service_id'].map(lambda x: 's' + str(x))
ecommerce_services_b['user_id'] = ecommerce_services_b['user_id'].map(lambda x: 'u' + str(x))
claim_items_b['item_id'] = claim_items_b['item_id'].map(lambda x: 'i' + str(x))
claim_items_b['user_id'] = claim_items_b['user_id'].map(lambda x: 'u' + str(x))
claim_services_b['service_id'] = claim_services_b['service_id'].map(lambda x: 's' + str(x))
claim_services_b['user_id'] = claim_services_b['user_id'].map(lambda x: 'u' + str(x))
info_items_b['item_id'] = info_items_b['item_id'].map(lambda x: 'i' + str(x))
info_items_b['user_id'] = info_items_b['user_id'].map(lambda x: 'u' + str(x))
info_services_b['service_id'] = info_services_b['service_id'].map(lambda x: 's' + str(x))
info_services_b['user_id'] = info_services_b['user_id'].map(lambda x: 'u' + str(x))
account_items_b['item_id'] = account_items_b['item_id'].map(lambda x: 'i' + str(x))
account_items_b['user_id'] = account_items_b['user_id'].map(lambda x: 'u' + str(x))
account_services_b['service_id'] = account_services_b['service_id'].map(lambda x: 's' + str(x))
account_services_b['user_id'] = account_services_b['user_id'].map(lambda x: 'u' + str(x))

purchase_items_b['item_id'] = purchase_items_b['item_id'].map(lambda x: 'i' + str(x))
purchase_items_b['user_id'] = purchase_items_b['user_id'].map(lambda x: 'u' + str(x))

purchase_items_t['item_id'] = purchase_items_t['item_id'].map(lambda x: 'i' + str(x))
purchase_items_t['user_id'] = purchase_items_t['user_id'].map(lambda x: 'u' + str(x))

purchase_items_f['item_id'] = purchase_items_f['item_id'].map(lambda x: 'i' + str(x))
purchase_items_f['user_id'] = purchase_items_f['user_id'].map(lambda x: 'u' + str(x))


''' Train Graph. '''
# Hyperparameters.
def vector_from_parameter_dict(parameter_dict):
    return list(map(lambda x: x[1], list(parameter_dict.items())))
def parameter_dict_from_vector(vector):
    return {
        "W_ECOMMERCE_ITEMS" : vector[0],
        "W_ECOMMERCE_ITEMS_BACK" : vector[1],
        "W_ECOMMERCE_SERVICES" : vector[2],
        "W_ECOMMERCE_SERVICES_BACK" : vector[3],
        "W_CLAIM_ITEMS" : vector[4],
        "W_CLAIM_ITEMS_BACK" : vector[5],
        "W_CLAIM_SERVICES" : vector[6],
        "W_CLAIM_SERVICES_BACK" : vector[7],
        "W_INFO_ITEMS" : vector[8],
        "W_INFO_ITEMS_BACK" : vector[9],
        "W_INFO_SERVICES" : vector[10],
        "W_INFO_SERVICES_BACK" : vector[11],
        "W_ACCOUNT_ITEMS" : vector[12],
        "W_ACCOUNT_ITEMS_BACK" : vector[13],
        "W_ACCOUNT_SERVICES" : vector[14],
        "W_ACCOUNT_SERVICES_BACK" : vector[15],
        "W_PURCHASE_ITEMS" : vector[16],
        "W_PURCHASE_ITEMS_BACK" : vector[17]
        }


# Building graph.
class InteractionGraph:
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def add_nodes_from_edge_array(self, edge_array, type_1, type_2):
        nodes = [(x[0], {'type': type_1}) for x in edge_array] + [(x[1], {'type': type_2}) for x in edge_array]
        self.graph.add_nodes_from(nodes)

    def add_edges_from_array(self, array, weight_front, weight_back, additional_weight_collumn=False):
        if additional_weight_collumn: 
            forward_edges = [(x[0], x[1], weight_front * x[2]) for x in array]
            back_edges = [(x[1], x[0], weight_back * x[2]) for x in array]
        else:
            forward_edges = [(x[0], x[1], weight_front) for x in array]
            back_edges = [(x[1], x[0], weight_back) for x in array]
            
        self.graph.add_weighted_edges_from(forward_edges)
        self.graph.add_weighted_edges_from(back_edges)
        
        
def build_graph(parameter_dictionary):
    
    multidigraph = InteractionGraph()
    
    ecommerce_items_array = ecommerce_items_b[['item_id', 'user_id']].to_numpy()
    ecommerce_services_array = ecommerce_services_b[['service_id', 'user_id']].to_numpy()
    claim_items_array = claim_items_b[['item_id', 'user_id']].to_numpy()
    claim_services_array = claim_services_b[['service_id', 'user_id']].to_numpy()
    info_items_array = info_items_b[['item_id', 'user_id']].to_numpy()
    info_services_array = info_services_b[['service_id', 'user_id']].to_numpy()
    account_items_array = account_items_b[['item_id', 'user_id']].to_numpy()
    account_services_array = account_services_b[['service_id', 'user_id']].to_numpy()
    purchase_items_array = purchase_items_b[['item_id', 'user_id']].to_numpy()

    multidigraph.add_nodes_from_edge_array(ecommerce_items_array, 'item', 'user')
    multidigraph.add_nodes_from_edge_array(ecommerce_services_array, 'service', 'user')
    multidigraph.add_nodes_from_edge_array(claim_items_array, 'item', 'user')
    multidigraph.add_nodes_from_edge_array(claim_services_array, 'service', 'user')
    multidigraph.add_nodes_from_edge_array(info_items_array, 'item', 'user')
    multidigraph.add_nodes_from_edge_array(info_services_array, 'service', 'user')
    multidigraph.add_nodes_from_edge_array(account_items_array, 'item', 'user')
    multidigraph.add_nodes_from_edge_array(account_services_array, 'service', 'user')
    multidigraph.add_nodes_from_edge_array(purchase_items_array, 'item', 'user')

    multidigraph.add_edges_from_array(ecommerce_items_array, parameter_dictionary["W_ECOMMERCE_ITEMS"], parameter_dictionary["W_ECOMMERCE_ITEMS_BACK"])
    multidigraph.add_edges_from_array(ecommerce_services_array, parameter_dictionary["W_ECOMMERCE_SERVICES"], parameter_dictionary["W_ECOMMERCE_SERVICES_BACK"])
    multidigraph.add_edges_from_array(claim_items_array, parameter_dictionary["W_CLAIM_ITEMS"], parameter_dictionary["W_CLAIM_ITEMS_BACK"])
    multidigraph.add_edges_from_array(claim_services_array, parameter_dictionary["W_CLAIM_SERVICES"], parameter_dictionary["W_CLAIM_SERVICES_BACK"])
    multidigraph.add_edges_from_array(info_items_array, parameter_dictionary["W_INFO_ITEMS"], parameter_dictionary["W_INFO_ITEMS_BACK"])
    multidigraph.add_edges_from_array(info_services_array, parameter_dictionary["W_INFO_SERVICES"], parameter_dictionary["W_INFO_SERVICES_BACK"])
    multidigraph.add_edges_from_array(account_items_array, parameter_dictionary["W_ACCOUNT_ITEMS"], parameter_dictionary["W_ACCOUNT_ITEMS_BACK"])
    multidigraph.add_edges_from_array(account_services_array, parameter_dictionary["W_ACCOUNT_SERVICES"], parameter_dictionary["W_ACCOUNT_SERVICES_BACK"])
    multidigraph.add_edges_from_array(purchase_items_array, parameter_dictionary["W_PURCHASE_ITEMS"], parameter_dictionary["W_PURCHASE_ITEMS_BACK"])

    graph = nx.DiGraph()
    for u,v,d in multidigraph.graph.edges(data=True):
        w = d['weight']
        if graph.has_edge(u,v):
            graph[u][v]['weight'] += w
        else:
            graph.add_edge(u,v,weight=w)

    return graph


def process(graph, nodes, test_set, user, k_neighbors, damping_factor, content_set, filter_dict):
    content = test_set[test_set[:,1]==user][:,0]
        
    pers = [1 if n==user else 0 for n in nodes]
    pers_dict = dict(zip(nodes, pers))
    ppr = nx.pagerank(graph, alpha=damping_factor, personalization=pers_dict)
    ppr_neighbors = dict([(node, ppr[node]) for node in content_set.index if node in ppr])
    ppr_neighbors_k = dict(sorted(ppr_neighbors.items(), key = itemgetter(1), reverse = True)[:k_neighbors])
    
    ppr_neighbors_df = pd.DataFrame.from_dict(ppr_neighbors_k, orient='index')
    score = content_set.mul(ppr_neighbors_df.iloc[:,0], axis=0).dropna().sum()
    score.loc[score.index.difference(filter_dict[user])] = 0.0
    
    recommendation_list_pos = score[score>0.0]
    
    mrr_t = []
    hr_t = []
    for t in range(5):
        recommendation_list = recommendation_list_pos[(-recommendation_list_pos).argsort()][:(t+1)].index
    
        mrr = 0
        hr = 0
        if any(i in recommendation_list for i in content):
            mrr = 1.0/(1+min(np.where(np.isin(recommendation_list,content))[0]))
            hr = 1
        mrr_t.append(mrr)
        hr_t.append(hr)
    
    return mrr_t, hr_t, user


def gen_evaluate_with_set(graph, test_set, k_neighbors, damping_factor, content_set, filter_dict):
    graph_nodes = list(graph.nodes)
    evaluation = list(zip(*Parallel(n_jobs=num_cores)(delayed(process)(graph, graph_nodes, test_set, user, k_neighbors, damping_factor, content_set, filter_dict) for user in np.unique(test_set[:,1]))))
    return (evaluation[0], evaluation[1], evaluation[2])

    
    
# Evaluating different param dictionaries.
content_file = purchase_items_b.set_index('user_id')
encoder = MultiLabelBinarizer()
content_file = pd.DataFrame(encoder.fit_transform(content_file['item_id'].str.split(',')), index=content_file.index, columns=encoder.classes_)
content_file = content_file.groupby(['user_id']).max()

id_user_collumn = 'user_id'
content_collumn = 'item_id'
test_filter = purchase_items_f.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
test_file = purchase_items_t
test_file = test_file[[content_collumn, id_user_collumn]].to_numpy()

k = 90
df = 0.4



with open('genetic_directed.json', 'r') as json_file:
    parameter_dict_genetic = json.load(json_file) 

rec_graph = build_graph(parameter_dict_genetic)


mrrs, hrs, users = gen_evaluate_with_set(rec_graph, test_file, k, df, content_file, test_filter)


mrrs1 = np.stack(mrrs, axis=0)
hrs1 = np.stack(hrs, axis=0)

mrrs2 = np.mean(mrrs1, axis=0)
hrs2 = np.mean(hrs1, axis=0)

mrrs_3 = mrrs1[:,2]
hrs_3 = hrs1[:,2]


varying_thresholds = (mrrs2, hrs2)
statistical_significans = (mrrs_3, hrs_3, users)


with open('varying_thresholds_directed_graph_optimized.pickle', 'wb') as handle:
    pickle.dump(varying_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('statistical_significans_directed_graph_optimized.pickle', 'wb') as handle:
    pickle.dump(statistical_significans, handle, protocol=pickle.HIGHEST_PROTOCOL)



parameter_dict_uniform = parameter_dict_from_vector(np.ones(18))

rec_graph = build_graph(parameter_dict_uniform)


mrrs, hrs, users = gen_evaluate_with_set(rec_graph, test_file, k, df, content_file, test_filter)


mrrs1 = np.stack(mrrs, axis=0)
hrs1 = np.stack(hrs, axis=0)

mrrs2 = np.mean(mrrs1, axis=0)
hrs2 = np.mean(hrs1, axis=0)

mrrs_3 = mrrs1[:,2]
hrs_3 = hrs1[:,2]


varying_thresholds = (mrrs2, hrs2)
statistical_significans = (mrrs_3, hrs_3, users)



with open('varying_thresholds_graph-based_uniform.pickle', 'wb') as handle:
    pickle.dump(varying_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('statistical_significans_graph-based_uniform.pickle', 'wb') as handle:
    pickle.dump(statistical_significans, handle, protocol=pickle.HIGHEST_PROTOCOL)

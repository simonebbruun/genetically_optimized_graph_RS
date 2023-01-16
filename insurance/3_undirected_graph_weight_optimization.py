import pandas as pd
import pygad
import numpy as np
import networkx as nx
import pickle
import json
from operator import itemgetter
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()


''' Import. '''

ecommerce_items_b = pd.read_csv('ecommerce_items_b.csv')
ecommerce_services_b = pd.read_csv('ecommerce_services_b.csv')
claim_items_b = pd.read_csv('claim_items_b.csv')
claim_services_b = pd.read_csv('claim_services_b.csv')
info_items_b = pd.read_csv('info_items_b.csv')
info_services_b = pd.read_csv('info_services_b.csv')
account_items_b = pd.read_csv('account_items_b.csv')
account_services_b = pd.read_csv('account_services_b.csv')

purchase_items_b = pd.read_csv('purchase_items_b.csv')

purchase_items_tr = pd.read_csv('purchase_items_tr.csv')

purchase_items_f = pd.read_csv('purchase_items_f.csv')



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

purchase_items_tr['item_id'] = purchase_items_tr['item_id'].map(lambda x: 'i' + str(x))
purchase_items_tr['user_id'] = purchase_items_tr['user_id'].map(lambda x: 'u' + str(x))

purchase_items_f['item_id'] = purchase_items_f['item_id'].map(lambda x: 'i' + str(x))
purchase_items_f['user_id'] = purchase_items_f['user_id'].map(lambda x: 'u' + str(x))


''' Train Graph. '''
# Hyperparameters.
def vector_from_parameter_dict(parameter_dict):
    return list(map(lambda x: x[1], list(parameter_dict.items())))
def parameter_dict_from_vector(vector):
    return {
        "W_ECOMMERCE_ITEMS" : vector[0],
        "W_ECOMMERCE_SERVICES" : vector[1],
        "W_CLAIM_ITEMS" : vector[2],
        "W_CLAIM_SERVICES" : vector[3],
        "W_INFO_ITEMS" : vector[4],
        "W_INFO_SERVICES" : vector[5],
        "W_ACCOUNT_ITEMS" : vector[6],
        "W_ACCOUNT_SERVICES" : vector[7],
        "W_PURCHASE_ITEMS" : vector[8]
        }


# Building graph.
class InteractionGraph:
    
    def __init__(self):
        self.graph = nx.MultiGraph()
        
    def add_nodes_from_edge_array(self, edge_array, type_1, type_2):
        nodes = [(x[0], {'type': type_1}) for x in edge_array] + [(x[1], {'type': type_2}) for x in edge_array]
        self.graph.add_nodes_from(nodes)

    def add_edges_from_array(self, array, weight, additional_weight_collumn=False):
        if additional_weight_collumn: 
            edges = [(x[0], x[1], weight * x[2]) for x in array]
        else:
            edges = [(x[0], x[1], weight) for x in array]
            
        self.graph.add_weighted_edges_from(edges)

def build_graph(parameter_dictionary):
    
    multigraph = InteractionGraph()
    
    ecommerce_items_array = ecommerce_items_b[['item_id', 'user_id']].to_numpy()
    ecommerce_services_array = ecommerce_services_b[['service_id', 'user_id']].to_numpy()
    claim_items_array = claim_items_b[['item_id', 'user_id']].to_numpy()
    claim_services_array = claim_services_b[['service_id', 'user_id']].to_numpy()
    info_items_array = info_items_b[['item_id', 'user_id']].to_numpy()
    info_services_array = info_services_b[['service_id', 'user_id']].to_numpy()
    account_items_array = account_items_b[['item_id', 'user_id']].to_numpy()
    account_services_array = account_services_b[['service_id', 'user_id']].to_numpy()
    purchase_items_array = purchase_items_b[['item_id', 'user_id']].to_numpy()

    multigraph.add_nodes_from_edge_array(ecommerce_items_array, 'item', 'user')
    multigraph.add_nodes_from_edge_array(ecommerce_services_array, 'service', 'user')
    multigraph.add_nodes_from_edge_array(claim_items_array, 'item', 'user')
    multigraph.add_nodes_from_edge_array(claim_services_array, 'service', 'user')
    multigraph.add_nodes_from_edge_array(info_items_array, 'item', 'user')
    multigraph.add_nodes_from_edge_array(info_services_array, 'service', 'user')
    multigraph.add_nodes_from_edge_array(account_items_array, 'item', 'user')
    multigraph.add_nodes_from_edge_array(account_services_array, 'service', 'user')
    multigraph.add_nodes_from_edge_array(purchase_items_array, 'item', 'user')

    multigraph.add_edges_from_array(ecommerce_items_array, parameter_dictionary["W_ECOMMERCE_ITEMS"])
    multigraph.add_edges_from_array(ecommerce_services_array, parameter_dictionary["W_ECOMMERCE_SERVICES"])
    multigraph.add_edges_from_array(claim_items_array, parameter_dictionary["W_CLAIM_ITEMS"])
    multigraph.add_edges_from_array(claim_services_array, parameter_dictionary["W_CLAIM_SERVICES"])
    multigraph.add_edges_from_array(info_items_array, parameter_dictionary["W_INFO_ITEMS"])
    multigraph.add_edges_from_array(info_services_array, parameter_dictionary["W_INFO_SERVICES"])
    multigraph.add_edges_from_array(account_items_array, parameter_dictionary["W_ACCOUNT_ITEMS"])
    multigraph.add_edges_from_array(account_services_array, parameter_dictionary["W_ACCOUNT_SERVICES"])
    multigraph.add_edges_from_array(purchase_items_array, parameter_dictionary["W_PURCHASE_ITEMS"])

    graph = nx.Graph()
    for u,v,d in multigraph.graph.edges(data=True):
        w = d['weight']
        if graph.has_edge(u,v):
            graph[u][v]['weight'] += w
        else:
            graph.add_edge(u,v,weight=w)

    return graph


def process(graph, nodes, train_set, user, k_neighbors, damping_factor, content_set, filter_dict):
    content = train_set[train_set[:,1]==user][:,0]
        
    pers = [1 if n==user else 0 for n in nodes]
    pers_dict = dict(zip(nodes, pers))
    ppr = nx.pagerank(graph, alpha=damping_factor, personalization=pers_dict)
    ppr_neighbors = dict([(node, ppr[node]) for node in content_set.index if node in ppr])
    ppr_neighbors_k = dict(sorted(ppr_neighbors.items(), key = itemgetter(1), reverse = True)[:k_neighbors])
    
    ppr_neighbors_df = pd.DataFrame.from_dict(ppr_neighbors_k, orient='index')
    score = content_set.mul(ppr_neighbors_df.iloc[:,0], axis=0).dropna().sum()
    score.loc[score.index.difference(filter_dict[user])] = 0.0
    
    recommendation_list_pos = score[score>0.0]
    recommendation_list_3 = recommendation_list_pos[(-recommendation_list_pos).argsort()][:3].index
    
    mrr_3 = 0
    if any(i in recommendation_list_3 for i in content):
        mrr_3 = 1.0/(1+min(np.where(np.isin(recommendation_list_3,content))[0]))

    return mrr_3

def gen_evaluate_with_set(graph, train_set, k_neighbors, damping_factor, content_set, filter_dict):
    graph_nodes = list(graph.nodes)
    mrrs_3 = Parallel(n_jobs=num_cores)(delayed(process)(graph, graph_nodes, train_set, user, k_neighbors, damping_factor, content_set, filter_dict) for user in np.unique(train_set[:,1]))
    return np.mean(mrrs_3)


# RS input.
content_file = purchase_items_b.set_index('user_id')
encoder = MultiLabelBinarizer()
content_file = pd.DataFrame(encoder.fit_transform(content_file['item_id'].str.split(',')), index=content_file.index, columns=encoder.classes_)
content_file = content_file.groupby(['user_id']).max()

id_user_collumn = 'user_id'
content_collumn = 'item_id'
train_filter = purchase_items_f.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
train_file = purchase_items_tr
train_file = train_file[[content_collumn, id_user_collumn]].to_numpy()

k = 90
df = 0.4


def gen_build_and_evaluate_graph(parameter_vector, train_set = train_file, k_neighbors = k, damping_factor = df, content_set = content_file, filter_dict = train_filter):
    rec_graph = build_graph(parameter_dict_from_vector(parameter_vector))
    evaluation = gen_evaluate_with_set(rec_graph, train_set, k_neighbors, damping_factor, content_set, filter_dict)
    return evaluation


def gen_build_and_evaluate_graph_idx(solution, _):
    return gen_build_and_evaluate_graph(solution)


def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    

# GA input.    
fitness_function = gen_build_and_evaluate_graph_idx

num_generations = 10
num_parents_mating = 4

sol_per_pop = 10
num_genes = 9

init_range_low = 0.01
init_range_high = 2

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "uniform"

mutation_type = "random"
mutation_percent_genes = ((1/9)*100)
random_mutation_min_val = -0.3
random_mutation_max_val = 0.3

gene_space = {'low': 0.01, 'high': 2}


best_solutions_fitness = []

stop_fitness_value = 0.0
stop_fitness_iterations = 3

def func_generation(ga_instance):
    
    best_solutions_fitness.append(ga_instance.best_solutions_fitness[-1])
    print(best_solutions_fitness)
        
    should_stop = False
    if (len(best_solutions_fitness) > stop_fitness_iterations):
        should_stop = True
        for i in range (1, stop_fitness_iterations):
            if abs(best_solutions_fitness[-i] - best_solutions_fitness[-(i+1)]) > stop_fitness_value:
                should_stop = False
    if should_stop:
        return "stop"


ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        gene_space=gene_space,
                        random_mutation_min_val = random_mutation_min_val,
                        random_mutation_max_val = random_mutation_max_val,
                        save_best_solutions=True,
                        callback_generation = func_generation
                        )



ga_instance.run()


with open('ga_instance_undirected.pickle', 'wb') as handle:
    pickle.dump(ga_instance, handle, protocol=pickle.HIGHEST_PROTOCOL)


best_solution = ga_instance.best_solutions
best_solution = np.stack(best_solution, axis=0)
best_solution_vector = best_solution[-1]

    
with open('genetic_undirected.json', 'w') as outfile:
    json.dump(parameter_dict_from_vector(best_solution_vector), outfile)

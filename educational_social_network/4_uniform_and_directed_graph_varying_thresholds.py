import numpy as np
import json
import networkx as nx
import warnings
from pandas.core.common import SettingWithCopyWarning
import pickle
from operator import itemgetter

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


''' Reading the data '''

with open('general_posts_b_t.pickle', 'rb') as handle:
    general_posts_b_t = pickle.load(handle)

with open('general_sources_b_t.pickle', 'rb') as handle:
    general_sources_b_t = pickle.load(handle)

with open('courses_follows_b_t.pickle', 'rb') as handle:
    courses_follows_b_t = pickle.load(handle)

with open('users_follows_b_t.pickle', 'rb') as handle:
    users_follows_b_t = pickle.load(handle)

with open('ratings_b_t.pickle', 'rb') as handle:
    ratings_b_t = pickle.load(handle)

with open('post_likes_b_t.pickle', 'rb') as handle:
    post_likes_b_t = pickle.load(handle)

with open('post_comments_b_t.pickle', 'rb') as handle:
    post_comments_b_t = pickle.load(handle)

with open('post_comment_likes_b_t.pickle', 'rb') as handle:
    post_comment_likes_b_t = pickle.load(handle)
    
with open('users_b.pickle', 'rb') as handle:
    users_b = pickle.load(handle)
    
    
''' Hyper Parameters '''
def vector_from_parameter_dict(parameter_dict):
    return list(map(lambda x: x[1], list(parameter_dict.items())))
def parameter_dict_from_vector(vector):
    return {
        "W_USER_USER" : vector[0],
        "W_USER_USER_BACK" : vector[1],
        "W_USER_COURSE" : vector[2],
        "W_USER_COURSE_BACK" : vector[3],
        "W_USER_POST" : vector[4],
        "W_USER_POST_BACK" : vector[5],
        "W_USER_SOURCE" : vector[6],
        "W_USER_SOURCE_BACK" : vector[7],
        "W_USER_POST_COMMENT" : vector[8],
        "W_USER_POST_COMMENT_BACK" : vector[9],
        "W_USER_POST_LIKE" : vector[10],
        "W_USER_POST_LIKE_BACK" : vector[11],
        "W_USER_POST_COMMENT_LIKE" : vector[12],
        "W_USER_POST_COMMENT_LIKE_BACK" : vector[13],
        "W_USER_SOURCE_RATING" : vector[14], #times the rating score
        "W_USER_SOURCE_RATING_BACK" : vector[15], #times the rating score
        "W_COMMENT_POST": vector[16],
        "W_COMMENT_POST_BACK" : vector[17],
        "W_USER_ORIGIN": vector[18],
        "W_USER_ORIGIN_BACK": vector[19],
        "W_USER_CITY": vector[20],
        "W_USER_CITY_BACK": vector[21]
        }


''' Building Graph '''
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
    
    multigraph = InteractionGraph()
    
    user_follow_array = users_follows_b_t[['origin_id_user', 'destination_id_user']].to_numpy()
    courses_follows_array = courses_follows_b_t[['id_course', 'id_user']].to_numpy()
    general_posts_array = general_posts_b_t[['id_general_posted', 'id_user']].to_numpy()
    general_sources_array = general_sources_b_t[['id_general_posted', 'id_user']].to_numpy()
    ratings_array = ratings_b_t[['id_general_posted', 'id_user', 'rating']].to_numpy()
    post_likes_array = post_likes_b_t[['id_general_posted', 'id_user']].to_numpy()
    post_comments_array = post_comments_b_t[['id_user', 'id_comment', 'id_general_posted']].to_numpy()
    comments_to_posts_array = post_comments_b_t[['id_comment', 'id_general_posted']].to_numpy()
    post_comment_likes_array = post_comment_likes_b_t[['id_user', 'id_comment']].to_numpy()
    user_origin_array = users_b[['id_user', 'curr_ori_id']].to_numpy()
    user_city_array = users_b[['id_user', 'city_id']].to_numpy()

    multigraph.add_nodes_from_edge_array(user_follow_array, 'user', 'user')
    multigraph.add_nodes_from_edge_array(courses_follows_array, 'course', 'user')
    multigraph.add_nodes_from_edge_array(general_posts_array, 'post', 'user')
    multigraph.add_nodes_from_edge_array(general_sources_array, 'source', 'user')
    multigraph.add_nodes_from_edge_array(ratings_array, 'rating', 'user')
    multigraph.add_nodes_from_edge_array(post_likes_array, 'post', 'user')
    multigraph.add_nodes_from_edge_array(post_comments_array, 'user', 'comment')
    multigraph.add_nodes_from_edge_array(post_comment_likes_array, 'user', 'comment')
    multigraph.add_nodes_from_edge_array(comments_to_posts_array, 'comment', 'post')
    multigraph.add_nodes_from_edge_array(user_origin_array, 'user', 'origin')
    multigraph.add_nodes_from_edge_array(user_city_array, 'user', 'city')

    multigraph.add_edges_from_array(user_follow_array, parameter_dictionary["W_USER_USER"], parameter_dictionary["W_USER_USER_BACK"])
    multigraph.add_edges_from_array(courses_follows_array, parameter_dictionary["W_USER_COURSE"], parameter_dictionary["W_USER_COURSE_BACK"])
    multigraph.add_edges_from_array(general_posts_array, parameter_dictionary["W_USER_POST"], parameter_dictionary["W_USER_POST_BACK"])
    multigraph.add_edges_from_array(general_sources_array, parameter_dictionary["W_USER_SOURCE"], parameter_dictionary["W_USER_SOURCE_BACK"])
    multigraph.add_edges_from_array(ratings_array, parameter_dictionary["W_USER_SOURCE_RATING"], parameter_dictionary["W_USER_SOURCE_RATING_BACK"], True)
    multigraph.add_edges_from_array(post_likes_array, parameter_dictionary["W_USER_POST_LIKE"], parameter_dictionary["W_USER_POST_LIKE_BACK"])
    multigraph.add_edges_from_array(post_comments_array, parameter_dictionary["W_USER_POST_COMMENT"], parameter_dictionary["W_USER_POST_COMMENT_BACK"])
    multigraph.add_edges_from_array(comments_to_posts_array, parameter_dictionary["W_COMMENT_POST"], parameter_dictionary["W_COMMENT_POST_BACK"])
    multigraph.add_edges_from_array(post_comment_likes_array, parameter_dictionary["W_USER_POST_COMMENT_LIKE"], parameter_dictionary["W_USER_POST_COMMENT_LIKE_BACK"])
    multigraph.add_edges_from_array(user_origin_array, parameter_dictionary["W_USER_ORIGIN"], parameter_dictionary["W_USER_ORIGIN_BACK"])
    multigraph.add_edges_from_array(user_city_array, parameter_dictionary["W_USER_CITY"], parameter_dictionary["W_USER_CITY_BACK"])
    
    return multigraph

class RecommendationEngine:
    
    def __init__(self, multigraph, damping_factor = 0.3):
        self.graph = nx.DiGraph()
        for u,v,d in multigraph.graph.edges(data=True):
            w = d['weight']
            if self.graph.has_edge(u,v):
                self.graph[u][v]['weight'] += w
            else:
                self.graph.add_edge(u,v,weight=w)
        self.nodes = list(self.graph.nodes)
        self.damping_factor = damping_factor
        self.user_follow_dict = {}
        for n in multigraph.graph.nodes.data():
            if n[1]['type'] == 'user':
                 self.user_follow_dict[n[0]] = set([n[0]])
        for e in multigraph.graph.edges:
            if e[0] in self.user_follow_dict:
                 self.user_follow_dict[e[0]].add(e[1])
    
    def generate_ppr(self, user, damping_factor):
        pers = [1 if n==user else 0 for n in self.nodes]
        pers_dict = dict(zip(self.nodes, pers))
        ppr = nx.pagerank(self.graph, alpha=damping_factor, personalization=pers_dict)
        ppr_sorted = dict(sorted(ppr.items(), key = itemgetter(1), reverse = True))
        ppr_list = [(k, v) for k, v in ppr_sorted.items()]
        
        return ppr_list

    def generate_recommendations(self, user, desired_types=[]):
        ppr = self.generate_ppr(user, self.damping_factor)
        return list(filter(lambda x: x[0] not in self.user_follow_dict[user] and (len(desired_types) == 0 or x[0][0] in desired_types), ppr))
    

''' Evaluation pipeline '''
def evaluate_with_set(rec_engine, test_set, content_type = 'c'):
    
    mrrs = []
    hrs = []
    for i in range(0, len(test_set)):
        edge = test_set[i]
        content = edge[0]
        user = edge[1]
    # generate N recommendations for this user
        recommendation_list = list(map(lambda x: x[0], rec_engine.generate_recommendations(user, [content_type])))
        
        mrr_t = []
        hr_t = []
        for t in range(20):
            recommendation_list_t = recommendation_list[:(t+1)]
        
            mrr = 0
            hr = 0
            if content in recommendation_list_t:
                mrr = 1.0/(1+recommendation_list_t.index(content))
                hr = 1
            mrr_t.append(mrr)
            hr_t.append(hr)
        mrrs.append(mrr_t)
        hrs.append(hr_t)        
    mrrs = np.stack(mrrs, axis=0)
    hrs = np.stack(hrs, axis=0)
    
    return (np.mean(mrrs, axis=0), np.mean(hrs, axis=0))


''' Evaluating different param dictionaries '''
test_file_path = 'courses_follows_t.pickle'
id_user_collumn = 'id_user'
content_collumn = 'id_course'
content_type = 'c'
df = 0.3


with open(test_file_path, 'rb') as handle:
    test_file = pickle.load(handle)

#Building a temp recommender for purposes of filtering test set
recommender = RecommendationEngine(build_graph(parameter_dict_from_vector(np.ones(22))), df)

test_file = test_file[test_file[id_user_collumn].isin(recommender.nodes)]
test_file = test_file[[content_collumn, id_user_collumn]].to_numpy()
    

# Uniform weights.
recommender = RecommendationEngine(build_graph(parameter_dict_from_vector(np.ones(22))), df)
evaluation = evaluate_with_set(recommender, test_file, content_type)


with open('varying_thresholds_graph-based_uniform.pickle', 'wb') as handle:
    pickle.dump(evaluation, handle, protocol=pickle.HIGHEST_PROTOCOL)


# User study weights.
with open('user_survey.json', 'r') as json_file:
    parameter_dict_study = json.load(json_file)

recommender = RecommendationEngine(build_graph(parameter_dict_study), df)
evaluation = evaluate_with_set(recommender, test_file, content_type)


with open('varying_thresholds_graph-based_study.pickle', 'wb') as handle:
    pickle.dump(evaluation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
# Genetically optimised weights
with open('genetic_courses_df.json', 'r') as json_file:
    parameter_dict_genetic = json.load(json_file)

recommender = RecommendationEngine(build_graph(parameter_dict_genetic), df)
evaluation = evaluate_with_set(recommender, test_file, content_type)


with open('varying_thresholds_directed_graph_optimized.pickle', 'wb') as handle:
    pickle.dump(evaluation, handle, protocol=pickle.HIGHEST_PROTOCOL)

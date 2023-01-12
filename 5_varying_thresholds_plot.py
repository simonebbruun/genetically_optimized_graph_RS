import pickle
import matplotlib.pyplot as plt


''' Import. '''

with open('varying_thresholds_graph-based_optimised.pickle', 'rb') as handle:
    graph_based_optimised = pickle.load(handle)
    
with open('varying_thresholds_undirected_graph_optimized.pickle', 'rb') as handle:
    undirected_graph_optimized = pickle.load(handle)
    
with open('varying_thresholds_graph-based_uniform.pickle', 'rb') as handle:
    graph_based_uniform = pickle.load(handle)
    
with open('varying_thresholds_graph-based_study.pickle', 'rb') as handle:
    graph_based_study = pickle.load(handle)
    
with open('varying_thresholds_SVD.pickle', 'rb') as handle:
    SVD = pickle.load(handle)
    
with open('varying_thresholds_NeuMF.pickle', 'rb') as handle:
    NeuMF = pickle.load(handle)

with open('varying_thresholds_NGCF.pickle', 'rb') as handle:
    NGCF = pickle.load(handle)
    
with open('varying_thresholds_most_popular.pickle', 'rb') as handle:
    most_popular = pickle.load(handle)
    
with open('varying_thresholds_KNN.pickle', 'rb') as handle:
    KNN = pickle.load(handle)
    

plt.plot(range(1,21), most_popular[0], label='Most Popular')
plt.plot(range(1,21), SVD[0], linestyle='dashed', label='SVD')
plt.plot(range(1,21), NeuMF[0], linestyle='dotted', label='NeuMF')
plt.plot(range(1,21), KNN[0], linestyle='dashdot', label='UB-KNN')
plt.plot(range(1,21), NGCF[0], marker='o', label='NGCF')
plt.plot(range(1,21), graph_based_uniform[0],  marker='|', label='Uniform Graph')
plt.plot(range(1,21), graph_based_study[0],  marker='v', color='C8', label='User Study Graph')
plt.plot(range(1,21), undirected_graph_optimized[0], marker='.', label='Genetically Undirected Graph')
plt.plot(range(1,21), graph_based_optimised[0], marker='s', label='Genetically Directed Graph')
plt.xticks([1, 5, 10, 15, 20])
plt.ylabel('MRR@k')
plt.xlabel('k')
plt.legend()
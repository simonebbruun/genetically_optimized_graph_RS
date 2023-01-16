import pickle
import matplotlib.pyplot as plt


''' Import. '''

with open('varying_thresholds_directed_graph_optimized.pickle', 'rb') as handle:
    directed_graph_optimized = pickle.load(handle)

with open('varying_thresholds_undirected_graph_optimized.pickle', 'rb') as handle:
    undirected_graph_optimized = pickle.load(handle)
    
with open('varying_thresholds_graph-based_uniform.pickle', 'rb') as handle:
    graph_based_uniform = pickle.load(handle)
    
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
    



plt.plot(range(1,6), most_popular[0], label='Most Popular')
plt.plot(range(1,6), SVD[0], linestyle='dashed', label='SVD')
plt.plot(range(1,6), NeuMF[0], linestyle='dotted', label='NeuMF')
plt.plot(range(1,6), KNN[0], linestyle='dashdot', label='UB-KNN')
plt.plot(range(1,6), NGCF[0], marker='o', label='NGCF')
plt.plot(range(1,6), graph_based_uniform[0],  marker='|', label='Uniform Graph')
plt.plot(range(1,6), undirected_graph_optimized[0], marker='.', color='tab:pink', label='Genetically Undirected Graph')
plt.plot(range(1,6), directed_graph_optimized[0], marker='s', color='tab:gray', label='Genetically Directed Graph')
plt.xticks([1, 2, 3, 4, 5])
plt.ylabel('MRR@k')
plt.xlabel('k')
plt.legend()

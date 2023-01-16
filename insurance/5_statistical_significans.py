import pickle
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar


''' Import. '''

with open('statistical_significans_directed_graph_optimized.pickle', 'rb') as handle:
    directed_graph_optimized = pickle.load(handle)
    
with open('statistical_significans_undirected_graph_optimized.pickle', 'rb') as handle:
    undirected_graph_optimized = pickle.load(handle)
    
with open('statistical_significans_graph-based_uniform.pickle', 'rb') as handle:
    graph_based_uniform = pickle.load(handle)
    
with open('statistical_significans_SVD.pickle', 'rb') as handle:
    SVD = pickle.load(handle)

with open('statistical_significans_NeuMF.pickle', 'rb') as handle:
    NeuMF = pickle.load(handle)
    
with open('statistical_significans_NGCF.pickle', 'rb') as handle:
    NGCF = pickle.load(handle)
    
with open('statistical_significans_most_popular.pickle', 'rb') as handle:
    most_popular = pickle.load(handle)
    
with open('statistical_significans_KNN.pickle', 'rb') as handle:
    KNN = pickle.load(handle)
    

''' Statistical significans. '''
# MRR@3
fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(most_popular[2], most_popular[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(SVD[2], SVD[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(NeuMF[2], NeuMF[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(NGCF[2], NGCF[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(KNN[2], KNN[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(graph_based_uniform[2], graph_based_uniform[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(undirected_graph_optimized[2], undirected_graph_optimized[0]))])
print(fvalue, pvalue)


# HR@3
data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(most_popular[2], most_popular[1]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(SVD[2], SVD[1]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(NeuMF[2], NeuMF[1]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(NGCF[2], NGCF[1]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(KNN[2], KNN[1]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(graph_based_uniform[2], graph_based_uniform[1]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=True)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[2], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(undirected_graph_optimized[2], undirected_graph_optimized[1]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=True)
print(result.pvalue)
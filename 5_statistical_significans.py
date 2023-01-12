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
    
with open('statistical_significans_graph-based_study.pickle', 'rb') as handle:
    graph_based_study = pickle.load(handle)
    
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
# MRR@5
fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(most_popular[4], most_popular[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(SVD[4], SVD[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(NeuMF[4], NeuMF[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(NGCF[4], NGCF[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(KNN[4], KNN[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(graph_based_study[4], graph_based_study[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(graph_based_uniform[4], graph_based_uniform[0]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[0]))],
                                [x for _, x in sorted(zip(undirected_graph_optimized[4], undirected_graph_optimized[0]))])
print(fvalue, pvalue)


# MRR@10
fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(most_popular[4], most_popular[1]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(SVD[4], SVD[1]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(NeuMF[4], NeuMF[1]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(NGCF[4], NGCF[1]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(KNN[4], KNN[1]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(graph_based_study[4], graph_based_study[1]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(graph_based_uniform[4], graph_based_uniform[1]))])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[1]))],
                                [x for _, x in sorted(zip(undirected_graph_optimized[4], undirected_graph_optimized[1]))])
print(fvalue, pvalue)


# HR@5
data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[2]))],
                                [x for _, x in sorted(zip(most_popular[4], most_popular[2]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[2]))],
                                [x for _, x in sorted(zip(SVD[4], SVD[2]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[2]))],
                                [x for _, x in sorted(zip(NeuMF[4], NeuMF[2]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[2]))],
                                [x for _, x in sorted(zip(NGCF[4], NGCF[2]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[2]))],
                                [x for _, x in sorted(zip(KNN[4], KNN[2]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[2]))],
                                [x for _, x in sorted(zip(graph_based_study[4], graph_based_study[2]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[2]))],
                                [x for _, x in sorted(zip(graph_based_uniform[4], graph_based_uniform[2]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=True)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[2]))],
                                [x for _, x in sorted(zip(undirected_graph_optimized[4], undirected_graph_optimized[2]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=True)
print(result.pvalue)


# HR@10
data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[3]))],
                                [x for _, x in sorted(zip(most_popular[4], most_popular[3]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[3]))],
                                [x for _, x in sorted(zip(SVD[4], SVD[3]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[3]))],
                                [x for _, x in sorted(zip(NeuMF[4], NeuMF[3]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[3]))],
                                [x for _, x in sorted(zip(NGCF[4], NGCF[3]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[3]))],
                                [x for _, x in sorted(zip(KNN[4], KNN[3]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[3]))],
                                [x for _, x in sorted(zip(graph_based_study[4], graph_based_study[3]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=False, correction=False)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[3]))],
                                [x for _, x in sorted(zip(graph_based_uniform[4], graph_based_uniform[3]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=True)
print(result.pvalue)

data_crosstab = pd.crosstab([x for _, x in sorted(zip(directed_graph_optimized[4], directed_graph_optimized[3]))],
                                [x for _, x in sorted(zip(undirected_graph_optimized[4], undirected_graph_optimized[3]))])
print(data_crosstab.loc[0,1]+data_crosstab.loc[1,0])
result = mcnemar(data_crosstab, exact=True)
print(result.pvalue)


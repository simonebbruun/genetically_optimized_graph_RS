# genetically_optimized_graph_RS
This is the source code for our paper Graph-based Recommendation for Sparse and Heterogeneous User Interactions.

## Requirements

- Pandas
- Pandasql
- NumPy
- NetworkX
- Operator
- Scikit-learn
- Joblib
- Multiprocessing
- Recommenders
- RecBole
- PyTorch
- SciPy
- PyGAD
- Pickle
- Json
- Statsmodels
- Maatplotlib


## Datasets

We use the educational social network dataset at [https://github.com/carmignanivittorio/ai_denmark_data] and the insurance dataset at [https://github.com/simonebbruun/cross-sessions_RS].


## Usage

Educational social network:
1. Split the data into training and test set using
   1_data_split.py
2. Tune the hyperparameters of the models using  
   2_graph_based_model_tune.py  
   2_KNN_tune.py  
   2_NeuMF_tune.py  
   2_NGCF_tune.py  
   2_SVD_tune.py
3. Optimize the weights of the graph-based models with genetic algorithm using
   3_directed_graph_weight_optimization.py
   3_undirected_graph_weight_optimization.py
4. Evaluate the models over the test set using  
   4_KNN_evaluation.py  
   4_KNN_varying_thresholds.py  
   4_most_popular_evaluation.py  
   4_most_popular_varying thresholds.py  
   4_NeuMF_evaluation.py  
   4_NeuMF_varying_thresholds.py  
   4_NGCF_evaluation_and_varying_thresholds.py  
   4_SVD_evaluation.py  
   4_SVD_varying_thresholds.py  
   4_undirected_graph_evaluation.py  
   4_undirected_graph_varying_thresholds.py
   4_uniform_and_directed_graph_evaluation.py
   4_uniform_and_directed_graph_varying_thresholds.py
3. Plot evaluation measures for varying thresholds and test for statistical significans using  
   5_varying_thresholds_plot.py  
   5_statistical_significans.py


Insurance:
1. Preprocess the data using
   1_data_preprocessing.py
2. Tune the hyperparameters of the models using  
   2_graph_based_model_tuning.py  
   2_KNN_tuning.py  
   2_NeuMF_tuning.py  
   2_NGCF_tuning.py  
   2_SVD_tuning.py
3. Optimize the weights of the graph-based models with genetic algorithm using
   3_directed_graph_weight_optimization.py
   3_undirected_graph_weight_optimization.py
4. Evaluate the models over the test set using  
   4_KNN_evaluation.py  
   4_most_popular_evaluation.py  
   4_NeuMF_evaluation.py  
   4_NGCF_evaluation.py  
   4_SVD_evaluation.py  
   4_undirected_graph_evaluation.py  
   4_uniform_and_directed_graph_evaluation.py  
3. Plot evaluation measures for varying thresholds and test for statistical significans using  
   5_varying_thresholds_plot.py  
   5_statistical_significans.py

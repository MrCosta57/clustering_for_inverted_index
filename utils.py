import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from Indexer import Indexer
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_repr(df: pd.DataFrame):
    """
    Get the TF-IDF representation of a Pandas Dataframe
    """
    doc_tfidf=TfidfVectorizer(lowercase=False, dtype=np.float32)

    #Computation of the sparse embedding
    sparse_doc=doc_tfidf.fit_transform(df["terms"])
    return sparse_doc, doc_tfidf.vocabulary_


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array"""

    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes.append(route)
    return routes


def TSP_solver(pairwise_distances: pd.DataFrame | np.ndarray):
    """Travelling Salesperson Problem (TSP) solver between objects"""
    data={}
    #needed to scale and make distance integer because routing solver work with int and distances are from [0, 1]
    data['distance_matrix'] = (pairwise_distances*1000).astype("int32") 
    data['num_vehicles'] = 1
    data['depot'] = 0

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    routes = get_routes(solution, routing, manager)
    
    # Display the routes.
    """ for i, route in enumerate(routes):
        print('Route', i, route)
    """
    return routes[0][:-1]


def random_search(estimator, sparse_docs: csr_matrix, std_inverted_index: dict, param_space: dict, n_iter: int, debug: bool=False, random_state:int =42):
    """ Random parameters search for the clustering estimator based on final compression ratio obtainable using VB encoding.
        It returns the best estimator NOT fitted on the data and the docid remapping dictionary.
        If `debug=True` returns a log dictionary of the times and the values
    """
    # Perform random parameter search
    best_params = None
    best_estimator = None
    best_remapping=None
    is_mixture="Mixture" in estimator.__class__.__name__

    std_index_size=Indexer.get_total_VB_enc_size(std_inverted_index)
    print(f"Standard inverted index avg posting list size: {std_index_size} bit")
    best_score=std_index_size
    
    start_time=time.process_time()
    if is_mixture:
        print("Starting LSA transformation...", end=" ")
        trunc_svd=TruncatedSVD(n_components=100, random_state=random_state) #For LSA, a value of 100 is recommended.
        sparse_docs_approx=trunc_svd.fit_transform(sparse_docs)
        sparse_docs=sparse_docs_approx.astype("float32")
        print("Done")
    end_time=time.process_time()
    lsa_delta_time=None
    if is_mixture: lsa_delta_time=end_time-start_time

    log_dict=dict(params=[], compressed_vals=[], tot_times=[], tsp_times=[], original_val=std_index_size, lsa_time=lsa_delta_time)

    total_combinations = np.prod([len(param_range) for param_range in param_space.values()])
    random_indices = np.random.choice(total_combinations, size=n_iter, replace=False)
    selected_combinations = []
    for index in random_indices:
        combination = {}
        for param, param_range in param_space.items():
            idx = index % len(param_range)
            combination[param] = param_range[idx]
            index //= len(param_range)
        selected_combinations.append(combination)

    for combination in tqdm(selected_combinations):
        # Randomly sample parameters from the search space
        if debug: log_dict["params"].append(combination)
        estimator.set_params(**combination)
        repr_elems=None
        labels=None

        start_time=time.process_time()
        if is_mixture:
            labels=estimator.fit_predict(sparse_docs)
            repr_elems=estimator.means_
        else:
            estimator.fit(sparse_docs)
            labels=estimator.labels_
            repr_elems=estimator.cluster_centers_ #kmeans.cluster_centers_[0] = centroid of cluster 0

        elem_distances=cosine_distances(repr_elems)

        tsp_start_time=time.process_time()
        tsp_solution_order=TSP_solver(elem_distances)
        tsp_end_time=time.process_time()
        tsp_delta_time=tsp_end_time-tsp_start_time

        starting_val=0
        docid_remapping={}
        for label in tsp_solution_order:
            indices=np.nonzero(labels==label)[0]
            dim=indices.shape[0]
            if dim!=0: #some clusters might be empty for some algorithms               
                distances=cosine_distances(sparse_docs[indices], repr_elems[label].reshape(1,-1)).reshape(-1)
                tmp_vals=dict(zip(indices[np.argsort(distances)], range(starting_val, starting_val+dim)))
                docid_remapping.update(tmp_vals)
                starting_val+=dim
            else:
                print(f"Cluster {label} is empty")
        end_time=time.process_time()
        delta_times=end_time-start_time

        new_inverted_index=Indexer.remap_index(std_inverted_index, docid_remapping)
        new_index_size=Indexer.get_total_VB_enc_size(new_inverted_index)
        if debug: 
            log_dict["compressed_vals"].append(new_index_size)
            log_dict["tot_times"].append(round(delta_times+(lsa_delta_time if is_mixture else 0), 4))
            log_dict["tsp_times"].append(round(tsp_delta_time, 4))

        # Check if this combination is the best so far
        if new_index_size < best_score:
            best_estimator = clone(estimator)
            best_params = combination
            best_score = new_index_size
            best_remapping=docid_remapping

            print(f"Improved avg posting list size: {new_index_size} bit ~", end="")
            print(round(((std_index_size-new_index_size)/std_index_size)*100, 4), "% reduction over the original")

    if best_estimator is None:
        print("No improvements!")
    else:
        print("Best parameters:", best_params)
        print("Best avg posting list size: ", best_score, "bit")

    if debug:
        return best_estimator, best_remapping, log_dict
    else:
        return best_estimator, best_remapping


def plot_results(log_dict: dict, method_name:str): #errorbar=None, get_k_only:int=10
    """
    Plot results of Clustering for inverted index compression analysis. 
    `get_k_only` get the top k vaue and the worst k value only, for the plots
    """
    params_name="n_components" if method_name=="Gaussian Mixture" else "n_clusters"
    params=np.array([elem[params_name] for elem in log_dict["params"]])
    tot_times=np.array(log_dict["tot_times"])
    tsp_times=np.array(log_dict["tsp_times"])
    std_index_size=log_dict["original_val"]
    compressed_vals=np.array(log_dict["compressed_vals"])
    params_index=np.argsort(params) #ordered params indices
    
    #Get compression percentage w.r.t original value
    percentage_compr=np.round((((std_index_size-compressed_vals)/std_index_size)*100), 4)
    colors=sns.color_palette("deep", n_colors=3)

    plt.figure(figsize=(8,6))
    sns.lineplot(x=params[params_index], y=percentage_compr[params_index], marker="<", color=colors[0])

    plt.title(method_name+' compression ratio')
    plt.xlabel('Number of clusters')
    plt.ylabel('Compression w.r.t original size (%)')
    plt.show()

    plt.figure(figsize=(8,6))
    sns.lineplot(x=params[params_index], y=tsp_times[params_index], marker="<", color=colors[1])
    plt.title(method_name+' TSP time spent')
    plt.xlabel('Number of clusters')
    plt.ylabel('Time spent (sec)')
    plt.show()

    plt.figure(figsize=(8,6))
    sns.lineplot(x=params[params_index], y=tot_times[params_index], marker="<", color=colors[2])

    plt.title(method_name+' total time spent')
    plt.xlabel('Number of clusters')
    plt.ylabel('Time spent (sec)')
    plt.show()
    
    min_idx=np.argmin(percentage_compr)
    max_idx=np.argmax(percentage_compr)
    print("Worst compression - params: ", params[min_idx], " value: ", percentage_compr[min_idx],  " TSP time: ", tsp_times[min_idx], " tot time: ", tot_times[min_idx])
    print("Best compression - params: ", params[max_idx], " value: ", percentage_compr[max_idx], " TSP time: ", tsp_times[max_idx], " tot time: ", tot_times[max_idx])
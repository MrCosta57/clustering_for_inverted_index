import glob, re, random
import pandas as pd
import numpy as np
from tqdm import tqdm
from Indexer import Indexer
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.sparse import csr_matrix
from sklearn.base import ClusterMixin, clone
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_data_files(file_path: str ="dataset/", format: str =".dat"):
    """
    Parse doc_id and contents of different documents in files according to this format:\n
    .I <docid>\n
    .W\n
    <textline>+\n
    <blankline>\n
    Files in the directory are read by name in lexicographic order
    """
    file_list=np.sort(glob.glob(file_path+"*"+format))
    res=pd.DataFrame([], columns=['doc_id', 'terms'])
    for f in file_list:
        with open(f, 'r') as file:
            content=file.read()
            regex=re.findall(r'\.I ([0-9]*)\n\.W\n(.*)', content)
            res=pd.concat([res, pd.DataFrame(regex, columns=['doc_id', 'terms'])], ignore_index=True, axis=0)
    res["doc_id"]=res["doc_id"].astype(int)
    res["terms"]=res["terms"].astype(str)
    return res


def get_tfidf_repr(df: pd.DataFrame):
    """
    Get the TF-IDF representation of a Pandas Dataframe
    """
    doc_tfidf=TfidfVectorizer(lowercase=False, dtype=np.float32)

    #Computation of the sparse embedding
    sparse_doc=doc_tfidf.fit_transform(df["terms"])
    return sparse_doc, doc_tfidf.vocabulary_


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""

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


''' def sort_csr_by_nonzero(matrix: csr_matrix) -> csr_matrix:
    """
    Sort in decreasing order the CSR matrix by the total number of non zero elem in each row 
    """
    # Count the number of non-zero entries in each row.
    nonzero_counts = matrix.getnnz(axis=1)
    # Get the sorted indices based on the counts.
    sorted_indices = np.argsort(nonzero_counts)[::-1]
    # Return the sorted matrix.
    return matrix[sorted_indices]

def stream_cluster(sorted_collection: csr_matrix, radius: float):
    """Cluster a sorted list of objects based on their distance to each other. 
    `sorted_collection` must be TD-IDF vectors normalized"""
    C=[] #set of all clusters
    cluster=[]
    for i, d in enumerate(sorted_collection):
        if i==0:
            cluster=[d]
            C.append(cluster)
        else:        
            dist_c=np.min([cosine_distances(c[0], d) for c in C])
            if dist_c<=radius:
                cluster.append(d)
            else:
                cluster=[d]
                C.append(cluster)
    return C '''


def random_search(estimator: ClusterMixin, sparse_docs: csr_matrix, std_inverted_index: dict, param_space: dict, n_iter: int, n_jobs: int = -1):
    """ Random parameters search for the clustering estimator based on the silhouette score.
        It returns the best estimator NOT fitted on the data and the docid remapping dictionary"""
    # Perform random parameter search
    best_params = None
    best_estimator = None
    std_index_size=Indexer.get_total_VB_enc_size(std_inverted_index)
    best_score=std_index_size
    best_remapping=None
    print(f"Standard inverted index dimension: {best_score} Bytes")

    for _ in tqdm(range(n_iter)):
        # Randomly sample parameters from the search space
        sampled_params = {param: random.choice(values) for param, values in param_space.items()}
        
        if "n_jobs" in estimator.get_params().keys():
            estimator.set_params(**sampled_params, n_jobs=n_jobs)
        else:
            estimator.set_params(**sampled_params)

        estimator.fit(sparse_docs)
        
        repr_elems=None
        is_dbscan="DBSCAN" in estimator.__class__.__name__
        if is_dbscan:
            core_labels=estimator.labels_[estimator.core_sample_indices_]
            core_index_list=[]
            for label in np.unique(core_labels):
                indices=np.nonzero(core_labels==label)[0]
                label_indices=estimator.core_sample_indices_[indices] #core sample with current label
                index=np.random.choice(label_indices) #randomly select one core sample
                core_index_list.append(index)
            repr_elems=sparse_docs[core_index_list] #list of core points (one that represents one cluster)
            if repr_elems.shape[0]==0:
                print("No core points found!")
                continue
        else:
            repr_elems=estimator.cluster_centers_ #kmeans.cluster_centers_[0] = centroid of cluster 0

        elem_distances=cosine_distances(repr_elems) 
        tsp_solution_order=TSP_solver(elem_distances)
        if is_dbscan: tsp_solution_order=tsp_solution_order+[-1]

        starting_val=0
        docid_remapping={}
        for label in tsp_solution_order:
            indices=np.nonzero(estimator.labels_==label)[0] #-1 is the noise cluster
            dim=indices.shape[0]
            if dim!=0: #some clusters might be empty 
                tmp_vals=None
                if label!=-1:
                    distances=cosine_distances(sparse_docs[indices], repr_elems[label].reshape(1,-1)).reshape(-1)
                    tmp_vals=dict(zip(indices[np.argsort(distances)], range(starting_val, starting_val+dim)))
                else:
                    permutation=np.random.permutation(range(starting_val, starting_val+dim))
                    tmp_vals=dict(zip(indices, permutation))
                docid_remapping.update(tmp_vals)
                starting_val+=dim
            else:
                print(f"Cluster {label} is empty")

        new_inverted_index=Indexer.remap_index(std_inverted_index, docid_remapping)
        new_index_size=Indexer.get_total_VB_enc_size(new_inverted_index)

        """ with open(input_path+"k_means_remapping.pkl", "wb") as file:
            pickle.dump(k_means_docid_remapping, file) """
        
        # Check if this combination is the best so far
        if new_index_size < best_score:
            best_estimator = clone(estimator)
            best_params = sampled_params
            best_score = new_index_size
            best_remapping=docid_remapping

            print(f"Improved inverted index dimension: {new_index_size} Bytes ~", end="")
            print(round(((std_index_size-new_index_size)/(std_index_size+new_index_size))*100, 3), "% reduction over the original")

    if best_estimator is None:
        print("No improvements!")
    else:
        print("Best parameters:", best_params)
        print("Best index dimension: ", best_score, "Bytes")

    return best_estimator, best_remapping

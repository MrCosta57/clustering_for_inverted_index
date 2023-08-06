import glob, re
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_file_list(file_path: str ="dataset/", format: str =".dat"):
    file_list=glob.glob(file_path+"*"+format)
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
    data['distance_matrix'] = pairwise_distances
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


def sort_csr_by_nonzero(matrix: csr_matrix) -> csr_matrix:
    # Count the number of non-zero entries in each row.
    nonzero_counts = matrix.getnnz(axis=1)
    # Get the sorted indices based on the counts.
    sorted_indices = np.argsort(nonzero_counts)[::-1]
    # Return the sorted matrix.
    return matrix[sorted_indices]


def stream_cluster(sorted_collection: csr_matrix, radius: float):
    """Cluster a sorted list of objects based on their distance to each other"""
    C=[]
    cluster=[]
    for i, d in enumerate(sorted_collection):
        if i==0:
            cluster=[d]
            C.append(cluster)
        else:        
            dist_c=np.min([cosine_distances(c[0], d) for c in C]) #c[0] is the medoid of the cluster
            if dist_c<=radius:
                cluster.append(d)
            else:
                cluster=[d]
                C.append(cluster)
    return C
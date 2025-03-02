# import os
# import glob

# def check_join_predicates_folder(base_dir="."):
#     """
#     Check if join_predicates folder exists in PQO directory for each query and method
    
#     Parameters:
#     base_dir (str): Base directory to start search from
    
#     Returns:
#     dict: Results showing existence of join_predicates folder for each query and method
#     """
#     # Store results
#     results = {}
    
#     # Find all imdb_*_original directories
#     query_dirs = glob.glob(os.path.join(base_dir, "imdb_*_original"))
    
#     for query_dir in query_dirs:
#         # Extract query_id from directory name
#         query_id = query_dir.split("_")[1]
#         results[query_id] = {}
        
#         # Check each method directory
#         method_dirs = os.listdir(query_dir)
#         for method in method_dirs:
#             pqo_path = os.path.join(query_dir, method, "inputs", "PQO")
#             join_predicates_path = os.path.join(pqo_path, "join_predicates")
            
#             # Check if the paths exist
#             has_pqo = os.path.exists(pqo_path)
#             has_join_predicates = os.path.exists(join_predicates_path)
            
#             results[query_id][method] = {
#                 "has_pqo": has_pqo,
#                 "has_join_predicates": has_join_predicates
#             }

#     return results

# def print_results(results):
#     """
#     Print only cases where join_predicates folder is missing
    
#     Parameters:
#     results (dict): Results from check_join_predicates_folder
#     """
#     missing_cases = []
    
#     for query_id, methods in results.items():
#         for method, status in methods.items():
#             if status['has_pqo'] and not status['has_join_predicates']:
#                 missing_cases.append((query_id, method))
    
#     if missing_cases:
#         print("\nMissing join_predicates folders:")
#         print("="*50)
#         for query_id, method in missing_cases:
#             print(f"Query {query_id} - Method {method}")
#     else:
#         print("all include!")

# def main():
#     # Check current directory
#     results = check_join_predicates_folder()
#     print_results(results)
    
#     # Count queries missing join_predicates
#     missing_join_predicates = []
#     for query_id, methods in results.items():
#         for method, status in methods.items():
#             if status['has_pqo'] and not status['has_join_predicates']:
#                 missing_join_predicates.append((query_id, method))
    
#     if missing_join_predicates:
#         print("\nQueries missing join_predicates folder:")
#         print("="*50)
#         for query_id, method in missing_join_predicates:
#             print(f"Query {query_id} - Method {method}")

# if __name__ == "__main__":
#     main()

import os
import glob

def check_join_predicates_folder(base_dir="."):
    """
    Check if join_predicates folder exists in PQO directory for each query and method
    
    Parameters:
    base_dir (str): Base directory to start search from
    
    Returns:
    dict: Results showing existence of join_predicates folder for each query and method
    """
    # Store results
    results = {}
    
    # Find all imdb_*_original directories
    query_dirs = glob.glob(os.path.join(base_dir, "imdb_*_original"))
    
    for query_dir in query_dirs:
        # Extract query_id from directory name
        query_id = query_dir.split("_")[1]
        results[query_id] = {}
        
        # Check each method directory
        method_dirs = os.listdir(query_dir)
        for method in method_dirs:
            pqo_path = os.path.join(query_dir, method, "inputs", "PQO")
            join_predicates_path = os.path.join(pqo_path, "join_predicates")
            
            # Check if the paths exist
            has_pqo = os.path.exists(pqo_path)
            has_join_predicates = os.path.exists(join_predicates_path)
            
            results[query_id][method] = {
                "has_pqo": has_pqo,
                "has_join_predicates": has_join_predicates
            }

    return results

def print_results(results):
    """
    Print only cases where join_predicates folder is missing
    
    Parameters:
    results (dict): Results from check_join_predicates_folder
    """
    missing_cases = []
    
    for query_id, methods in results.items():
        for method, status in methods.items():
            if status['has_pqo'] and not status['has_join_predicates']:
                missing_cases.append((query_id, method))
    
    if missing_cases:
        print("\nMissing join_predicates folders:")
        print("="*50)
        for query_id, method in missing_cases:
            print(f"Query {query_id} - Method {method}")
    else:
        print("all include!")

def main():
    # Check current directory
    results = check_join_predicates_folder()
    print_results(results)
    
    # Count queries missing join_predicates
    missing_join_predicates = []
    for query_id, methods in results.items():
        for method, status in methods.items():
            if status['has_pqo'] and not status['has_join_predicates']:
                missing_join_predicates.append((query_id, method))
    
    if missing_join_predicates:
        print("\nQueries missing join_predicates folder:")
        print("="*50)
        for query_id, method in missing_join_predicates:
            print(f"Query {query_id} - Method {method}")

if __name__ == "__main__":
    main()
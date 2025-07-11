import collections
import csv
import random
import subprocess
import collections
import json
import logging
import os
import numpy as np
import psycopg2
import datetime

from absl import app
from absl import flags

from kepler.training_data_collection_pipeline import query_utils
import re


# Define general flags
_SELECTION = flags.DEFINE_string(
    "selection", None, 
    "Cardinality, cardinality_full, kepler, or csv.")
flags.mark_flag_as_required("selection")
_RANDOM_SEED = flags.DEFINE_integer(
    "random_seed", 2024,
    "random seed setting")
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")
_DB_NAME = flags.DEFINE_string(
    "db_name", 'imdbloadbase',
    "database name.")

# query related
_TEMPLATE_FILE = flags.DEFINE_string(
    "template_file", None, 
    "Parameterized query template file.")
flags.mark_flag_as_required("template_file")
_QUERY_ID = flags.DEFINE_string(
    "query_id", None, 
    "Name of query to train on")
flags.mark_flag_as_required("query_id")
# csv related flags - required for csv
_METADATA_FILE = flags.DEFINE_string(
    "metadata_file", None, 
    "Template's metadata: params with frequency.")

# training & testing related
_COUNT = flags.DEFINE_integer(
    "count", 2200, 
    "The max number of parameters to generate per query.")
_TEST_SIZE = flags.DEFINE_integer(
    "test_size", 200, 
    "Number of test queries.")

# csv related flags - special table case it
EXTRA_TABLE = ['it_miidx', 'it_mi', 'it_pi']

available_it_for_mi = ['genres'] * 13 + ['budget'] * 2 + ['release dates'] * 20 + ['countries'] * 10
available_it_for_pi = ['mini biography'] * 3 + ['trivia'] * 2 + ['height'] * 1
available_it_for_miidx = ['top 250 rank'] * 2 + ['bottom 10 rank'] * 2 + ['rating'] * 16 + ['votes'] * 11

cct_for_cc_subject_id = ["IN ('cast', 'crew')"] * 4 + ["= 'cast'"] * 12 + ["!= 'complete+verified'"] * 2 
cct_for_cc_status_id = ["= 'complete'"] * 3 + ["LIKE '%complete%'"] * 7 + [" = 'complete+verified'"] * 9

# kepler template format additional information
INT_TABLE_DICT = {
    "t": {
        "production_year": {
            "min": 1880,
            "max": 2019
        },
        "episode_nr": {
            "min": 1,
            "max": 91821
        }
    }
}


###################
# helper methods
# csv related, check if the param-frequency file is provided
def check_required_values(metadata_file):
    """
    Check if the required variables have values. If any of them is missing, raise an error.

    Args:
    metadata_file: Path to the metadata file.

    Raises:
    ValueError: If any of the required variables is missing.
    """
    
    if not metadata_file:
        raise ValueError("Metadata file (_METADATA_FILE) is missing or not provided.")
    
    print("All required values are provided.")



def escape_single_quotes(param):
    """
    Escapes single quotes in a string parameter by replacing each single quote with two single quotes.    
    Args:
        param: The parameter to escape. Can be any type, but only strings will be modified.
        
    Returns:
        If param is a string, returns a new string with all single quotes doubled.
        Otherwise, returns the original param unchanged.
        
    Examples:
        >>> escape_single_quotes("O'Reilly")
        "O''Reilly"
        >>> escape_single_quotes(123)
        123
    """
    # If param is a string and contains single quotes, replace them with two single quotes
    if isinstance(param, str):
        return param.replace("'", "''")
    return param



def save_metadata_to_csv(output_dir, metadata, query_id):
    """
    Saves the metadata to a CSV file in the output directory with the query_id as filename.
    
    This function creates a two-column CSV file (Key, Value) where each row represents
    a key-value pair from the metadata dictionary.
    
    Args:
        output_dir (str): The directory path where the metadata CSV file will be saved.
        metadata (dict): A dictionary containing the metadata information to save.
        query_id (str): Identifier used to name the output CSV file.
    
    Returns:
        None: The function doesn't return a value but prints confirmation message.
    
    Example:
        >>> metadata = {"timestamp": "2023-04-01", "status": "complete"}
        >>> save_metadata_to_csv("/path/to/output", metadata, "query123")
        Metadata saved to /path/to/output/query123.csv
    """
    # Create the full path for the metadata file using the query_id
    metadata_file = os.path.join(output_dir, f"{query_id}.csv")
    
    # Write the metadata to CSV
    with open(metadata_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(["Key", "Value"])
        
        # Write metadata
        for key, value in metadata.items():
            writer.writerow([key, value])

    print(f"Metadata saved to {metadata_file}")



###################
# Cardinality related
def get_param_and_cardinality(db_name, single_select_table, single_select_column, select_column, table, not_null_column, join_condition, value_type="count"):
    """
    Step 1: Retrieves parameters and their cardinality from a database using SQL queries.
    
    This function executes SQL queries to fetch column values and their counts from database tables.
    It handles different join conditions, provides fallback query execution, and can return either
    raw counts or tokenized results with wildcards for pattern matching.
    
    Args:
        db_name (str): Database name to connect to.
        single_select_table (str): Table name for simplified query fallback.
        single_select_column (str): Column name for simplified query fallback.
        select_column (str): Column name to select and group by in the primary query.
        table (str): Main table name for the query.
        not_null_column (str): WHERE condition for filtering non-null values.
        join_condition (str): SQL join condition string.
        value_type (str, optional): Output format - "count" for raw counts or "tokenizer" 
                                    for tokenized results. Defaults to "count".
    
    Returns:
        list: A list of tuples. For value_type="count", returns (value, count) pairs.
              For value_type="tokenizer", returns (wildcard_token, count) pairs with
              values tokenized and formatted with '%' wildcards.
    """
    # connect to database
    db_config = query_utils.DatabaseConfiguration(
      dbname=db_name, user="kepler", password="kepler", host="localhost"
    )
    query_manager = query_utils.QueryManager(db_config)

    # original query
    if 'outer join' in join_condition.lower():
        def clean_not_null_condition(condition):
            condition = condition.strip()
            if condition.lower().endswith('and'):
                condition = condition[:-3].strip() # remove AND
            return condition
        cleaned_not_null_column = clean_not_null_condition(not_null_column)
        
        query = f"""SELECT {select_column}, COUNT(*) as count_all 
                    FROM {single_select_table} 
                    {join_condition} 
                    WHERE {cleaned_not_null_column} 
                    GROUP BY {select_column}"""
    else:
        query = f"""SELECT {select_column}, COUNT(*) as count_all 
                    FROM {table} 
                    WHERE {not_null_column} 
                    {join_condition} 
                    GROUP BY {select_column}"""

    addition_query = " ORDER BY COUNT(*) DESC"
    print(f"---- query: {query} {addition_query}")
    
    # execute query and get result
    try:
        full_query = query + addition_query
        result = query_manager.execute(full_query)
    except Exception as e:
        error_type = type(e).__name__
        logging.error(f"{error_type}: {e}")
        result = []
    
    # if original query has any error, execute simple version without multiple group conditions
    if len(result) == 0:
        query = f"""SELECT {single_select_column}, COUNT(*) as count_all FROM {table}
            WHERE {not_null_column}
            {join_condition}
            GROUP BY {single_select_column} """
        print(f"---- query: {query} {addition_query}")
        
        try:
            full_query = query + addition_query
            result = query_manager.execute(full_query)
        except Exception as e:
            error_type = type(e).__name__
            logging.error(f"{error_type}: {e}")
            return []

    # get result
    print(f"---- Most popular sample: {result[0]}")
    
    # tokenizer method, split everything except for text
    def tokenizer(value):
        return re.split(r'[!@#$%^&*()_\-|\?<>,.:;"\'{}\[\]~`/\s]+', value)
    
    from decimal import Decimal
    if value_type == "count":
        return [
            (
                int(value[0]) if isinstance(value[0], (Decimal, float)) and not isinstance(value[0], int)
                else value[0].strftime('%Y-%m-%d') if isinstance(value[0], datetime.date)
                else value[0],
                value[-1]
            )
            for value in result
        ]
    elif value_type == "tokenizer":
        tokenized_result = []
        total_length = len(result)
        print(f"result length: {total_length}")
        
        # MARK: if the result is too large - 100 million - avoid kill/oom error
        if total_length > 10000000:
            # re-execute, get percentiles
            quartile_results = query_manager.execute(f"""
                WITH counts AS (
                    {query}
                )
                SELECT percentile_cont(0.1) WITHIN GROUP (ORDER BY count_all) as q1,
                    percentile_cont(0.2) WITHIN GROUP (ORDER BY count_all) as q2,
                    percentile_cont(0.3) WITHIN GROUP (ORDER BY count_all) as q3,
                    percentile_cont(0.4) WITHIN GROUP (ORDER BY count_all) as q4,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY count_all) as q5,
                    percentile_cont(0.6) WITHIN GROUP (ORDER BY count_all) as q6,
                    percentile_cont(0.7) WITHIN GROUP (ORDER BY count_all) as q7,
                    percentile_cont(0.8) WITHIN GROUP (ORDER BY count_all) as q8,
                    percentile_cont(0.9) WITHIN GROUP (ORDER BY count_all) as q9
                FROM counts;
            """)
            
            if len(quartile_results) == 1:
                q1, q2, q3, q4, q5, q6, q7, q8, q9 = quartile_results[0]
            else:
                raise ValueError("Expected exactly one row of quartile results")
                
            # stratified sampling - avoid kill/oom error
            stratified_query = f"""
                WITH counts AS (
                    {query}
                ),
                stratified AS (
                    SELECT *,
                        CASE 
                            WHEN count_all <= {q1} THEN 1
                            WHEN count_all <= {q2} THEN 2
                            WHEN count_all <= {q3} THEN 3
                            WHEN count_all <= {q4} THEN 4
                            WHEN count_all <= {q5} THEN 5
                            WHEN count_all <= {q6} THEN 6
                            WHEN count_all <= {q7} THEN 7
                            WHEN count_all <= {q8} THEN 8
                            WHEN count_all <= {q9} THEN 9
                            ELSE 10
                        END as stratum,
                        random() as rand
                    FROM counts
                )
                SELECT *
                FROM stratified
                WHERE random() <= 0.1  -- 10% sampling rate
                ORDER BY stratum, rand
            """
            print(stratified_query)
            result = query_manager.execute(stratified_query)
            print(f"modified result length - after stratified sampling: {len(result)}")
            
            # remove stratum & rand (last two columns)
            result = [columns[:-2] for columns in result]
        
        for value in result: # "value" is a [string, card]; "result" is a list of unqiue [string, card]
            tokens = tokenizer(str(value[0]))  # param tokenizer
            tokens = [token for token in tokens if token] # remove empty token
            # print(f"current token length: {len(tokens)}")
            
            # If length of tokens is 1, split the token into individual characters
            if len(tokens) == 1:
                tokens = [random.choice(tokens[0])] # randomly choose one of the character
            
            tokens = [f"%{token}%" for token in tokens] # add wildcards
            for token in tokens:
                tokenized_result.append((token, value[-1]))

        return tokenized_result



def get_param_and_cardinality_full(db_name, base_predicate_column, from_clause, full_where_conditions):
    """
    Retrieves column values and their occurrence counts from a database table.
    
    This function executes SQL queries to fetch specified column values and count their occurrences,
    grouping by the column values. It includes error handling with detailed logging and processes
    the query results to standardize data types (converting Decimals to integers and formatting dates).
    
    Args:
        db_name (str): Database name to connect to.
        base_predicate_column (str): Column name to select and group by.
        from_clause (str): SQL FROM clause specifying tables and joins.
        full_where_conditions (str): SQL WHERE conditions for filtering records.
        
    Returns:
        list: A list of tuples where each tuple contains (column_value, count). 
              The column_value may be processed (e.g., Decimals converted to int, 
              dates formatted as strings). Returns an empty list if query fails.
    """
    db_config = query_utils.DatabaseConfiguration(
      dbname=db_name, user="kepler", password="kepler", host="localhost"
    )
    query_manager = query_utils.QueryManager(db_config)

    # original query
    query = f"""SELECT {base_predicate_column}, COUNT(*) as count_all 
                {from_clause} 
                {full_where_conditions} 
                GROUP BY {base_predicate_column}"""

    addition_query = " ORDER BY COUNT(*) DESC"
    print(f"---- query: {query} {addition_query}")
    
    try:
        full_query = query + addition_query
        result = query_manager.execute(full_query)
        print(f"CURRENT RESULT LENGTH: {len(result)}\n")
    except Exception as e:
        error_type = type(e).__name__
        logging.error(f"{error_type}: {e}")
        result = []
    
    if len(result) == 0:
        return []

    print(f"---- Most popular sample: {result[0]}")

    from decimal import Decimal
    
    # Process each row of results
    processed_results = []
    for row in result:
        # Convert all values in the row
        processed_values = []
        
        # Process all values except the last one (count)
        for value in row[:-1]:
            # Handle Decimal and float types
            if (isinstance(value, (Decimal, float)) and not isinstance(value, int)):
                processed_values.append(int(value))
            # Handle date type
            elif isinstance(value, datetime.date):
                processed_values.append(value.strftime('%Y-%m-%d'))
            # Other types remain unchanged
            else:
                processed_values.append(value)
        
        # Add as (values, count)
        processed_results.append((tuple(processed_values), row[-1]))
    
    return processed_results



def create_buckets(data, num_buckets):
    """
    Step 2: Divides data points into a specified number of buckets based on the second value of each data point.
    
    This function distributes data items into buckets by value range. If all values are identical,
    it distributes items evenly across buckets. Otherwise, it creates buckets of equal value ranges
    and assigns items to the appropriate bucket based on their value.
    
    Args:
        data (list): List of tuples where each tuple's last element is the value used for bucketing.
        num_buckets (int): Number of buckets to create.
        
    Returns:
        list: A list of bucket lists, where each bucket contains data points assigned to it.
    """
    # Extract the second values from the pairs
    second_values = [d[-1] for d in data]
    
    # Find the minimum and maximum of the second values
    min_value = min(second_values)
    max_value = max(second_values)
    print(f"!!! Cardinality range of current column [{min_value}, {max_value}]")
    
    # Calculate the interval width
    interval_width = (max_value - min_value) / num_buckets
    
    # Initialize the buckets
    buckets = [[] for _ in range(num_buckets)]
    
    if interval_width == 0:
        # When all values are the same, distribute items evenly across buckets
        for i, item in enumerate(data):
            buckets[i % num_buckets].append(item)
    else:
        # Assign each item to the appropriate bucket based on interval width
        for item in data:
            _, value = item
            bucket_index = min(int((value - min_value) / interval_width), num_buckets - 1)
            buckets[bucket_index].append(item)
    
    return buckets



def sample_from_buckets(buckets, total_samples):
    """
    Step 3: Samples data points from multiple buckets with replacement to create a representative dataset.
    
    This function distributes the total number of samples evenly across non-empty buckets and 
    randomly selects items from each bucket with replacement. It ensures the final sample size
    matches the requested total by distributing any remainder samples across buckets and 
    truncating if necessary.
    
    Args:
        buckets (list): List of bucket lists, where each bucket contains data points.
        total_samples (int): Total number of samples to collect across all buckets.
        
    Returns:
        list: A combined list of sampled data points from all buckets, with exactly
                total_samples number of items.
    """
    # Sample based on non-empty bucket
    non_empty_buckets = [bucket for bucket in buckets if bucket]
    num_non_empty_buckets = len(non_empty_buckets)
    samples_per_bucket = total_samples // num_non_empty_buckets
    remainder_samples = total_samples % num_non_empty_buckets
    
    sampled_data = []
    
    # Sample from each bucket
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue  # Skip empty bucket
        
        num_samples = samples_per_bucket + (1 if i < remainder_samples else 0)
        
        # with place back - change: use random.choices instead of choice
        sampled_data.extend(random.choices(bucket, k=num_samples))
    
    # Ensure the result is exactly total_samples in case of rounding issues
    return sampled_data[:total_samples]



# MARK: add CDF implementation
def get_range_param_and_cardinality(data, operator):
    """
    Generate param and cumulative cardinality adjustments based on the operator.
    
    Args:
    - data: The input list of tuples (param, cardinality).
    - operator: The operator type ('>', '>=', '<', '<=', '=').
    
    Returns:
    - A list of tuples (param, cardinality), where `param` is the value,
      and `cardinality` is the cumulative count based on the operator.
      
    Example 1:
    - Input: 
        - data = [(1, 10), (2, 20), (3, 30), (4, 40)]
        - operator = "<" -> "< param"
    - Output:
        - data = [(1, 0), (2, 10), (3, 30), (4, 60)]  # Cumulative without including the current
    
    Example 2:
    - Input: 
        - data = [(1, 10), (2, 20), (3, 30), (4, 40)]
        - operator = ">=" -> ">= param"
    - Output:
        - data = [(1, 100), (2, 90), (3, 70), (4, 40)]
    
    """
    # Sort the data by the param value
    data_sorted = sorted(data, key=lambda x: x[0])  # Sort by param (first element of tuple)
    
    # Prepare the param_cardinality list
    param_cardinality = []
    
    # Calculate cumulative cardinality based on the operator
    if operator == '>':
        cumulative = 0
        for param, cardinality in reversed(data_sorted):  # Start from the largest value and go downwards
            param_cardinality.append((param, cumulative))  # Append the cumulative before adding current cardinality
            cumulative += cardinality
        param_cardinality.reverse()  # Reverse back to maintain increasing order by value
    
    elif operator == '>=':
        cumulative = 0
        for param, cardinality in reversed(data_sorted):
            cumulative += cardinality
            param_cardinality.append((param, cumulative))
        param_cardinality.reverse()
    
    elif operator == '<':
        cumulative = 0
        for param, cardinality in data_sorted:  # Start from the smallest value and go upwards
            param_cardinality.append((param, cumulative))  # Append the cumulative before adding current cardinality
            cumulative += cardinality
    
    elif operator == '<=':
        cumulative = 0
        for param, cardinality in data_sorted:
            cumulative += cardinality
            param_cardinality.append((param, cumulative))
    
    elif operator == '=':
        param_cardinality = [(param, cardinality) for param, cardinality in data_sorted]

    return param_cardinality



def sample_in_operator(buckets, total_samples, num_samples_per_bucket=5, shuffle=True):
    """
    Creates IN operator combinations by sampling multiple items from each bucket.
    
    This function generates combinations for SQL IN operators by sampling multiple items
    from each non-empty bucket. It creates multiple rounds of sampling to reach the desired
    total number of combinations. Each combination contains num_samples_per_bucket items
    from a single bucket, using controlled random selection with seeds for reproducibility.
    
    Args:
        buckets (list): List of bucket lists, where each bucket contains data points.
        total_samples (int): Total number of IN operator combinations to generate.
        num_samples_per_bucket (int, optional): Number of items to include in each IN operator.
                                                Defaults to 5. DSB's default use 3
        shuffle (bool, optional): Whether to shuffle the final result. Defaults to True.
        
    Returns:
        list: A list of IN operator combinations, where each combination is a list of
                values to be used in an IN clause. Returns exactly total_samples combinations.
    """
    # Sample based on non-empty bucket
    non_empty_buckets = [bucket for bucket in buckets if bucket]
    num_non_empty_buckets = len(non_empty_buckets)
    
    # Sample rounds
    num_rounds = total_samples // num_non_empty_buckets
    remainder_rounds = total_samples % num_non_empty_buckets

    sampled_data = []
    
    # Sample from each bucket for num_rounds times, each with num_samples_per_bucket data
    for i, bucket in enumerate(buckets):        
        if not bucket:
            continue  # Skip empty bucket
        
        num_combination = num_rounds + (1 if i < remainder_rounds else 0)
        seed = _RANDOM_SEED.value
        
        for _ in range(num_combination):
            current_in_tuple = []
            local_random = random.Random(seed)
            seed += 1
            
            current_in_tuple.extend(
                [item[0] for item in local_random.choices(bucket, k=num_samples_per_bucket)]
            )
            sampled_data.append(current_in_tuple)

    if shuffle:
        random.shuffle(sampled_data)
    return sampled_data[:total_samples]



def format_in_clause_data(in_clause_data):
    """
    Processes each nested list in in_clause_data, formats it as 'a', 'b', 'c',
    and returns the list of formatted strings ready for SQL IN clause.
    
    Args:
    - in_clause_data: A nested list of lists containing strings to be formatted.
    
    Returns:
    - A list of formatted strings, where each string corresponds to a formatted group.
    """
    formatted_list = []

    # Iterate over each list in in_clause_data
    for literal in in_clause_data:
        # Format each group as a single string with elements separated by ', '
        
        formatted_group = ', '.join([f"'{escape_single_quotes(str(item))}'" for item in literal])
        
        # Remove the head and tail single quotes
        formatted_group = formatted_group[1:-1]
        
        formatted_list.append(f"{formatted_group}")
    
    # print(formatted_list[:5])
    return formatted_list



def gen_param_by_cardinality(db_name, template_file, N=10, K=50, debug=True):
    """
    Generates parameter values for database query templates based on column cardinality.
    
    This function processes query templates and creates parameter lists by analyzing 
    column value distributions in the database. It divides values into buckets based on 
    cardinality and samples from these buckets to ensure a representative parameter set.
    The function handles various operators (=, LIKE, IN, etc.) and joins differently.
    
    Args:
        db_name (str): Database name to connect to.
        template_file (str): Path to the JSON file containing query templates.
        N (int, optional): Number of buckets to split the cardinality range. Defaults to 10.
        K (int, optional): Total number of samples to generate per parameter. Defaults to 50.
        debug (bool, optional): Whether to print debug information. Defaults to True.
        
    Returns:
        list: A list of parameter lists, where each parameter list corresponds to a predicate
                in the query templates. Format: [[@param1_1, @param1_2, ..., @param1_K], 
                [@param2_1, ..., @param2_K], ...].
    """
    with open(template_file) as f:
        all_param_lists = []
        templates = json.load(f)
        for query_id, template in templates.items():
            for predicate in template['predicates']:
                # enumerate all neighbor tables joining with the current one 
                # and finally mix all params together
                temp_params_list_mix = []
                for i in range(len(predicate['join_tables_alias'])):
                    # add alias to avoid ambiguity
                    select_table_alias = predicate['alias']
                    join_table_alias = predicate['join_tables_alias'][i]
                    
                    # conditions
                    left_or_right = predicate['left_or_right'][i]
                    join_condition = predicate['join_conditions'][i]
                    
                    # possibly no alias
                    select_table_stmt = predicate['table'] + " AS " + select_table_alias + ", " if select_table_alias else predicate['table'] + ", "
                    join_table_stmt = predicate['join_tables'][i] + " AS " + join_table_alias if join_table_alias else predicate['join_tables'][i]
                    from_table = select_table_stmt + join_table_stmt
                    
                    select_operator = predicate['operator']
                    select_data_type = predicate['data_type']
                    
                    
                    # we don't have left_or_right == "r"
                    # Since based on the setup, no matter what join_conditions, we always take the predicate as input
                    # which means always l or both
                    if left_or_right == 'both':
                        select_column = f"{select_table_alias}.{predicate['column']}" if select_table_alias else predicate['column']
                        not_null_column = f"{select_table_alias}.{predicate['column']} IS NOT NULL AND " if select_table_alias else f"{predicate['column']} IS NOT NULL AND "
                        # it's possible that the joining table has multiple selections on multiple columns
                        for column_ in predicate['join_tables_column'][i]:
                            select_column += f", {join_table_alias}.{column_}" if join_table_alias else f", {column_}"
                            not_null_column += f"{join_table_alias}.{column_} IS NOT NULL AND " if join_table_alias else f"{column_} IS NOT NULL AND "
                            
                    elif left_or_right == 'l':
                        select_column = f"{select_table_alias}.{predicate['column']}" if select_table_alias else predicate['column']
                        not_null_column = f"{select_table_alias}.{predicate['column']}" + ' IS NOT NULL AND ' if select_table_alias else f"{predicate['column']} IS NOT NULL AND "
                    
                    single_select_table = predicate['table'] + " AS " + select_table_alias if select_table_alias else predicate['table']
                    single_select_column = f"{select_table_alias}.{predicate['column']}" if select_table_alias else predicate['column']
                    
                    # Get cardinality regarding operator type
                    if select_operator in ["LIKE", "NOT LIKE"]:
                        value_type = "tokenizer" # For now, all the others ("=", "IN", <, >, etc) belong to "count"
                    else:
                        value_type = "count"
                        
                    print(f"current value_type is {value_type}")

                    # MARK: 'it' is a special joining table that requires specific predicates (imdbloadbase)
                    if select_table_alias == 'it' and join_table_alias in ['mi', 'miidx', 'pi'] and select_operator == '=':
                        print("special table it")
                        if join_table_alias == 'pi':
                            for _ in range(K):
                                temp_params_list_mix.append(random.choice(available_it_for_pi))
                        elif join_table_alias == 'mi':
                            for _ in range(K):
                                temp_params_list_mix.append(random.choice(available_it_for_mi))
                        elif join_table_alias == 'miidx':
                            for _ in range(K):
                                temp_params_list_mix.append(random.choice(available_it_for_miidx))
                    else:
                        # Step 1: get param and their cardinality
                        ranking_result = get_param_and_cardinality(db_name, single_select_table, single_select_column, select_column, from_table, not_null_column, join_condition, value_type)

                        # range CDF - currently only accept int
                        if select_operator in [">", ">=", "<", "<=", "="] and select_data_type == "int":
                            ranking_result = get_range_param_and_cardinality(ranking_result, select_operator)
                        
                        # Step 2: Create bucket                        
                        buckets = create_buckets(ranking_result, N)
                        
                        # Step 3: Sample from bucket
                        # If IN operator, sample K // bucket_num for each bucket, each with 5 params
                        if select_operator == "IN":
                            sampled_data = sample_in_operator(buckets, K)
                            sampled_data = format_in_clause_data(sampled_data)
                            print("get sampled", len(sampled_data))
                            temp_params_list = sampled_data
                        else:
                            sampled_data = sample_from_buckets(buckets, K)
                            temp_params_list = [i[0] for i in sampled_data]
                            random.shuffle(temp_params_list)
                            temp_params_list = [escape_single_quotes(param) for param in temp_params_list]
                        
                        # Debug output - show ranges of each bucket and the first 10 item
                        for i, bucket in enumerate(buckets):
                            if debug: 
                                if len(bucket) > 10:
                                    print(f"Bucket {i + 1}: {bucket[-10:]}")
                                    second_values = [value for _, value in bucket]
                                    print(f"  === Cardinality range of bucket {i+1}: [{min(second_values)}, {max(second_values)}]")
                                else:
                                    if bucket:
                                        second_values = [value for _, value in bucket]
                                        print(f"  === Cardinality range of bucket {i+1}: [{min(second_values)}, {max(second_values)}]")

                        # Mix param lists sampled by each join (pair of table)
                        temp_params_list_mix = temp_params_list_mix + temp_params_list
                        
                random.shuffle(temp_params_list_mix)
                all_param_lists.append(temp_params_list_mix[:K])

        return all_param_lists



def gen_param_by_cardinality_full(db_name, template_file, N=10, K=50, debug=True):
    """
    Generates parameter values for database query templates based on column cardinality.
    This time, we have all tables joined together and then calculate the cardinality.
    Instead of (single value, cardinality), we now have (row value, cardinality)
    
    This function processes query templates and creates parameter lists by analyzing 
    column value distributions in the database. It divides values into buckets based on 
    cardinality and samples from these buckets to ensure a representative parameter set.
    The function handles various operators (=, LIKE, IN, etc.) and joins differently.
    
    Args:
        db_name (str): Database name to connect to.
        template_file (str): Path to the JSON file containing query templates.
        N (int, optional): Number of buckets to split the cardinality range. Defaults to 10.
        K (int, optional): Total number of samples to generate per parameter. Defaults to 50.
        debug (bool, optional): Whether to print debug information. Defaults to True.
        
    Returns:
        list: A list of parameter lists, where each parameter list corresponds to a predicate
                in the query templates. Format: [[@param1_1, @param1_2, ..., @param1_K], 
                [@param2_1, ..., @param2_K], ...].
    """
    with open(template_file) as f:
        all_param_lists = []
        templates = json.load(f)
        for query_id, template in templates.items():
            # Find JOIN conditions
            def extract_where_conditions(query, param):
                """
                Extracts WHERE clause up to the last AND before @param
                
                Args:
                    query (str): The original SQL query
                    param (str): The parameter to find (e.g., '@param0')
                
                Returns:
                    str: WHERE clause up to the specified point
                """
                # Find WHERE position
                where_pos = query.upper().find('WHERE ')
                if where_pos == -1:
                    raise ValueError("Could not find WHERE in query")
                    
                # Find parameter position
                param_pos = query.find(param)
                if param_pos == -1:
                    raise ValueError(f"Could not find {param} in query")
                
                # Get substring from WHERE to param
                query_section = query[where_pos:param_pos]
                
                # Find last AND before param
                last_and_pos = query_section.rstrip().upper().rfind('AND ')
                if last_and_pos == -1:
                    raise ValueError(f"Could not find AND before {param}")
                
                # Return WHERE clause up to last AND
                return query_section[:last_and_pos].strip()

            original_query = template['query']
            # print(original_query)
            join_condition = extract_where_conditions(original_query, '@param0')
            # print('join_condition', join_condition, '\n')
            
            # Build base predicate columns list
            base_predicate_column_elements = []
            
            for predicate in template['predicates']:
                alias = predicate.get('original_alias', predicate['alias'])
                base_predicate_column_elements.append(f"{alias}.{predicate['column']}")
                
            # Combine columns for SELECT and NOT NULL conditions
            base_predicate_column = ', '.join(base_predicate_column_elements)
            not_null_conditions = ' AND '.join(f"{element} IS NOT NULL" for element in base_predicate_column_elements)
            full_where_conditions = join_condition + ' AND ' + not_null_conditions
            
            # get from tables, add tablesample statement
            def extract_and_modify_from_clause(query, tablesample_percentage, seed):
                """
                Extract the FROM clause from a SQL query and modify it to include TABLESAMPLE
                
                Args:
                    query (str): Original SQL query
                    tablesample_percentage (float): Percentage for TABLESAMPLE
                    seed (int): Seed for REPEATABLE
                    
                Returns:
                    str: Modified FROM clause
                """
                # Extract content between FROM and WHERE
                from_start = query.upper().find('FROM ')
                where_start = query.upper().find('WHERE ')
                if from_start == -1 or where_start == -1:
                    raise ValueError("Could not find FROM or WHERE in query")
                    
                # Extract the original FROM clause
                original_from = query[from_start:where_start].strip()
                
                # Remove the initial "FROM " keyword
                table_list = original_from[5:].strip()
                
                # Split into individual table references
                table_refs = [ref.strip() for ref in table_list.split(',')]
                
                # Process each table reference
                new_from_clauses = []
                for ref in table_refs:
                    # Parse table name and alias
                    parts = ref.strip().split(' AS ')
                    if len(parts) != 2:
                        raise ValueError(f"Invalid table reference format: {ref}")
                        
                    table_name = parts[0].strip()
                    alias = parts[1].strip()
                    
                    # Create new FROM clause with TABLESAMPLE
                    new_clause = f"""(
                        SELECT * FROM {table_name} 
                        TABLESAMPLE SYSTEM({tablesample_percentage}) 
                        REPEATABLE({seed})
                    ) AS {alias}"""
                    
                    new_from_clauses.append(new_clause)
                
                # Combine all clauses
                return "FROM " + ",\n    ".join(new_from_clauses)
            
            # change table sample percentage and seed, originally (10, 2024)
            tablesample_percentage = 10
            seed = 2024
            
            while tablesample_percentage <= 100:
                # Try up to 10 different seeds for each percentage
                for _ in range(10):
                    # Build FROM clause with current parameters
                    from_clause = extract_and_modify_from_clause(original_query, tablesample_percentage, seed)
                    
                    # Try to get results
                    ranking_result = get_param_and_cardinality_full(
                        db_name, 
                        base_predicate_column, 
                        from_clause, 
                        full_where_conditions
                    )
                    
                    # If we got enough results, break
                    if len(ranking_result) >= 10000:
                        break
                        
                    # Try next seed
                    seed += 1
                
                if len(ranking_result) >= 10000:
                    break
                    
                # If we didn't find enough results after 10 seeds,
                # increase the sampling percentage
                tablesample_percentage += 10
                seed = 2024  # Reset seed for new percentage
            
            # ranking result format: (tuple(processed_values), row[-1])
            # change to list of [(processed_values[i], row[-1]) for i in range(len(processed_values))]
            new_ranking_result = [[] for _ in range(len(ranking_result[0][0]))]
            for processed_values, row_last in ranking_result:
                for i, value in enumerate(processed_values):
                    new_ranking_result[i].append((value, row_last))
            print([sublist[0] for sublist in new_ranking_result])
            
            # Step 1: enumerate through new_ranking_result [[(value0, card0)], [(value1, card1)], etc]
            for idx, param_ranking in enumerate(new_ranking_result):
                print(f"\n{idx} - predicate {param_ranking[:5]}")
                temp_params_list_mix = [] # final param list for this specific predicate (same as for predicate in predicates)
                select_operator = template['predicates'][idx]["operator"]
                select_data_type = template['predicates'][idx]["data_type"]
            
                # range CDF - currently only accept int
                if select_operator in [">", ">=", "<", "<=", "="] and select_data_type == "int":
                    print(f"Before range-based: {param_ranking[:5]}")
                    with open('pg_test.json', 'w') as f:
                        param_ranking = sorted(param_ranking)
                        json.dump(param_ranking, f)
                    param_ranking = get_range_param_and_cardinality(param_ranking, select_operator)
                    print(f"After range-based: {param_ranking[:5]}")
                    with open('pg_test_output.json', 'w') as f:
                        param_ranking = sorted(param_ranking)
                        json.dump(param_ranking, f)
                # LIKE operator - split sentenct -> word, word -> character; random choose one, add wildcard
                elif select_operator.upper() == "LIKE":
                    print("LIKE operator original", param_ranking[:5])
                    new_param_ranking = []
                    
                    for value, card in param_ranking:
                        value = str(value)
                        
                        # split by space
                        tokens = value.split()
                        if len(tokens) == 1:
                            # random choose one character
                            selected = random.choice(tokens[0])
                        else:
                            # random choose one word
                            selected = random.choice(tokens)
                            
                        # add wildcard
                        wildcard_value = f"%{selected}%"
                        new_param_ranking.append((wildcard_value, card))
                    
                    param_ranking = new_param_ranking
                    print("LIKE operator modified", param_ranking[:5])
                
                # Step 2: Create bucket                        
                buckets = create_buckets(param_ranking, N)
                
                # Step 3: Sample from bucket
                # If IN operator, sample K // bucket_num for each bucket, each with 5 params
                # else sample from bucket
                if select_operator == "IN":
                    sampled_data = sample_in_operator(buckets, K, shuffle=False) # NO shuffle here!!
                    sampled_data = format_in_clause_data(sampled_data)
                    print("IN sampled", len(sampled_data))
                    temp_params_list = sampled_data
                else:
                    sampled_data = sample_from_buckets(buckets, K)
                    temp_params_list = [i[0] for i in sampled_data]
                    # NO shuffle here!! large-large, small-small, shuffle after one-on-one
                    temp_params_list = [escape_single_quotes(param) for param in temp_params_list]
                
                # Debug output - show ranges of each bucket and the first 10 item
                for i, bucket in enumerate(buckets):
                    if debug: 
                        if len(bucket) > 10:
                            print(f"Bucket {i + 1}: {bucket[-10:]}")
                            second_values = [value for _, value in bucket]
                            print(f"  === Cardinality range of bucket {i+1}: [{min(second_values)}, {max(second_values)}]")
                        else:
                            if bucket:
                                second_values = [value for _, value in bucket]
                                print(f"  === Cardinality range of bucket {i+1}: [{min(second_values)}, {max(second_values)}]")

                # One result for each predicate
                temp_params_list_mix = temp_params_list 
                all_param_lists.append(temp_params_list_mix[:K])

        return all_param_lists



###################
# Kepler related
def run_kepler_pipeline(db_name, template_file, count):
    """
    Executes the Kepler parameter generation pipeline as a subprocess.
    
    This function creates the necessary output directory structure and runs the Kepler
    training data collection pipeline to generate query parameters. It handles the 
    configuration of the pipeline, including database connection details, input/output
    paths, and the number of parameters to generate.
    
    Args:
        db_name (str): Database name to connect to.
        template_file (str): Path to the template file containing query structures.
        count (int): Number of parameter sets to generate.
        
    Returns:
        str: Path to the output directory containing generated parameters if successful.
                None is implied if the pipeline execution fails.
    """
    output_dir = os.path.join(f"imdb_{_QUERY_ID.value}_original", "kepler", "inputs") # MARK: manually change to your desired folder name
    counts_output_file = os.path.join(output_dir, "output_counts.json")
    parameters_output_dir = output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    # run the kepler command
    command = [
        "python3", "-m", "kepler.training_data_collection_pipeline.parameter_generator_main",
        "--database", db_name,
        "--user", "kepler",
        "--password", "kepler",
        "--host=localhost",
        "--template_file", template_file,
        "--counts_output_file", counts_output_file,
        "--parameters_output_dir", parameters_output_dir,
        "--count", str(count)
    ]

    try:
        subprocess.run(command, check=True)
        print("Kepler pipeline executed successfully.")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error executing pipeline: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")



###################
# CSV related
# Step 0: find specific matching element (table.column operator with params) within the predicate_template_metadata
def get_matching_elements(metadata, template_predicates):
    """
    Matches template predicates with their corresponding metadata elements.
    
    This function searches for metadata entries that match each predicate's 
    specifications (alias, column, operator, and data type). It handles special 
    cases for the "it" alias by looking for specific metadata keys or constructing 
    composite keys based on join table aliases.
    
    Args:
        metadata (dict): Dictionary containing metadata organized by table aliases.
        template_predicates (list): List of predicate dictionaries from query templates.
        
    Returns:
        list: A list of matching metadata elements for each predicate. Each element
                contains details about the column, operator, and data type. None is 
                included for predicates with no matching metadata.
    """
    matching_elements = []
    
    for predicate in template_predicates:
        alias = predicate["alias"]
        column = predicate["column"]
        operator = predicate["operator"].lower()
        data_type = predicate["data_type"]
        
        # Check if alias is "it"
        if alias == "it":
            if "it" in metadata:
                # Use metadata["it"] if available
                current_metadata = metadata["it"]
            else:
                # Otherwise, use join_tables_alias if available
                join_tables_alias = predicate.get("join_tables_alias")
                if join_tables_alias:
                    join_table = join_tables_alias[0].replace("_", "")
                    alias_with_join = f"{alias}_{join_table}" # extra table list for it (e.g. it_pi, it_miidx, etc)
                    current_metadata = metadata.get(alias_with_join, [])
                else:
                    current_metadata = []
        else:
            # For other aliases, use the original alias
            current_metadata = metadata.get(alias, [])
        
        # print(f"current_metadata is {current_metadata}")
        matching_element = next(
            (item for item in current_metadata if item["column"] == column and 
                                             item["operator"].lower() == operator and
                                             item["data_type"] == data_type),
            None
        )
        matching_elements.append(matching_element)
    
    return matching_elements



# Step 1: generate param list by frequency
def gen_sql_by_template(params_list, frequencies_list, K):
    """
    Generates SQL parameter combinations by sampling based on frequency distributions.
    
    This function samples parameter values from multiple parameter lists according to 
    their frequency distributions. It creates combinations of parameters across different
    lists, ensuring no duplicate values within each combination. The sampling is weighted
    by the frequency of each parameter value's occurrence.
    
    Args:
        params_list (list): List of parameter lists, each containing possible values.
        frequencies_list (list): List of frequency lists corresponding to each parameter list.
        K (int): Number of parameter combinations to generate.
        
    Returns:
        list: A list of parameter combinations, where each combination contains one value
                from each parameter list. Duplicates within combinations are filtered out.
    """
    sampled_literals = []

    # sample the param based on the frequency
    for params, frequencies in zip(params_list, frequencies_list):
        total_frequency = sum(frequencies)
        probabilities = [freq / total_frequency for freq in frequencies]
        sampled_literals.append(np.random.choice(params, p=probabilities, size=K*2))

    sampled_literals = np.transpose(sampled_literals).tolist()
    sampled_literals = [i for i in sampled_literals if len(set(i)) == len(i)][:K] 


    return sampled_literals 



# Step 2: save param & frequency
def get_literal_frequencies(literals):
    """
    calculate param with frequency
    """
    frequency_dict = collections.defaultdict(int)
    
    for literal in literals:
        frequency_dict[json.dumps(literal)] += 1
    
    return frequency_dict



def split_literals_and_store_with_frequency(query_id, sampled_literals, seed_value, output_dir, train_size_list=[50, 400], test_size=200):
    """
    Splits sampled literals into training and testing sets and stores them with frequency information.
    
    This function divides the sampled literals into training and testing datasets, calculates
    the frequency of each literal combination, and stores the results in JSON files. It supports
    creating multiple training datasets of different sizes for incremental training experiments.
    
    Args:
        query_id (str): Identifier for the query being processed.
        sampled_literals (list): List of literal combinations to split and process.
        seed_value (int): Random seed value for reproducibility.
        output_dir (str): Directory to store the output files.
        train_size_list (list, optional): List of training set sizes to generate. 
                                            Defaults to [50, 400].
        test_size (int, optional): Size of the test set. Defaults to 200.
        
    Returns:
        tuple: A tuple containing:
                - train_dict_dict (dict): Dictionary mapping training set sizes to dictionaries
                    of literals with their frequencies.
                - test_dict (dict): Dictionary of test literals with their frequencies.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # split train & test
    sampled_literals = list(sampled_literals)
    random.shuffle(sampled_literals) 
    test_literals = sampled_literals[:test_size]
    train_literals = sampled_literals[test_size:]
    
    # get distinct train & test literals
    distinct_train_literals = list(set(map(tuple, train_literals)))
    distinct_train_literals = [list(item) for item in distinct_train_literals]
    
    distinct_test_literals = list(set(map(tuple, test_literals)))
    distinct_test_literals = [list(item) for item in distinct_test_literals]
    
    # calculate frequency
    train_literal_frequencies = get_literal_frequencies(train_literals)
    test_literal_frequencies = get_literal_frequencies(test_literals)
    
    # Store all train_dicts in a dictionary, keyed by train_size
    train_dict_dict = {}
    full_train_dict = {json.dumps(literal): train_literal_frequencies[json.dumps(literal)] for literal in distinct_train_literals}
    train_dict_dict[len(train_literals)] = full_train_dict
    test_dict = {json.dumps(literal): test_literal_frequencies[json.dumps(literal)] for literal in distinct_test_literals}
    
    for train_size in train_size_list:
        # Get the subset of train_literals based on the current train_size
        train_subset = train_literals[:train_size]
        new_train_literal_freq = get_literal_frequencies(train_subset)

        # Get distinct train literals for this subset
        distinct_train_subset = list(set(map(tuple, train_subset)))
        distinct_train_subset = [list(item) for item in distinct_train_subset]
        
        # Create a frequency dictionary for this subset
        train_dict = {json.dumps(literal): new_train_literal_freq[json.dumps(literal)] for literal in distinct_train_subset}
        
        # Add the current train size dictionary to the list of dictionaries
        train_dict_dict[train_size] = train_dict
    
    
    # Ensure the directory exists
    base_dir = "frequency"
    output_dir_path = os.path.join(output_dir, base_dir)
    os.makedirs(output_dir_path, exist_ok=True)

    # Save each train literal dict with frequency based on train_size_list, including full size
    for train_size, train_dict in train_dict_dict.items():
        train_output_file = os.path.join(output_dir_path, f"{query_id}_train_{train_size}_freq.json")
        with open(train_output_file, 'w') as train_file:
            json.dump(train_dict, train_file, indent=4)

    # Save test literals with frequency
    test_output_file = os.path.join(output_dir_path, f"{query_id}_test_freq.json")
    with open(test_output_file, 'w') as test_file:
        json.dump(test_dict, test_file, indent=4)
    
    
    return train_dict_dict, test_dict



###################
# PQO input related
def save_pqo_files(query_id, data, output_dir, description): # description = {train_size}_training / testing
    """
    Saves parameterized query files for Parametric Query Optimization (PQO) testing.
    
    This function takes a query template and parameter combinations, substitutes 
    the parameters into the template, and saves the resulting queries to a JSON file.
    Each query is assigned a unique key based on the query ID, description, and index.
    The function avoids overwriting existing files with the same name.
    
    Args:
        query_id (str): Identifier for the query template.
        data (dict): Dictionary containing query templates and parameter combinations.
        output_dir (str): Directory to save the output JSON file.
        description (str): Description string (typically includes training size or "testing").
        
    Returns:
        None: The function doesn't return a value but saves the generated queries to a file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_json = {}

    new_sql_template = data[query_id]["query"]
    literals = data[query_id]["params"]

    for index, combination in enumerate(literals):
        query_str = new_sql_template
        for i, param in enumerate(combination):
            param = str(param).strip()
            pattern = re.compile(rf"@param{i}\b")
            query_str = pattern.sub(param, query_str)
        
        key = f"{query_id}_{description}_{index}"
        output_json[key] = query_str

    # PQO query file naming
    file_name = f'{query_id}_{description}.json'
    output_file = os.path.join(output_dir, file_name)
    if os.path.exists(output_file):
        print(f"{file_name} already exists, skipping save.")
        return
    
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(output_json, json_file, indent=4)



def save_pqo_predicates(query_id, training_data, testing_data, output_dir, train_size):
    """
    Saves query predicates in a formatted text file for Parametric Query Optimization (PQO).
    
    This function extracts predicate combinations from both training and testing data
    and saves them in a formatted text file. Each predicate is formatted according to
    its data type and operator, with proper quoting for text values and parentheses for
    IN operators. The function creates separate files for training and testing data.
    
    Args:
        query_id (str): Identifier for the query.
        training_data (dict): Dictionary containing training query parameters and predicates.
        testing_data (dict): Dictionary containing testing query parameters and predicates.
        output_dir (str): Directory to save the output text files.
        train_size (int): Size of the training dataset, used in output filename.
        
    Returns:
        None: The function doesn't return a value but saves the formatted predicates to files.
    """
    os.makedirs(output_dir, exist_ok=True)

    def save_combinations(data, file_name):
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            print(f"{file_name} already exists, skipping save.")
            return
        
        with open(output_file, 'w', encoding='utf-8') as file:
            for i, combination in enumerate(data["params"]):
                formatted_items = []
                for j, item in enumerate(combination):
                    predicate = data["predicates"][j]
                    table = predicate["alias"]
                    column = predicate["column"]
                    operator = predicate["operator"]
                    data_type = predicate["data_type"]
                    
                    # Format the param based on its data type
                    if data_type == "text":
                        formatted_param = f"'{item.strip()}'"
                    else:
                        formatted_param = item
                    
                    # Format the param if the operator is "in"
                    if operator.lower() == "in":
                        formatted_param = f"({formatted_param})"
                    
                    # Create the formatted string as table.column operator param
                    formatted_items.append(f"{table}.{column} {operator} {formatted_param}" if table else f"{column} {operator} {formatted_param}")
                
                # Create the combination string
                combination_str = '["' + '", "'.join(formatted_items) + '"]'
                
                # Write to the file
                if i < len(data["params"]) - 1:
                    file.write(combination_str + ',\n')
                else:
                    file.write(combination_str + '\n')


    # Save training combinations
    save_combinations(training_data[query_id], f'{query_id}_{train_size}_training.txt')

    # Save testing combinations
    save_combinations(testing_data[query_id], f'{query_id}_testing.txt')



# Save outputs
def prepare_directories(output_dir):
    """
    Prepares the directory structure for storing the templates and PQO files.

    Args:
    output_dir (str): The base directory for saving files.

    Returns:
    dict: A dictionary with the paths to various directories.
    """
    dirs = {
        'training': os.path.join(output_dir, 'training'),
        'testing': os.path.join(output_dir, 'testing'),
        'PQO_query': os.path.join(output_dir, 'PQO', 'query'),
        'PQO_predicates': os.path.join(output_dir, 'PQO', 'predicates'),
        'metadata': os.path.join(output_dir, 'metadata')
    }

    # Create the directories if they don't exist
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs
   


def load_template(template_file, query_id):
    """
    Load the query and predicates from a JSON template file.
    
    Args:
    template_file (str): Path to the JSON file containing the query templates.
    query_id (str): query id related to the template
    
    Returns:
    tuple: query_id, query, and predicates from the first template.
    """
    # Load the JSON template file
    with open(template_file, 'r') as file:
        templates = json.load(file)
    
    # Extract the corresponding query template
    template = templates[query_id]
    query = template['query']
    predicates = template['predicates']
    
    # Check if 'params' exists in the template, if not set it to None
    original_param_list = template.get('params', None)
    
    return query, predicates, original_param_list



def save_templates_and_pqo(query_id, query, predicates, training_params, testing_params, dirs, train_size, mode="original"):
    """
    Save the full, training, and testing templates, and corresponding PQO files.

    Args:
    query_id (str): The ID of the query.
    query (str): The query string.
    predicates (list): List of predicates.
    training_params (list): List of training parameters.
    testing_params (list): List of testing parameters.
    dirs (dict): A dictionary containing the paths to the directories for saving the files.
    """
    # Create training, and testing template dictionaries
    training_template = {
        query_id: {
            "query": query,
            "predicates": predicates,
            "params": training_params
        }
    }

    testing_template = {
        query_id: {
            "query": query,
            "predicates": predicates,
            "params": testing_params
        }
    }

    # Save training template
    training_output_file = os.path.join(dirs['training'], f"{query_id}_training_{mode}_{train_size}.json")
    with open(training_output_file, 'w') as out_file:
        json.dump(training_template, out_file, indent=4)

    # Save testing template
    testing_output_file = os.path.join(dirs['testing'], f"{query_id}_testing_{mode}.json")
    if os.path.exists(testing_output_file):
        print(f"{query_id}_testing_{mode}.json already exists, skipping save.")
    else:
        with open(testing_output_file, 'w') as out_file:
            json.dump(testing_template, out_file, indent=4)

    if mode == "original":
        # PQO directories for predicates and queries
        pqo_query_dir = os.path.join(dirs['PQO_query'])
        pqo_predicates_dir = os.path.join(dirs['PQO_predicates'])
        
        # Save PQO files
        save_pqo_files(query_id, training_template, pqo_query_dir, f'training_{train_size}')
        save_pqo_files(query_id, testing_template, pqo_query_dir, f'testing')
        save_pqo_predicates(query_id, training_template, testing_template, pqo_predicates_dir, train_size)



# generate training & testing set
def generate_param_list_from_dict(data_dict, mode):
    """
    Generates a parameter list from a dictionary of parameter combinations with frequencies.
    
    This function converts a dictionary of parameter combinations (stored as JSON strings) 
    into a list based on the specified mode. In 'distinct' mode, each unique combination 
    appears exactly once. In 'original' mode, combinations are replicated according to 
    their frequency count, preserving the original distribution.
    
    Args:
        data_dict (dict): Dictionary with parameter combinations as keys (JSON strings) 
                            and frequencies as values.
        mode (str): Generation mode - either 'distinct' (unique combinations) or 
                    'original' (frequency-weighted combinations).
        
    Returns:
        list: A shuffled list of parameter combinations based on the specified mode.
    """
    param_list = []
    if mode == 'distinct':
        # In 'distinct' mode, use keys (unique literals) only once
        param_list = [eval(key) for key in data_dict.keys()]
    elif mode == 'original':
        # In 'original' mode, replicate keys by their frequencies
        for key, freq in data_dict.items():
            param_list.extend([eval(key)] * freq)
    random.shuffle(param_list)
    return param_list



# filter the param list
def filter_invalid_combinations(param_list, predicates):
    """
    Filters out invalid parameter combinations where the same column in the same table 
    has conflicting parameter values, ensuring proper relationships between
    <, >, <=, and >= operators, and that all parameter values for the same column
    (even with different operators like LIKE and NOT LIKE) are distinct.
    
    Args:
    - param_list: List of parameter combinations.
    - predicates: List of predicates containing column, table, and operator information.
    
    Returns:
    - A filtered list of valid parameter combinations.
    """
    filtered_params = []
    
    # Group operators by type
    like_group = ['LIKE', 'NOT LIKE']
    equals_group = ['=', '!=']
    comparison_group = ['<', '>', '<=', '>=']

    for params in param_list:
        valid = True
        table_column_map = {}  # Store the values for each table+column combination by group

        # Iterate through predicates and associated params
        for i, predicate in enumerate(predicates):
            table = predicate['table']
            column = predicate['column']
            operator = predicate['operator']
            value = params[i]
            
            table_column_key = f"{table}.{column}"  # Combine table and column for uniqueness
            
            # Initialize the map if this column hasn't been encountered before
            if table_column_key not in table_column_map:
                table_column_map[table_column_key] = {'like_group': [], 'equals_group': [], 'comparison_group': []}
            
            # Place the value into the correct group based on the operator
            if operator in like_group:
                group = 'like_group'
            elif operator in equals_group:
                group = 'equals_group'
            elif operator in comparison_group:
                group = 'comparison_group'
            else:
                continue  # Skip unknown operators for now

            # Ensure all values in the group are distinct
            if value in table_column_map[table_column_key][group]:
                valid = False
                break
            else:
                # Store (operator, value) in comparison group if it's comparison-related
                if group == 'comparison_group':
                    table_column_map[table_column_key][group].append((operator, value))
                else:
                    table_column_map[table_column_key][group].append(value)

        # After collecting all params for this row, check the comparison conditions
        if valid:
            comparison_map = table_column_map[table_column_key]['comparison_group']
            
            # Iterate through all comparison operators to check their relationships
            for op1, val1 in comparison_map:
                for op2, val2 in comparison_map:
                    # Check conflicting comparisons between <, >, <=, >=
                    if op1 == '>' and op2 == '<' and val1 >= val2:
                        valid = False
                    if op1 == '>=' and op2 == '<=' and val1 > val2:
                        valid = False
                    if op1 == '>' and op2 == '<=' and val1 >= val2:
                        valid = False
                    if op1 == '>=' and op2 == '<' and val1 > val2:
                        valid = False

        # Add valid parameter combinations to the filtered list
        if valid:
            filtered_params.append(params)
    
    return filtered_params



def execute_query(cursor, query):
    try:
        # make sure it return result
        def modify_query(query: str) -> str:
            """
            Modify the query by splitting on FROM and replacing the SELECT clause with SELECT *
            """
            query_lower = query.lower()
            from_index = query_lower.find('from')
            
            if from_index == -1:
                return query
                
            # Use the original query's case for the remaining part
            modified_query = 'SELECT * ' + query[from_index:]
            
            return modified_query
        
        query = query.strip()
        query = modify_query(query)
        if query.endswith(";"):
            query = query[:-1]
        query += " LIMIT 5;"

        cursor.execute(query)
        rows = cursor.fetchall()
        return rows
    except psycopg2.OperationalError as e:
        logging.error(f"OperationalError: {e}")
        return []
    
    except psycopg2.DatabaseError as e:
        logging.error(f"DatabaseError: {e}")
        return []
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return []
    


def test_query_returns(query, param_list, db_config, query_id, method="cardinality", count=251):
    """
    Tests a SQL query with different parameter combinations and collects those that return results.
    
    This function connects to a database using the provided configuration, then iterates through
    a list of parameter sets, substituting them into the query template. For each parameter set,
    it executes the query and checks if it returns any rows. Parameter sets that produce results
    are collected and counted. The function stops after finding a specified number of successful
    parameter sets (default 251) or after testing all provided parameter sets.
    
    The function tracks progress by printing parameter information during execution and saves
    statistical data about the test run to a CSV file in the '0_original_analysis' directory.
    
    Args:
        query (str): The SQL query template with placeholders for parameters (@param0, @param1, etc.)
        param_list (list): A list of parameter sets, where each set is a list of values to substitute
        db_config (dict): Database connection configuration with keys: dbname, user, password, host, port
        query_id (str): Identifier for the query, used in the output CSV file
        method (str, optional): Method identifier, used in the output CSV file. Defaults to "cardinality".
        count (int, optional): Maximum number of successful parameter sets to collect. Defaults to 251.
    
    Returns:
        list: A list of parameter sets that successfully returned results when used with the query
    """
    # connect to db
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )

    cursor = conn.cursor()
    
    # initialize
    counter = 0
    clean_param_list = []
    
    param_counter = 0

    # test query if it returns value
    for params in param_list:
        # get full query instance
        print(f'{param_counter}. {counter} - {params}')
        test_query = query
        
        for i, param in enumerate(params):
            param = str(param).strip()
            # Attempt to create and use the regex pattern
            try:
                pattern = re.compile(rf"@param{i}\b")  # Compile regex
                test_query = pattern.sub(param, test_query)  # Perform substitution
            except re.error as e:
                print(f"Skipping param {param} due to regex error: {e}")
                continue  # Skip this param and move to the next one
        
        rows = execute_query(cursor, test_query)
        cursor.execute('COMMIT')  # commit after each execution to clear any locks
        
        if len(rows) > 0:
            counter += 1
            clean_param_list.append(params)
        
        if counter % 100 == 0 and counter != 0:
            print(counter)
        # if counter != 0:
        #     print(counter)
            
        if counter == count:
            break
        
        param_counter += 1

    # close db
    cursor.close()
    conn.close()
    
    if not os.path.exists('0_original_analysis'):
        os.makedirs('0_original_analysis')
    
    # Prepare the file path
    file_path = f'0_original_analysis/original_workload_count_{count}.csv'
    
    # Prepare the new row data
    new_row = [query_id, method, param_counter]
    
    # Check if file exists
    if os.path.exists(file_path):
        # Append to existing file
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)
    else:
        # Create new file with header and first row
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['query_id', 'method', 'param_counter'])  # Write header
            writer.writerow(new_row)  # Write data
    
    return clean_param_list



# generation based on selection
def process_train_and_test_data(output_dir, query_id, query, predicates, train_dict_dict, test_dict):
    """
    Process training and testing data for query optimization by generating parameter lists,
    saving templates, and creating Parametric Query Optimization (PQO) artifacts.
    
    This function iterates through different modes ("original" and "distinct") and various 
    training set sizes to generate parameter lists for both training and testing datasets.
    For each combination of mode and training size, it calls helper functions to generate
    the parameter lists, saves the query templates with these parameters, and creates PQO 
    artifacts. The function also tracks metadata about the generated datasets, including 
    the number of training and testing parameters for each configuration, and saves this 
    metadata to a CSV file.
    
    Args:
        output_dir (dict): Dictionary containing output directory paths for different artifacts
        query_id (str): Identifier for the query being processed
        query (str): The SQL query template with placeholders for parameters
        predicates (list): List of predicate definitions used in the query
        train_dict_dict (dict): Nested dictionary mapping training sizes to parameter dictionaries
        test_dict (dict): Dictionary containing parameter values for testing
        
    Returns:
        None: Results are saved to files in the specified output directories
    """
    # Initialize a metadata dictionary to store relevant information
    metadata = {}

    # Generate params for training and testing
    for mode in ["original", "distinct"]:
        for train_size in sorted(train_dict_dict.keys()):  # train_size_list + full size
            training_params = generate_param_list_from_dict(train_dict_dict[train_size], mode)
            testing_params = generate_param_list_from_dict(test_dict, mode)
            print(f"{mode} (train size: {train_size}): training {len(training_params)}, testing {len(testing_params)}")

            # Save metadata for each mode
            metadata[f"{mode}_testing_params"] = len(testing_params)
            metadata[f"{mode}_{train_size}_training_params"] = len(training_params)

            # Save templates and PQO for each train size
            save_templates_and_pqo(query_id, query, predicates, training_params, testing_params, output_dir, train_size, mode)

    # Save metadata
    save_metadata_to_csv(output_dir['metadata'], metadata, query_id)
    


###################
# generation based on selection
def gen_full_cardinality(db_config, output_dir, template_file, query_id, param_choice_list, seed_value, count, train_size_list=[50, 400], test_size=200, method="cardinality", shuffle=False):
    """
    Generates and processes cardinality data for query optimization by testing parameter combinations,
    filtering those that return results, and preparing training and testing datasets.
    
    This function takes a SQL query template and a list of potential parameter values, tests various
    parameter combinations to find those that return results from the database, and then splits these
    successful combinations into training and testing sets. It handles the complete workflow from
    loading templates, testing parameter combinations, filtering valid ones, shuffling if needed,
    and splitting data into appropriate training and testing sets with different sizes. The resulting
    data is processed and saved for use in query optimization tasks.
    
    Args:
        db_config (dict): Database connection configuration
        output_dir (str): Base directory for output files
        template_file (str): Path to the SQL query template file
        query_id (str): Identifier for the query being processed
        param_choice_list (list): List of lists, where each inner list contains possible values for a parameter
        seed_value (int): Random seed for reproducibility
        count (int): Maximum number of successful parameter combinations to collect
        train_size_list (list, optional): List of different training set sizes. Defaults to [50, 400].
        test_size (int, optional): Size of the testing set. Defaults to 200.
        method (str, optional): Method identifier for tracking purposes. Defaults to "cardinality".
        shuffle (bool, optional): Whether to shuffle parameter combinations before testing. Defaults to False.
        
    Returns:
        None: Results are saved to files in the specified output directories
    """
    # preparation
    dirs = prepare_directories(output_dir)
    query, predicates, _ = load_template(template_file, query_id)
    
    # Ensure param_choice_list contains lists of parameters
    if not all(isinstance(param, list) for param in param_choice_list):
        raise ValueError("param_choice_list should contain lists of parameters.")
    
    # One-on-One
    param_list = list(zip(*param_choice_list))
    param_list = [list(params) for params in param_list] # Convert tuples to lists
    param_list = filter_invalid_combinations(param_list, predicates)
    if shuffle:
        random.shuffle(param_list)
    print('filter_out card', len(param_list))
    
    # execute
    param_list = test_query_returns(query, param_list, db_config, query_id, method=method)
    random.shuffle(param_list)
    
    # Process train and testing files
    train_dict_dict, test_dict = split_literals_and_store_with_frequency(query_id, param_list, seed_value, output_dir, train_size_list, test_size)
    process_train_and_test_data(dirs, query_id, query, predicates, train_dict_dict, test_dict)



def gen_full_csv(output_dir, template_file, query_id, train_dict_dict, test_dict):
    """
    Generates full template format for query optimization by processing existing parameter dictionaries 
    
    It loads a query template, prepares the necessary directories, and then processes the 
    provided training and testing parameter dictionaries to generate template files and 
    query optimization artifacts. 
    
    Args:
        output_dir (str): Base directory for output files
        template_file (str): Path to the SQL query template file
        query_id (str): Identifier for the query being processed
        train_dict_dict (dict): Nested dictionary mapping training sizes to parameter dictionaries
        test_dict (dict): Dictionary containing parameter values for testing
        
    Returns:
        None: Results are saved to files in the specified output directories
    """
    # preparation
    dirs = prepare_directories(output_dir)
    query, predicates, _ = load_template(template_file, query_id)

    # Generate params for training and testing
    process_train_and_test_data(dirs, query_id, query, predicates, train_dict_dict, test_dict)



def gen_full_kepler(output_dir, original_file, query_id, seed_value, db_config, train_size_list=[50, 400], test_size=200):
    """
    Generates and processes data for Kepler-based query optimization by loading parameters from templates,
    adjusting numeric values, handling special operators, and preparing training and testing datasets.
    
    This function is specialized for Kepler's query optimization requirements. It loads query templates 
    and parameter lists, performs necessary adjustments to numeric ranges, handles 'IN' operators 
    differently from other operators, escapes single quotes in parameter values where appropriate, 
    tests parameter combinations against the database, and splits successful combinations into 
    training and testing sets. Special care is taken to adjust numeric predicate boundaries and to 
    handle parameters used with 'IN' operators differently from other parameters.
    
    Args:
        output_dir (str): Base directory for output files
        original_file (str): Path to the original template file containing query and parameters
        query_id (str): Identifier for the query being processed
        seed_value (int): Random seed for reproducibility
        db_config (dict): Database connection configuration
        train_size_list (list, optional): List of different training set sizes. Defaults to [50, 400].
        test_size (int, optional): Size of the testing set. Defaults to 200.
        
    Returns:
        None: Results are saved to files in the specified output directories
    """
    # preparation
    dirs = prepare_directories(output_dir)
    query, predicates, original_param_list = load_template(original_file, query_id)
        
    # match param_generator _get_numeric_operator_adjustment
    def adjust_predicates(predicates):
        for predicate in predicates:
            if predicate.get("data_type") == "int":
                if "min" in predicate:
                    predicate["min"] -= 1  # round -> min -= 1
                if "max" in predicate:
                    predicate["max"] += 1  # round -> max += 1

        return predicates
    predicates = adjust_predicates(predicates)
    
    # find in operator indices, no need for escape_single_quotes for that specific param
    def find_in_operator_indices(predicates):
        """Find indices of predicates where operator is 'in'."""
        return [index for index, predicate in enumerate(predicates) 
                if predicate.get('operator', '').lower() == 'in']
    in_operator_indices = find_in_operator_indices(predicates)
    
    # Escape single quotes
    param_list = [
        [
            param if i in in_operator_indices else escape_single_quotes(param)
            for i, param in enumerate(params)
        ]
        for params in original_param_list
    ]
    param_list = test_query_returns(query, param_list, db_config, query_id, method="kepler")
    
    # Process train and testing files
    train_dict_dict, test_dict = split_literals_and_store_with_frequency(query_id, param_list, seed_value, output_dir, train_size_list, test_size)
    process_train_and_test_data(dirs, query_id, query, predicates, train_dict_dict, test_dict)

    
  
def main(unused_argv):
    # metadata related args
    selection = _SELECTION.value
    seed_value = _RANDOM_SEED.value
    output_dir = _OUTPUT_DIR.value
    db_name = _DB_NAME.value
    
    # specific query related args
    query_id = _QUERY_ID.value
    template_file = _TEMPLATE_FILE.value
    metadata_file = _METADATA_FILE.value
    
    # training & testing related args
    count = _COUNT.value
    test_size = _TEST_SIZE.value
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    db_config = {
        'dbname': db_name,
        'user': 'kepler',
        'password': 'kepler',
        'host': 'localhost',
        'port': '5431'
    }
    print(db_config['dbname'])
    
    # save input files
    if selection == "cardinality":
        base_dir = "cardinality/inputs"
        full_output_dir = os.path.join(output_dir, base_dir)
        param_choice_list = gen_param_by_cardinality(db_name, template_file, N = 10, K = count) # K change from 50 to count
        print(len(param_choice_list), len(param_choice_list[0]), [param_choice_list[i][0] for i in range(len(param_choice_list))])
        gen_full_cardinality(db_config, full_output_dir, template_file, query_id, param_choice_list, seed_value, count, train_size_list=[50, 400], test_size=test_size)
    elif selection == "cardinality_full":
        base_dir = "cardinality_full/inputs"
        full_output_dir = os.path.join(output_dir, base_dir)
        param_choice_list = gen_param_by_cardinality_full(db_name, template_file, N = 10, K = count) # K change from 50 to count
        print(len(param_choice_list), len(param_choice_list[0]), [param_choice_list[i][0] for i in range(len(param_choice_list))])
        gen_full_cardinality(db_config, full_output_dir, template_file, query_id, param_choice_list, seed_value, count, train_size_list=[50, 400], test_size=test_size, method="cardinality_full", shuffle=True) # shuffle after one-on-one
    elif selection == "csv":
        # save input files
        base_dir = "csv/inputs"
        full_output_dir = os.path.join(output_dir, base_dir)
        
        # Check if all required values are provided
        check_required_values(metadata_file)
                
        # Step 0: find corresponding params & frequencies
        with open(_METADATA_FILE.value) as f:
            metadata = json.load(f)
        with open(_TEMPLATE_FILE.value) as f:
            template_file_content = json.load(f)
        
        template_predicates = template_file_content[query_id]["predicates"]
            
        matching_elements = get_matching_elements(metadata, template_predicates)
        # print(f'csv matching elements:\n {matching_elements}')
        params, frequencies = [], []
        for element in matching_elements:
            if element:
                params.append(element["params"])
                frequencies.append(element["frequencies"])
        print('template', len(template_predicates), '- generate param', len(params))
                
        # Step 1: generate param list based on the frequency
        sampled_literals = gen_sql_by_template(params, frequencies, count)
        sampled_literals = np.array(sampled_literals)
        np.random.shuffle(sampled_literals)
        sampled_literals = sampled_literals.tolist()
        # sampled_literals = filter_invalid_combinations(sampled_literals, template_predicates)
        print(len(sampled_literals), len(sampled_literals[0]))
        
        # execute
        query, _, _ = load_template(template_file, query_id)
        sampled_literals = test_query_returns(query, sampled_literals, db_config, query_id, method="csv")
                
        # Step 2: get the train & test frequency
        train_dict_dict, test_dict = split_literals_and_store_with_frequency(query_id, sampled_literals, seed_value, full_output_dir, train_size_list=[50, 400], test_size=test_size)
        
        # Step 3: generate kepler & PQO input format
        gen_full_csv(full_output_dir, template_file, query_id, train_dict_dict, test_dict)
    elif selection == "kepler":
        output_dir = run_kepler_pipeline(db_name, template_file, count)
        new_template = os.path.join(output_dir, f"{query_id}-{count}.json")
        print(new_template)
        gen_full_kepler(output_dir, new_template, query_id, seed_value, db_config, train_size_list=[50, 400], test_size=200)
        


if __name__ == "__main__":
    app.run(main)
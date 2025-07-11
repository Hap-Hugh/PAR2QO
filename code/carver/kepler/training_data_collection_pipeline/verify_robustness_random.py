from collections import defaultdict
import csv
import psycopg2
import json
from postgres import *
from utility import modify_query, get_count
import statistics
import os
import numpy as np
from absl import app
from absl import flags

random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

db_config = {
    'dbname': 'imdbloadbase',
    'user': 'kepler',
    'password': 'kepler',
    'host': 'localhost',
    'port': '5431'
}

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")

_MODEL_RESULT_CSV_FILE = flags.DEFINE_string(
    "model_result_csv_file", None,
    "CSV file regarding kepler model result.")
flags.mark_flag_as_required("model_result_csv_file")

_TESTING_PARAMETER_VALUES = flags.DEFINE_string(
    "testing_parameter_values",
    None,
    "File containing parameter values for testing set."
)
flags.mark_flag_as_required("testing_parameter_values")

_TEMPLATE_FILE = flags.DEFINE_string(
    "template", None,
    "original template.")
flags.mark_flag_as_required("model_result_csv_file")

_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")
_DATABASE = flags.DEFINE_string("database", "imdbloadbase", "Name of the database")
_VERIFY = flags.DEFINE_bool("verify", True, "Whether start verify")
_DROP = flags.DEFINE_bool("drop", False, "Whether drop the tables")
_TABLE_INSTANCE = flags.DEFINE_string("i", None, "db instance to check robustness")
flags.mark_flag_as_required("i")



#####################
# db modification
def verify_by_multiple_random_instances(i, window_size):
    # db_name = "imdbloadbase"
    # conn = psycopg2.connect("host=/tmp dbname=" + db_name)
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )
    conn.set_session(autocommit=True)
    cursor_ = conn.cursor()
    cache = {}


    # Define the number of samples and the sampling range
    raw_size_title = 2528312
    new_tables = []

    # Loop over the samples and generate the sampled tables for all the required tables
    tables_to_sample_by_title = ["movie_companies", "movie_keyword", "cast_info", "movie_link", "movie_info",
                                 "complete_cast", "aka_title", "movie_info_idx"]


    try:


        # Construct the SQL query to generate the random_title table for the current sample
        query = f"""
            CREATE TABLE random_title_{i} AS 
            SELECT * FROM title 
            TABLESAMPLE BERNOULLI({window_size}) 
            REPEATABLE({random_seeds[i]}) 
            ORDER BY production_year;
        """
        
        # Execute the query
        new_tables.append(f"random_title_{i}")
        cursor_.execute(query)
        print(f"Generated random_title_{i} with random seed {random_seeds[i]}, window_size = {window_size}")
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])

        # Generate sampled tables based on title
        for table_name in tables_to_sample_by_title:
            query_ = f"""
                CREATE TABLE random_{table_name}_{i} AS
                SELECT *
                FROM {table_name}
                WHERE {table_name}.movie_id IN (
                    SELECT id
                    FROM random_title_{i}
                );
            """
            new_tables.append(f"random_{table_name}_{i}")
            cursor_.execute(query_)
            print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
            cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])

        # Generate sampled keyword based on movie_keyword
        query_ = f"""
            CREATE TABLE random_keyword_{i} AS
            SELECT *
            FROM keyword
            WHERE keyword.id IN (
                SELECT keyword_id
                FROM random_movie_keyword_{i}
            );
        """
        new_tables.append(f"random_keyword_{i}")
        cursor_.execute(query_)
        
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])




        # Generate sampled company_name based on movie_companies
        query_ = f"""
            CREATE TABLE random_company_name_{i} AS
            SELECT *
            FROM company_name
            WHERE company_name.id IN (
                SELECT company_id
                FROM random_movie_companies_{i}
            );
        """
        new_tables.append(f"random_company_name_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])


        # Generate sampled aka_name based on cast_info
        query_ = f"""
            CREATE TABLE random_aka_name_{i} AS
            SELECT *
            FROM aka_name
            WHERE aka_name.id IN (
                SELECT person_id
                FROM random_cast_info_{i}
            );
        """
        new_tables.append(f"random_aka_name_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])



        # Generate sampled name based on aka_name
        query_ = f"""
            CREATE TABLE random_name_{i} AS
            SELECT *
            FROM name
            WHERE name.id IN (
                SELECT id
                FROM random_aka_name_{i}
            );
        """
        new_tables.append(f"random_name_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])



        # Generate sampled person_info based on name
        query_ = f"""
            CREATE TABLE random_person_info_{i} AS
            SELECT *
            FROM person_info
            WHERE person_info.person_id IN (
                SELECT id
                FROM random_name_{i}
            );
        """
        new_tables.append(f"random_person_info_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])




        # Generate sampled link_type based on movie_link
        query_ = f"""
            CREATE TABLE random_link_type_{i} AS
            SELECT *
            FROM link_type
            WHERE link_type.id IN (
                SELECT link_type_id
                FROM random_movie_link_{i}
            );
        """
        new_tables.append(f"random_link_type_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])




        # Generate sampled info_type based on person_info and movie_info
        query_ = f"""
            CREATE TABLE random_info_type_{i} AS
            SELECT *
            FROM info_type
            WHERE info_type.id IN (
                SELECT info_type_id
                FROM random_movie_info_{i}
            ) OR info_type.id IN (
                SELECT info_type_id
                FROM random_person_info_{i}
            ) OR info_type.id IN (
                SELECT info_type_id
                FROM random_movie_info_idx_{i}
            );
        """
        new_tables.append(f"random_info_type_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])

        
        # Generate company_type
        query_ = f"""
            CREATE TABLE random_company_type_{i} AS
            SELECT *
            FROM company_type
            WHERE company_type.id IN (
                SELECT company_type_id
                FROM random_movie_companies_{i}
            );
        """
        new_tables.append(f"random_company_type_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])


        # Generate random_kind_type
        query_ = f"""
            CREATE TABLE random_kind_type_{i} AS
            SELECT *
            FROM kind_type
            WHERE kind_type.id IN (
                SELECT kind_id
                FROM random_title_{i}
            );
        """
        new_tables.append(f"random_kind_type_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])


        # Generate char_name
        query_ = f"""
            CREATE TABLE random_char_name_{i} AS
            SELECT *
            FROM char_name
            WHERE char_name.id IN (
                SELECT person_role_id
                FROM random_cast_info_{i}
            );
        """
        new_tables.append(f"random_char_name_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])


        # Generate role_type
        query_ = f"""
            CREATE TABLE random_role_type_{i} AS
            SELECT *
            FROM role_type
            WHERE role_type.id IN (
                SELECT role_id
                FROM random_cast_info_{i}
            );
        """
        new_tables.append(f"random_role_type_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])


        # Generate comp_cast_type
        query_ = f"""
            CREATE TABLE random_comp_cast_type_{i} AS
            SELECT *
            FROM comp_cast_type
            WHERE comp_cast_type.id IN (
                SELECT status_id
                FROM random_complete_cast_{i}
            ) OR comp_cast_type.id IN (
                SELECT subject_id
                FROM random_complete_cast_{i}
            );
        """
        new_tables.append(f"random_comp_cast_type_{i}")
        cursor_.execute(query_)
        print(f"\"{new_tables[-1]}\": {get_count(cursor_, new_tables[-1])},")
        cache[new_tables[-1]] = get_count(cursor_, new_tables[-1])


        # create primary key
        for t in new_tables:
            query_ = f"ALTER TABLE {t} ADD PRIMARY KEY (id);"
            cursor_.execute(query_)


        # create index for fk
        index_query = f'''create index company_id_movie_companies on movie_companies(company_id);
                        create index company_type_id_movie_companies on movie_companies(company_type_id);
                        create index info_type_id_movie_info_idx on movie_info_idx(info_type_id);
                        create index info_type_id_movie_info on movie_info(info_type_id);
                        create index info_type_id_person_info on person_info(info_type_id);
                        create index keyword_id_movie_keyword on movie_keyword(keyword_id);
                        create index kind_id_aka_title on aka_title(kind_id);
                        create index kind_id_title on title(kind_id);
                        create index linked_movie_id_movie_link on movie_link(linked_movie_id);
                        create index link_type_id_movie_link on movie_link(link_type_id);
                        create index movie_id_aka_title on aka_title(movie_id);
                        create index movie_id_cast_info on cast_info(movie_id);
                        create index movie_id_complete_cast on complete_cast(movie_id);
                        create index subject_id_complete_cast on complete_cast(subject_id);
                        create index status_id_complete_cast on complete_cast(status_id);
                        create index movie_id_movie_companies on movie_companies(movie_id);
                        create index movie_id_movie_info_idx on movie_info_idx(movie_id);
                        create index movie_id_movie_keyword on movie_keyword(movie_id);
                        create index movie_id_movie_link on movie_link(movie_id);
                        create index movie_id_movie_info on movie_info(movie_id);
                        create index person_id_aka_name on aka_name(person_id);
                        create index person_id_cast_info on cast_info(person_id);
                        create index person_id_person_info on person_info(person_id);
                        create index person_role_id_cast_info on cast_info(person_role_id);
                        create index role_id_cast_info on cast_info(role_id);'''

        index_query = index_query.replace('(', f'_{i}(')
        index_query = index_query.replace(' on ', f'_random_{i} on random_')
        cursor_.execute(index_query)
        print(index_query)

        # add fk
        fk_query = '''
                ALTER TABLE title ADD FOREIGN KEY (kind_id) REFERENCES kind_type;
                ALTER TABLE aka_name ADD FOREIGN KEY (id) REFERENCES name;
                -- psql:add_fks.sql:3: ERROR:  insert or update on table "cast_info" violates foreign key constraint "cast_info_person_id_fkey"
                --   DETAIL:  Key (person_id)=(901344) is not present in table "aka_name".
                -- ALTER TABLE cast_info ADD FOREIGN KEY (person_id) REFERENCES aka_name;
                ALTER TABLE cast_info ADD FOREIGN KEY (movie_id) REFERENCES title;
                ALTER TABLE cast_info ADD FOREIGN KEY (person_role_id) REFERENCES char_name;
                ALTER TABLE cast_info ADD FOREIGN KEY (role_id) REFERENCES role_type;
                ALTER TABLE complete_cast ADD FOREIGN KEY (movie_id) REFERENCES title;
                ALTER TABLE complete_cast ADD FOREIGN KEY (subject_id) REFERENCES comp_cast_type;
                ALTER TABLE complete_cast ADD FOREIGN KEY (status_id) REFERENCES comp_cast_type;
                ALTER TABLE movie_companies ADD FOREIGN KEY (movie_id) REFERENCES title;
                ALTER TABLE movie_info ADD FOREIGN KEY (movie_id) REFERENCES title;
                ALTER TABLE movie_info ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
                ALTER TABLE movie_info_idx ADD FOREIGN KEY (movie_id) REFERENCES title;
                ALTER TABLE movie_info_idx ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
                ALTER TABLE movie_keyword ADD FOREIGN KEY (movie_id) REFERENCES title;
                ALTER TABLE movie_keyword ADD FOREIGN KEY (keyword_id) REFERENCES keyword;
                ALTER TABLE movie_link ADD FOREIGN KEY (movie_id) REFERENCES title;
                ALTER TABLE movie_link ADD FOREIGN KEY (link_type_id) REFERENCES link_type;
                ALTER TABLE person_info ADD FOREIGN KEY (person_id) REFERENCES name;
                ALTER TABLE person_info ADD FOREIGN KEY (info_type_id) REFERENCES info_type;
            '''
        # TODO check if alter successfully
        fk_query = fk_query.replace(' ADD ', f'_{i} ADD ')
        fk_query = fk_query.replace(';', f'_{i};')
        fk_query = fk_query.replace('TABLE ', f'TABLE random_')
        fk_query = fk_query.replace('REFERENCES ', f'REFERENCES random_')
        cursor_.execute(fk_query)
        print(fk_query)


        # create histgram
        cursor_.execute("ANALYZE VERBOSE;")
        print("Done ANALYZE VERBOSE;")
    except Exception as e:
        print(e)
        drop_table(new_tables)

    if i != "base":
        conn.close()
        return new_tables
    print(cache)
        

    # Close the database connection
    conn.close()
    return cache


def drop_table(new_tables):
    # db_name = "imdbloadbase"
    # conn = psycopg2.connect("host=/tmp dbname=" + db_name)
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )
    conn.set_session(autocommit=True)
    cursor_ = conn.cursor()
    for t in new_tables:
        try:
            cursor_.execute(f"drop table {t} CASCADE;")
            print(f"drop table {t} CASCADE;")
        except:
            continue
    return


def drop_sampled_tables(i):
    """
    Drop all sampled tables with the given suffix '_{i}'.
    
    Parameters:
    - i: The suffix to append to each table name (e.g., '1', '2').
    """
    # db_name = "imdbloadbase"
    # conn = psycopg2.connect("host=/tmp dbname=" + db_name)
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )
    conn.set_session(autocommit=True)
    cursor_ = conn.cursor()
    
    # backup table prefix
    table_prefixes = [
        "random_title_base",
        "random_movie_companies_base",
        "random_movie_keyword_base",
        "random_cast_info_base",
        "random_movie_link_base",
        "random_movie_info_base",
        "random_complete_cast_base",
        "random_aka_title_base",
        "random_movie_info_idx_base",
        "random_keyword_base",
        "random_company_name_base",
        "random_aka_name_base",
        "random_name_base",
        "random_person_info_base",
        "random_link_type_base",
        "random_info_type_base",
        "random_company_type_base",
        "random_kind_type_base",
        "random_char_name_base",
        "random_role_type_base",
        "random_comp_cast_type_base"
    ]

    for prefix in table_prefixes:
        table_name = prefix.replace("_base", f"_{i}")
        query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
        print(f"Executing: {query}")
        cursor_.execute(query)



''' backup
drop table random_title_0 CASCADE;
drop table random_movie_companies_0 CASCADE;
drop table random_movie_keyword_0 CASCADE;
drop table random_cast_info_0 CASCADE;
drop table random_movie_link_0 CASCADE;
drop table random_movie_info_0 CASCADE;
drop table random_complete_cast_0 CASCADE;
drop table random_aka_title_0 CASCADE;
drop table random_movie_info_idx_0 CASCADE;
drop table random_keyword_0 CASCADE;
drop table random_company_name_0 CASCADE;
drop table random_aka_name_0 CASCADE;
drop table random_name_0 CASCADE;
drop table random_person_info_0 CASCADE;
drop table random_link_type_0 CASCADE;
drop table random_info_type_0 CASCADE;
drop table random_company_type_0 CASCADE;
drop table random_kind_type_0 CASCADE;
drop table random_char_name_0 CASCADE;
drop table random_role_type_0 CASCADE;
drop table random_comp_cast_type_0 CASCADE;
'''

'''
SELECT column_name
FROM information_schema.constraint_column_usage
WHERE table_name = 'random_keyword_0'
AND constraint_name = (
    SELECT constraint_name
    FROM information_schema.table_constraints
    WHERE table_name = 'random_keyword_0'
    AND constraint_type = 'PRIMARY KEY'
);
'''

#####################
# read kepler plan output
def load_data(json_file_path, csv_file_path, iteration):
    # load testing data
    with open(json_file_path, 'r') as json_file:
        query_data = json.load(json_file)

    # Use defaultdict to store param_plan_map with frequencies
    param_plan_map = defaultdict(lambda: {'confidence': -1, 'plan_id': -1, 'plan_content': '', 'frequency': 0})

    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # print(row['iteration'])
            if int(row['iteration']) == iteration:
                params = row['params']
                confidence = float(row['confidence'])

                # If this param already exists and the new one has a higher confidence
                if confidence > param_plan_map[params]['confidence']:
                    param_plan_map[params] = {
                        'confidence': confidence,
                        'plan_id': row['plan_id'],
                        'plan_content': row['plan_content'],
                        'frequency': param_plan_map[params]['frequency'] + 1,
                    }
                else:
                    # Just increment the frequency if not updating the plan
                    param_plan_map[params]['frequency'] += 1

    # Convert the defaultdict to a list of dicts
    param_plan_map = [{'params': params, **details} for params, details in param_plan_map.items()]
    
    # Extract all confidence values from param_plan_map
    confidence_values = [entry['confidence'] for entry in param_plan_map]
    
    # Calculate average and standard deviation of confidence values
    avg_confidence = statistics.mean(confidence_values) if confidence_values else 0
    std_confidence = statistics.stdev(confidence_values) if len(confidence_values) > 1 else 0
    
    return query_data, param_plan_map, avg_confidence, std_confidence

# def execute_query(cursor, query):
#     start_time = time.time()
#     cursor.execute(query)
#     return time.time() - start_time

def execute_query(cursor, query):
    """
    Execute the given query and return the execution time, result row count, and cost.
    """
    start_time = time.time()
    cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
    execution_time = time.time() - start_time

    # Parse EXPLAIN JSON to extract row count and cost
    explain_result = cursor.fetchone()[0]
    
    # Ensure explain_result is a JSON string
    if isinstance(explain_result, list):
        # If it's a list, convert it back to a JSON string
        explain_result = json.dumps(explain_result)
        
    explain_data = json.loads(explain_result)

    # Extract row count and cost
    total_cost = explain_data[0]['Plan']['Total Cost']
    row_count = explain_data[0]['Plan']['Plan Rows']

    return execution_time, row_count, total_cost



def test_query_latency(query_id, new_query_template, param_plan_map, db_config):
    # connect to db
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )

    cursor = conn.cursor()

    results = []
    
    # TODO: For Slack: Only test first 100 entries with the highest confidence
    dbname = db_config['dbname']
    if dbname == "stack":
        filtered_param_plan_map = sorted(param_plan_map, key=lambda x: x['confidence'], reverse=True)[:100]
    else:
        filtered_param_plan_map = param_plan_map
    
    counter = 0

    # test hinted query
    for entry in filtered_param_plan_map:
        params = entry['params']
        plan_id = int(entry['plan_id'])
        plan_content = entry['plan_content']
        frequency = entry['frequency']
        confidence = entry['confidence']
        
        # Convert params from string representation of a list to an actual list
        params_list = eval(params)
        
        # get full query instance new_query_template
        query = new_query_template
        for i, param in enumerate(params_list):
            query = query.replace(f"@param{i}", param.strip())
            
        if counter == 0:
            print(query)

        # add hint
        if plan_id != -1: # CHANGE: default plan regression
            hinted_query = f"{plan_content} {query}"
        else:
            hinted_query = query

        hinted_latencies = []
        default_latencies = []
        hinted_cardinality, hinted_cost, default_cardinality, default_cost = 0, 0, 0, 0

        # Run each query 5 times
        cursor.execute("LOAD 'pg_hint_plan';")
        cursor.execute('COMMIT')
        
        for range_num in range(5):
            hinted_latency, h_cardinality, h_cost = execute_query(cursor, hinted_query)
            cursor.execute('COMMIT')  # commit after each execution to clear any locks
            hinted_latencies.append(hinted_latency)
            
            default_latency, d_cardinality, d_cost = execute_query(cursor, query)
            cursor.execute('COMMIT')  # commit after each execution to clear any locks
            default_latencies.append(default_latency)
            
            if range_num == 0:
                hinted_cardinality = h_cardinality
                hinted_cost = h_cost
                default_cardinality = d_cardinality
                default_cost = d_cost


        # Calculate median of latencies
        median_hinted_latency = np.median(hinted_latencies)
        median_default_latency = np.median(default_latencies)

        results.append({
            'params': params,
            'frequency': frequency,
            'run': 'median',
            'hinted_latency': median_hinted_latency * 1000,
            'default_latency': median_default_latency * 1000,
            'hinted_row_count': hinted_cardinality,
            'default_row_count': default_cardinality,
            'hinted_cost': hinted_cost,
            'default_cost': default_cost,
            'confidence': confidence,
            'plan_id': plan_id,
            'plan_content': plan_content
        })

        print(f"{query_id} - {counter} - Params: {params}")
        print(f"Hinted Latency (median): {median_hinted_latency * 1000} ms")
        print(f"Default Latency (median): {median_default_latency * 1000} ms")
        print(f"Hinted row count: {hinted_cardinality}")
        print(f"Default row count: {default_cardinality}")
        print(f"Hinted cost: {hinted_cost}")
        print(f"Default cost: {default_cost}")
        print("="*50)
        
        counter += 1

    # close db
    cursor.close()
    conn.close()

    return results

def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['params', 'frequency', 'run', 'hinted_latency', 'default_latency', 'hinted_row_count', 'default_row_count', 'hinted_cost', 'default_cost', 'confidence', 'plan_id', 'plan_content'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")

def calculate_hint_efficiency(results, avg_confidence, std_confidence):
    total = sum(result['frequency'] for result in results)
    improved = sum(result['frequency'] for result in results if result['hinted_latency'] < result['default_latency'])
    
    # Total latency calculation
    total_hinted_latency = sum(result['hinted_latency'] * result['frequency'] for result in results)
    total_default_latency = sum(result['default_latency'] * result['frequency'] for result in results)
    avg_hinted_latency = total_hinted_latency / total if total > 0 else 0
    avg_default_latency = total_default_latency / total if total > 0 else 0
    
    # Improved latency calculation
    improved_hinted_latency = sum(result['hinted_latency'] * result['frequency'] for result in results if result['hinted_latency'] < result['default_latency'])
    improved_default_latency = sum(result['default_latency'] * result['frequency'] for result in results if result['hinted_latency'] < result['default_latency'])
    avg_improved_hinted_latency = improved_hinted_latency / improved if improved > 0 else 0
    avg_improved_default_latency = improved_default_latency / improved if improved > 0 else 0
    
    # Improved confidence info
    improved_confidences = [result['confidence'] for result in results if result['hinted_latency'] < result['default_latency']]
    avg_improved_confidence = statistics.mean(improved_confidences) if improved_confidences else 0
    std_improved_confidence = statistics.stdev(improved_confidences) if len(improved_confidences) > 1 else 0


    return {
        'improved_count': improved,
        'total': total,
        'avg_improved_hinted_latency': avg_improved_hinted_latency,
        'avg_improved_default_latency': avg_improved_default_latency,
        'avg_improved_confidence': avg_improved_confidence,
        'std_improved_confidence': std_improved_confidence,
        'avg_hinted_latency': avg_hinted_latency,
        'avg_default_latency': avg_default_latency,
        'avg_confidence': avg_confidence,
        'std_confidence': std_confidence
    }


#####################
# main
def main(unused_arg):
    query_id = _QUERY_ID.value
    model_result_csv = _MODEL_RESULT_CSV_FILE.value
    testing_json = _TESTING_PARAMETER_VALUES.value
    output_dir = _OUTPUT_DIR.value
    db = _DATABASE.value
    i = _TABLE_INSTANCE.value
    
    # original sql template
    template = _TEMPLATE_FILE.value
    with open(template, 'r') as json_file:
        query_data = json.load(json_file)
    sql_template = query_data[query_id]["query"]
    
    summary_results = []
    
    # Determine iteration based on file name
    if model_result_csv.endswith('_batch_0.csv'):
        iteration = 0
    else:
        iteration = 10
    
    # load data
    _, param_plan_map, avg_confidence, std_confidence = load_data(testing_json, model_result_csv, iteration)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Verification
    start_verify = _VERIFY.value
    if start_verify:
        s = int(i)  
           
        # Create a subfolder
        verify_dir = os.path.join(output_dir, f'verify_{s}')
        os.makedirs(verify_dir, exist_ok=True)
            
        for check_num in range(9):
            # Initialization
            output_file = os.path.join(verify_dir, f'{query_id}_latency_comparison_{check_num}.csv')
            
            # get sql template (sample version)
            new_sql_template = modify_query("random_", f"_{check_num}", sql_template)
            
            # test latency
            results = test_query_latency(query_id, new_sql_template, param_plan_map, db_config)
            
            # save result
            save_results_to_csv(results, output_file)

            # calculate efficiency and add to summary
            efficiency = calculate_hint_efficiency(results, avg_confidence, std_confidence)
            summary_results.append({
                'query_id': f'{query_id}_sample_{s}',
                'improved_count': efficiency['improved_count'],
                'total': efficiency['total'],
                'percentage_improved': (efficiency['improved_count'] / efficiency['total']) * 100,  # convert to percentage
                'avg_improved_hinted_latency': efficiency['avg_improved_hinted_latency'],
                'avg_improved_default_latency': efficiency['avg_improved_default_latency'],
                'avg_improved_confidence': efficiency['avg_improved_confidence'],
                'std_improved_confidence': efficiency['std_improved_confidence'],
                'avg_hinted_latency': efficiency['avg_hinted_latency'],
                'avg_default_latency': efficiency['avg_default_latency'],
                'avg_confidence': efficiency['avg_confidence'],
                'std_confidence': efficiency['std_confidence']
            })
            
        # Save summary results to CSV
        summary_file = os.path.join(verify_dir, f'{query_id}_efficiency_summary.csv')

        # Check if the file already exists and has content
        file_exists = os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0

        with open(summary_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['query_id', 'improved_count', 'total', 'percentage_improved', 'avg_improved_hinted_latency', 'avg_improved_default_latency', 'avg_improved_confidence', 'std_improved_confidence', 'avg_hinted_latency', 'avg_default_latency', 'avg_confidence', 'std_confidence'])

            # Only write the header if the file is new or empty
            if not file_exists:
                writer.writeheader()

            for result in summary_results:
                writer.writerow(result)

        print(f"Summary results saved to {summary_file}")
    else:
        for s in range(0, 9):
            print(s)
            verify_by_multiple_random_instances(s, 20)


    drop_tables = _DROP.value
    if drop_tables:
        for s in range(0, 9):
            print(s)
            drop_sampled_tables(s)
    
    
if __name__ == "__main__":
    app.run(main)

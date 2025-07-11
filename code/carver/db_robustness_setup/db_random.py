import psycopg2
from postgres import *
from utility import get_count

random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# TODO: change configuration
db_config = {
    'dbname': 'imdbloadbase',
    'user': 'kepler',
    'password': 'kepler',
    'host': 'localhost',
    'port': '5431'
}


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


#####################
# main
def main():
    
    # TODO: change range
    for s in range(0, 9):
    # for s in [1]:
        print(s)
        drop_sampled_tables(s)

    for s in range(0, 9):
    # for s in [1]:
        print(s)
        verify_by_multiple_random_instances(s, 20)


    
    
if __name__ == "__main__":
    main()
import os
import json
from datetime import datetime, date
from decimal import Decimal
from kepler.training_data_collection_pipeline.query_utils import QueryManager, DatabaseConfiguration

# Function to convert Decimal objects to float
def convert_decimal_to_float(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_decimal_to_float(value) for key, value in obj.items()}
    return obj

# Settings
database_config = DatabaseConfiguration(
    dbname='dsb',
    user='kepler',
    password='kepler',
    host='localhost'
)

# connect to db
query_manager = QueryManager(database_configuration=database_config)

# get all table and columns
query_manager.refresh_cursor()
query_manager._cursor.execute("""
    SELECT table_name, column_name 
    FROM information_schema.columns 
    WHERE table_schema = 'public'
""")
tables_columns = query_manager._cursor.fetchall()

# create output folder
output_dir = 'dsb_training_metadata'
os.makedirs(output_dir, exist_ok=True)

for table, column in tables_columns:
    distinct_values = convert_decimal_to_float(query_manager.get_distinct_values(table, column))
    most_common_values = convert_decimal_to_float(query_manager.get_most_common_values(table, column))
    most_common_frequencies = convert_decimal_to_float(query_manager.get_most_common_frequencies(table, column))

    # save distinct_values
    output_file_distinct_values = os.path.join(output_dir, f'{table}-{column}-distinct_values')
    with open(output_file_distinct_values, 'w') as f:
        json.dump(distinct_values, f)  # save as JSON list

    # save most_common_values
    output_file_most_common_values = os.path.join(output_dir, f'{table}-{column}-most_common_values')
    with open(output_file_most_common_values, 'w') as f:
        json.dump(most_common_values, f)  # save as JSON list

    # save most_common_frequencies
    output_file_most_common_frequencies = os.path.join(output_dir, f'{table}-{column}-most_common_frequencies')
    with open(output_file_most_common_frequencies, 'w') as f:
        json.dump(most_common_frequencies, f)  # save as JSON list

# close connection
query_manager._cursor.close()
query_manager._conn.close()

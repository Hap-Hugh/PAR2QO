import os
import shutil
import re

def organize_pqo_files(base_src_dir):
   # Iterate through all method folders 
   for method_dir in os.listdir(base_src_dir):
       if not method_dir.endswith('_workload'):
           continue
           
       method = method_dir.replace('_workload', '')
       src_dir = os.path.join(base_src_dir, method_dir)
       
       # Iterate through all CSV files
       for filename in os.listdir(src_dir):
           if not filename.endswith('.csv'):
               continue
               
           # Extract parameters using regex
           pattern = r'pqo-q(\d+)-t(\d+)-(\d+)-b0\.5_freq\.csv'
           match = re.match(pattern, filename)
           
           if not match:
               print(f"Skipping file {filename} - doesn't match expected pattern")
               continue
               
           qid, tid, n = match.groups()
           
           # Build new target path
           new_dir = f"imdb_PQO_repo/imdb_{qid}-{tid}_original_PQO/{method}"
           os.makedirs(new_dir, exist_ok=True)
           
           # Full paths for source and destination files
           src_file = os.path.join(src_dir, filename)
           dst_file = os.path.join(new_dir, filename)
           
           # Copy file to new location
           print(f"Moving {src_file} to {dst_file}")
           shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    base_src_dir = "imdb_PQO"
    organize_pqo_files(base_src_dir)
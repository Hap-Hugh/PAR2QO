import os
import json

robustness_choices = ['category', 'random', 'sliding']
db_instance_range = [1, 4]
methods = ['cardinality', 'csv', 'kepler']
input_folder = ['inputs/PQO/query', 'inputs/testing', 'inputs/training']
query_ids = [f"{i}-0" for i in range(1, 34)]


# 11-0, 21-0, 27-0, 32-0 originally has min(lt.link) in select condition
replace_contents = {
    '1-0': 'SELECT MIN(mc.note) AS production_note, MIN(t.title) AS movie_title, MIN(t.production_year) AS movie_year',
    '2-0': 'SELECT MIN(t.title) AS movie_title', 
    '3-0': 'SELECT MIN(t.title) AS movie_title', 
    '4-0': 'SELECT MIN(mi_idx.info) AS rating, MIN(t.title) AS movie_title', 
    '5-0': 'SELECT MIN(t.title) AS typical_european_movie', 
    '6-0': 'SELECT MIN(k.keyword) AS movie_keyword, MIN(n.name) AS actor_name, MIN(t.title) AS marvel_movie', 
    '7-0': 'SELECT MIN(n.name) AS of_person, MIN(t.title) AS biography_movie', 
    '8-0': 'SELECT MIN(an.name) AS actress_pseudonym, MIN(t.title) AS japanese_movie_dubbed', 
    '9-0': 'SELECT MIN(an.name) AS alternative_name, MIN(chn.name) AS character_name, MIN(t.title) AS movie', 
    '10-0': 'SELECT MIN(chn.name) AS uncredited_voiced_character, MIN(t.title) AS russian_movie', 
    '11-0': 'SELECT MIN(cn.name) AS from_company, MIN(t.title) AS non_polish_sequel_movie', 
    '12-0': 'SELECT MIN(cn.name) AS movie_company, MIN(mi_idx.info) AS rating, MIN(t.title) AS drama_horror_movie', 
    '13-0': 'SELECT MIN(mi.info) AS release_date, MIN(miidx.info) AS rating, MIN(t.title) AS german_movie',
    '14-0': 'SELECT MIN(mi_idx.info) AS rating, MIN(t.title) AS northern_dark_movie', 
    '15-0': 'SELECT MIN(mi.info) AS release_date, MIN(t.title) AS internet_movie', 
    '16-0': 'SELECT MIN(an.name) AS cool_actor_pseudonym, MIN(t.title) AS series_named_after_char', 
    '17-0': 'SELECT MIN(n.name) AS member_in_charnamed_american_movie, MIN(n.name) AS a1', 
    '18-0': 'SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(t.title) AS movie_title', 
    '19-0': 'SELECT MIN(n.name) AS voicing_actress, MIN(t.title) AS voiced_movie', 
    '20-0': 'SELECT MIN(t.title) AS complete_downey_ironman_movie', 
    '21-0': 'SELECT MIN(cn.name) AS company_name, MIN(t.title) AS western_follow_up', 
    '22-0': 'SELECT MIN(cn.name) AS movie_company, MIN(mi_idx.info) AS rating, MIN(t.title) AS western_violent_movie', 
    '23-0': 'SELECT MIN(kt.kind) AS movie_kind, MIN(t.title) AS complete_us_internet_movie',
    '24-0': 'SELECT MIN(chn.name) AS voiced_char_name, MIN(n.name) AS voicing_actress_name, MIN(t.title) AS voiced_action_movie_jap_eng', 
    '25-0': 'SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(n.name) AS male_writer, MIN(t.title) AS violent_movie_title', 
    '26-0': 'SELECT MIN(chn.name) AS character_name, MIN(mi_idx.info) AS rating, MIN(n.name) AS playing_actor, MIN(t.title) AS complete_hero_movie', 
    '27-0': 'SELECT MIN(cn.name) AS producing_company, MIN(t.title) AS complete_western_sequel', 
    '28-0': 'SELECT MIN(cn.name) AS movie_company, MIN(mi_idx.info) AS rating, MIN(t.title) AS complete_euro_dark_movie', 
    '29-0': 'SELECT MIN(chn.name) AS voiced_char, MIN(n.name) AS voicing_actress, MIN(t.title) AS voiced_animation', 
    '30-0': 'SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(n.name) AS writer, MIN(t.title) AS complete_violent_movie', 
    '31-0': 'SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(n.name) AS writer, MIN(t.title) AS violent_liongate_movie', 
    '32-0': 'SELECT MIN(t1.title) AS first_movie, MIN(t2.title) AS second_movie', 
    '33-0': 'SELECT MIN(cn1.name) AS first_company, MIN(cn2.name) AS second_company, MIN(mi_idx1.info) AS first_rating, MIN(mi_idx2.info) AS second_rating, MIN(t1.title) AS first_movie, MIN(t2.title) AS second_movie'
}


def replace_select_star_in_file(query_id, file_path, replace_content, is_pqo_query):
    """
    Reads a JSON file, replaces 'SELECT *' in relevant places, and writes back.
    Returns the number of replacements made in the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        replace_count = 0  # Local counter for the file
        modified = False  # To track if we made any changes

        if is_pqo_query:
            # Replace all values in the JSON
            for key in data.keys():
                if isinstance(data[key], str) and 'SELECT *' in data[key]:
                    data[key] = data[key].replace('SELECT *', replace_content)
                    replace_count += 1
                    modified = True
        else:
            # Replace only data[query_id]['query']
            if query_id in data and 'query' in data[query_id]:
                if 'SELECT *' in data[query_id]['query']:
                    data[query_id]['query'] = data[query_id]['query'].replace('SELECT *', replace_content)
                    replace_count += 1
                    modified = True

        # If modifications were made, write back to the file
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"Modified: {file_path} (Replacements: {replace_count})")
        else:
            print(f"No changes: {file_path}")

        return replace_count

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def process_folders():
    """
    Process all JSON files in the folder hierarchy and replace 'SELECT *'.
    """
    base_dir = os.getcwd()
    total_replacements = 0

    for query_id in query_ids:
        replace_content = replace_contents[query_id]
        
        for robustness_choice in robustness_choices:
            for db_instance in db_instance_range:
                for method in methods:
                    for folder in input_folder:
                        folder_path = os.path.join(
                            "0_sample_repo",
                            f"imdb_{query_id}_sample",
                            robustness_choice,
                            f"db_instance_{db_instance}",
                            method,
                            folder
                        )

                        if not os.path.exists(folder_path):
                            continue

                        # Process all JSON files in the folder
                        for root, _, files in os.walk(folder_path):
                            for file in files:
                                if file.endswith('.json'):
                                    file_path = os.path.join(root, file)
                                    is_pqo_query = 'PQO/query' in folder
                                    replacements = replace_select_star_in_file(query_id, file_path, replace_content, is_pqo_query)
                                    total_replacements += replacements

    # Print the total number of replacements
    print(f"\nTotal replacements made across all files: {total_replacements}")


if __name__ == "__main__":
    process_folders()

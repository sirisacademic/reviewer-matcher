# necessary imports 
import requests
import pandas as pd
import time
import os
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ExpertProfiler:
    def __init__(self, data_path, api_key, base_url="https://api.openalex.org", hf_token=None):
        """
        Initialize the ExpertProfiler with a data path, API key, base URL, and optional Hugging Face token.

        Args:
            data_path (str): The path to the data folder.
            api_key (str): The API key for accessing OpenAlex.
            base_url (str): The base URL for the OpenAlex API.
            hf_token (str, optional): Hugging Face API token for accessing private models.
        """
        self.data_path = data_path
        self.api_key = api_key
        self.base_url = base_url
        self.hf_token = hf_token

    def query_openalex_by_name(full_name):
        # constructs the url for the authors endpoint in the openalex api
        url = f"{BASE_URL}/authors"
        # sets the search parameters with the author's full name
        params = {"search": full_name}
        # sends a get request to the api with the url and search parameters
        response = requests.get(url, params=params)
        # checks if the request was successful
        if response.status_code == 200:
            # parses the json response
            data = response.json()
            # checks if there are any results in the response
            if 'results' in data and data['results']:
                # returns the first result if available
                return data['results'][0]
        # returns none if there was no successful response or no results
        return None

    def query_openalex_by_orcid(orcid):
        # constructs the url for the specific author based on their orcid
        url = f"{BASE_URL}/authors/https://orcid.org/{orcid}"
        # sends a get request to the api with the author url
        response = requests.get(url)
        # checks if the request was successful
        if response.status_code == 200:
            # returns the json response
            return response.json()
        # returns none if there was no successful response
        return None
    
    def query_openalex_works(author_id):
        # constructs the url for the works endpoint in the openalex api
        url = f"{BASE_URL}/works"
        # sets the filter to retrieve works by the specified author id
        # sorts results by publication date in descending order
        # limits the results to 5 per page
        params = {"filter": f"authorships.author.id:{author_id}", "sort": "publication_date:desc", "per-page": 5}
        # sends a get request to the api with the url and filter parameters
        response = requests.get(url, params=params)
        # checks if the request was successful
        if response.status_code == 200:
            # returns the json response
            return response.json()
        # returns none if there was no successful response
        return None
    
    def get_author_info(full_name=None, orcid=None):
        # checks if an orcid is provided
        if orcid:
            # queries openalex by orcid and returns the result
            return query_openalex_by_orcid(orcid)
        # checks if a full name is provided if no orcid is given
        elif full_name:
            # queries openalex by full name and returns the result
            return query_openalex_by_name(full_name)
        # returns none if neither orcid nor full name is provided
        return None

    def extract_author_details(author_info):
        # checks if author_info is none, indicating no data was returned
        if author_info is None:
            # returns a dictionary with empty or none values if no author information is available
            return {
                "name": None,
                "orcid": None,
                "works_count": None,
                "cited_by_count": None,
                "recent_work_titles": [],
                "work_types": [],
                "areas_of_interest": []
            }
    
        # extracts the author id from the author_info object
        author_id = author_info.get("id", "").split("/")[-1]
        # queries openalex for works by the author using the author id
        works_info = query_openalex_works(author_id)
    
        # initializes lists and counters for details about the author's works
        recent_work_titles = []
        open_access_count = 0
        work_types = []
        institutions = []
        
        # checks if works information was retrieved and contains results
        if works_info and "results" in works_info:
            # extracts titles of recent works from the results
            recent_work_titles = [work.get("title") for work in works_info["results"]]
            # extracts types of each work from the results
            work_types = [work.get("type") for work in works_info["results"]]
    
        # constructs a dictionary with the author's details
        details = {
            "name": author_info.get("display_name"),
            "orcid": author_info.get("orcid"),
            "works_count": author_info.get("works_count"),
            "cited_by_count": author_info.get("cited_by_count"),
            # includes recent work titles in the details
            "recent_work_titles": recent_work_titles,
            # includes work types in the details
            "work_types": work_types,
            # extracts areas of interest from concepts associated with the author if available
            "areas_of_interest": [concept.get("display_name") for concept in author_info.get("x_concepts", [])] 
            if author_info.get("x_concepts") else [],
        }
        
        # returns the dictionary with the author's details
        return details

    def compute_completeness(data):
        # counts the number of non-null values in each column of the data
        non_null_counts = data.notnull().sum()
        # gets the total number of rows in the data
        total_counts = len(data)
        # calculates the percentage of completeness for each column
        completeness_percentages = (non_null_counts / total_counts) * 100
        # computes the average completeness across all columns
        overall_completeness = completeness_percentages.mean()
        # returns the overall completeness as a single percentage value
        return overall_completeness

    def enrich_author_data(df, method="name"):
        # initializes an empty list to store enriched author data
        enriched_data = []
        # iterates over each row in the dataframe
        for index, row in df.iterrows():
            # retrieves the full name and orcid from the current row
            full_name = row['Full Name:']
            orcid = row['ORCID Number:']
            author_info = None
    
            # retrieves author information based on the specified method
            if method == "orcid" and orcid:
                # queries openalex using orcid if method is set to orcid
                author_info = get_author_info(orcid=orcid)
            elif method == "name" and full_name:
                # queries openalex using full name if method is set to name
                author_info = get_author_info(full_name=full_name)
            elif method == "both":
                # tries to retrieve author info by orcid first if method is set to both
                if orcid:
                    author_info = get_author_info(orcid=orcid)
                # if orcid query fails, attempts retrieval by full name
                if not author_info and full_name:
                    author_info = get_author_info(full_name=full_name)
    
            # extracts details about the author using the author_info
            author_details = extract_author_details(author_info)
    
            # ensures author_details is always a dictionary with consistent keys
            if author_details is None:
                author_details = {
                    "name": None,
                    "orcid": None,
                    "works_count": None,
                    "cited_by_count": None,
                    "recent_work_titles": [],
                    "work_types": [],
                    "areas_of_interest": []
                }
    
            # appends the enriched author details to the list
            enriched_data.append(author_details)
    
        # converts the list of enriched author data into a dataframe and returns it
        return pd.DataFrame(enriched_data)

    def compute_completeness_for_method(df, method):
        # enriches the author data in the dataframe using the specified method
        enriched_df = enrich_author_data(df, method=method)
        # calculates the completeness percentage of the enriched dataframe
        completeness_percentage = compute_completeness(enriched_df)
        # returns the completeness percentage and the enriched dataframe
        return completeness_percentage, enriched_df

    def get_all_publications(author_id):
        # constructs the url for the works endpoint in the openalex api
        url = f"{BASE_URL}/works"
        # sets the filter to retrieve all works by the specified author id with a limit of 200 per page
        params = {
            "filter": f"authorships.author.id:{author_id}",
            "per-page": 200
        }
        # initializes an empty list to store all works retrieved
        all_works = []
        # loops to retrieve all works, handling pagination with the next cursor
        while True:
            # sends a get request to the api with the url and filter parameters
            response = requests.get(url, params=params)
            # checks if the request was successful
            if response.status_code == 200:
                # parses the json response
                data = response.json()
                # extends the list with the results from the current page
                all_works.extend(data['results'])
                # checks if there is a next page cursor in the metadata
                if 'meta' in data and 'next_cursor' in data['meta'] and data['meta']['next_cursor']:
                    # updates the cursor parameter to retrieve the next page
                    params['cursor'] = data['meta']['next_cursor']
                else:
                    # breaks the loop if no further pages are available
                    break
            else:
                # breaks the loop if the request was unsuccessful
                break
            # pauses to avoid overwhelming the api with requests
            time.sleep(1)  
        # returns the list of all works retrieved
        return all_works
    
    def get_author_publications(full_name=None, orcid=None):
        # initializes author_info as none
        author_info = None
        # checks if an orcid is provided
        if orcid:
            # queries openalex by orcid
            author_info = query_openalex_by_orcid(orcid)
        # checks if a full name is provided if no orcid is given
        elif full_name:
            # queries openalex by full name
            author_info = query_openalex_by_name(full_name)
        
        # returns none and an empty list if no author information was found
        if author_info is None:
            return None, []
    
        # extracts the author id from the author_info
        author_id = author_info.get("id", "").split("/")[-1]
        # retrieves all publications for the author using the author id
        publications = get_all_publications(author_id)
        
        # returns the author's display name (or the full name if display name is not available) and the publications list
        return author_info.get("display_name", full_name), publications
    
    def save_publications_to_csv(author_name, publications, save_path):
        # creates a dataframe from the list of publications
        df = pd.DataFrame(publications)
        # adds a column for the author's name to the dataframe
        df['author_name'] = author_name
        # ensures the directory in the save path exists, creating it if necessary
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # saves the dataframe to a csv file at the specified save path without the index column
        df.to_csv(save_path, index=False)
        # prints a message indicating where the enriched data was saved
        print(f"Enriched data saved to {save_path}.")

    def predict_gender_namsor(name, api_key):
        api_key = "d17525f409e675a5c89c428e1aae6871"
        try:
            # removes titles and honorifics from the name using regular expressions
            titles = ['Dr', 'Prof', 'Prof. ', 'Professor', 'Mr', 'Ms', 'Mrs', 'Miss']
            pattern = re.compile(r'\b(?:' + '|'.join(re.escape(title) for title in titles) + r')\.?\b', re.IGNORECASE)
            name = pattern.sub('', name).strip()
            
            # splits the name into first and last names
            first_name, last_name = name.split()[0], name.split()[-1] 
            # constructs the url for the namsor api with the first and last names
            url = f"https://v2.namsor.com/NamSorAPIv2/api2/json/gender/{first_name}/{last_name}"
            # sets the api key in the headers
            headers = {'X-API-KEY': api_key}
            # sends a get request to the namsor api
            response = requests.get(url, headers=headers)
            
            # checks if the response is successful
            if response.status_code == 200:
                # parses the json response and retrieves the likely gender
                result = response.json()
                return result.get('likelyGender', 'unknown')
            else:
                # returns 'unknown' if the request was not successful
                return 'unknown'
        except Exception as e:
            # handles any exceptions and prints an error message
            print(f"Error processing name '{name}': {e}")
            # returns 'unknown' in case of an exception
            return 'unknown'

    def enrich_data_with_predicted_gender(df, api_key):
        # applies the predict_gender_namsor function to each name in the 'Full Name:' column
        df['Predicted Gender:'] = df['Full Name:'].apply(lambda name: predict_gender_namsor(name, api_key))
        # returns the dataframe with the new 'Predicted Gender:' column
        return df

    def save_gender_to_csv(df, folder_path):
        # ensures the directory in the file path exists, creating it if necessary
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        # saves the dataframe to a csv file at the constructed file path without the index column
        df.to_csv(folder_path, index=False)
        # prints a message indicating where the enriched data was saved
        print(f"Enriched data saved to {folder_path}.")
    
    def calculate_gender_metrics(y_true, y_pred):
        # creates a dataframe with actual and predicted values
        results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    
        # calculates the accuracy as the mean of correct predictions
        accuracy = (results['Actual'] == results['Predicted']).mean()
    
        # calculates true positives where both predicted and actual are 'male'
        tp = ((results['Predicted'] == 'male') & (results['Actual'] == 'male')).sum()
        # calculates false positives where predicted is 'male' but actual is 'female'
        fp = ((results['Predicted'] == 'male') & (results['Actual'] == 'female')).sum()
        # calculates precision as tp divided by (tp + fp), handles division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
        # calculates false negatives where predicted is 'female' but actual is 'male'
        fn = ((results['Predicted'] == 'female') & (results['Actual'] == 'male')).sum()
        # calculates recall as tp divided by (tp + fn), handles division by zero
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # calculates f1 score as the harmonic mean of precision and recall, handles division by zero
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        # returns the calculated accuracy, precision, recall, and f1 score
        return accuracy, precision, recall, f1

    def classify_research_phase(text):
        research_phase_classifier = pipeline("text-classification", model="SIRIS-Lab/batracio5")
        # checks if text is a non-empty string
        if isinstance(text, str) and text.strip():
            # uses the research_phase_classifier to get the label of the text
            return research_phase_classifier(text)[0]['label']
        # returns 'unknown' if the input is invalid
        return 'Unknown'  # default label for invalid input

    def classify_domain(text):
        domain_classifier = pipeline("text-classification", model="SIRIS-Lab/biomedicine-classifier")
        # checks if text is a non-empty string
        if isinstance(text, str) and text.strip():
            # uses the domain_classifier to get the label of the text
            return domain_classifier(text)[0]['label']
        # returns 'unknown' if the input is invalid
        return 'Unknown'  # default label for invalid input

    def classify_publications(author_file, output_folder):
        # reads the author file into a dataframe
        df = pd.read_csv(author_file)
    
        # applies research phase and domain classification on each title in the dataframe
        df['research_phase'] = df['title'].apply(classify_research_phase)
        df['domain'] = df['title'].apply(classify_domain)
    
        # extracts author name from the filename for naming the output file
        author_name = os.path.basename(author_file).replace('.csv', '')
        # sets the path for saving the classified output
        output_path = os.path.join(output_folder, f"{author_name}_classified.csv")
        # ensures the output directory exists, creating it if necessary
        os.makedirs(output_folder, exist_ok=True)
        # saves the classified dataframe to the output path
        df.to_csv(output_path, index=False)
        # prints a message confirming the processed file and saved location
        print(f"Processed {author_file}, saved to {output_path}")
    
    def classify_all_publications(input_folder, output_folder):
        # iterates over each file in the input folder
        for author_file in os.listdir(input_folder):
            # checks if the file has a '.csv' extension
            if author_file.endswith('.csv'):
                # processes the publication file and saves it to the output folder
                classify_publications(os.path.join(input_folder, author_file), output_folder)

    def compute_classification_statistics(output_folder):
        # initializes an empty dataframe to hold all data
        all_data = pd.DataFrame()
        
        # iterates over each classified file in the output folder
        for classified_file in os.listdir(output_folder):
            # checks if the file has a '.csv' extension
            if classified_file.endswith('.csv'):
                # reads each classified file and appends its data to the all_data dataframe
                df = pd.read_csv(os.path.join(output_folder, classified_file))
                all_data = pd.concat([all_data, df], ignore_index=True)
        
        # calculates the percentage of each research phase
        research_phase_counts = all_data['research_phase'].value_counts(normalize=True) * 100
        # calculates the percentage of each domain
        domain_counts = all_data['domain'].value_counts(normalize=True) * 100
        
        # returns the calculated statistics for research phase and domain
        return research_phase_counts, domain_counts

    def combine_data(output_folder):
        # initializes an empty list to store dataframes
        data_frames = []
        
        # iterates over each file in the output folder
        for file in os.listdir(output_folder):
            # checks if the file ends with '_classified.csv'
            if file.endswith('_classified.csv'):
                # reads the CSV file into a dataframe
                df = pd.read_csv(os.path.join(output_folder, file))
                # appends the dataframe to the list
                data_frames.append(df)
        
        # combines all the dataframes in the list into a single dataframe
        combined_df = pd.concat(data_frames, ignore_index=True)
        # returns the combined dataframe
        return combined_df

    def extract_mesh_terms_string(mesh_terms):
        # checks if the input is a non-empty string
        if isinstance(mesh_terms, str) and mesh_terms.strip():
            try:
                # converts the string representation of a list to an actual list of dictionaries
                terms = ast.literal_eval(mesh_terms)
                
                # checks if the result is a non-empty list
                if isinstance(terms, list) and terms:
                    # concatenates descriptor names into a single string
                    terms_list = [term['descriptor_name'] for term in terms if 'descriptor_name' in term]
                    # returns the concatenated terms as a string
                    return ' '.join(terms_list)
                else:
                    return ''  # returns empty if the list is empty
                
            except (ValueError, SyntaxError) as e:
                # handles any parsing errors and prints an error message
                print(f"Error parsing MeSH terms: {e}")
                return ''  # returns empty in case of an error
        else:
            return ''  # returns empty if input is not a valid string

    def rank_mesh_terms_across_all(output_folder):
        # loads the combined data from the output folder
        combined_df = combine_data(output_folder)
        
        # extracts mesh terms as strings for tf-idf processing
        combined_df['mesh_term'] = combined_df['mesh'].apply(lambda terms: extract_mesh_terms_string(terms))
        
        # filters out rows with empty mesh term strings for tf-idf
        valid_terms_df = combined_df[combined_df['mesh_term'].str.strip() != '']
        
        # checks if there are any non-empty mesh term strings
        if valid_terms_df.empty:
            print("Error: No valid MeSH terms found for TF-IDF vectorization.")
            return None
    
        # initializes the tf-idf vectorizer
        vectorizer = TfidfVectorizer()
    
        # applies tf-idf vectorizer to the mesh terms
        tfidf_matrix = vectorizer.fit_transform(valid_terms_df['mesh_term'])
    
        # sums the tf-idf scores for each term across all documents
        summed_tfidf = tfidf_matrix.sum(axis=0)
        # gets the terms (mesh descriptors) from the vectorizer
        terms = vectorizer.get_feature_names_out()
    
        # creates a dataframe with terms and their corresponding tf-idf scores
        ranked_terms_df = pd.DataFrame(summed_tfidf.T, index=terms, columns=["tfidf_score"])
        # sorts the terms by their tf-idf score in descending order
        ranked_terms_df = ranked_terms_df.sort_values(by="tfidf_score", ascending=False)
    
        # returns the ranked dataframe of terms and their tf-idf scores
        return ranked_terms_df

    def calculate_author_seniority(input_folder, output_folder):
        # initializes a list to store results
        author_seniority_list = []
        
        # iterates over all csv files in the input folder
        for author_file in os.listdir(input_folder):
            if author_file.endswith('.csv'):
                # reads the author file into a dataframe
                df = pd.read_csv(os.path.join(input_folder, author_file))
                # extracts the author name from the file name
                author_name = os.path.basename(author_file).replace('.csv', '')
                
                # calculates total publications for the author
                total_publications = len(df)
                # calculates total citations for the author
                total_citations = df['cited_by_count'].sum()
                
                # calculates the years active based on the publication dates
                df['created_date'] = pd.to_datetime(df['created_date'])
                # calculates the difference in years between the first and last publication
                years_active = (df['created_date'].max() - df['created_date'].min()).days // 365 if total_publications > 0 else 0
                
                # appends the calculated data to the results list
                author_seniority_list.append({
                    'Author': author_name,
                    'Total Publications': total_publications,
                    'Total Citations': total_citations,
                    'Years Active': years_active
                })
        
        # converts the results list into a dataframe
        author_seniority = pd.DataFrame(author_seniority_list)
        
        # ensures the output folder exists and saves the results as a csv file
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'author_seniority.csv')
        author_seniority.to_csv(output_path, index=False)
        print(f"Author seniority data saved to {output_path}")

    def classify_mental_health(input_folder, output_folder):
        # loads the sciroshot model and tokenizer from hugging face
        model_name = "BSC-LT/sciroshot"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # function to classify a single title
        def classify_title(title):
            # checks if the title is a valid string
            if not isinstance(title, str) or title.strip() == "":
                return -1  # returns -1 for empty or non-string titles
            # tokenizes the title and prepares the input for the model
            inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True, max_length=512)
            # uses the model to predict the class of the title
            with torch.no_grad():
                outputs = model(**inputs)
            # applies softmax to get probabilities and selects the predicted class
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities).item()
            return predicted_class
    
        # processes all publication files in the input folder
        for author_file in os.listdir(input_folder):
            if author_file.endswith('.csv'):  # change this if files have a different suffix
                # loads the author's publication file into a dataframe
                df = pd.read_csv(os.path.join(input_folder, author_file))
                # checks if the 'title' column exists
                if 'title' not in df.columns:
                    print(f"Error: 'title' column not found in {author_file}")
                    continue
    
                # applies the classification to each title in the dataframe
                df['mental_health_class'] = df['title'].apply(classify_title)
                
                # saves the results to the output folder
                author_name = os.path.basename(author_file).replace('.csv', '')
                output_path = os.path.join(output_folder, f"{author_name}_mental_health.csv")
                os.makedirs(output_folder, exist_ok=True)
                df.to_csv(output_path, index=False)
                print(f"Processed {author_file}, saved to {output_path}")

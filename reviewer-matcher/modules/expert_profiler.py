# necessary imports 
import requests
import pandas as pd
import time
import os
import re
import ast
from datetime import datetime
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ExpertProfiler:
    def __init__(self, config_manager):
        """
        Initialize the ExpertProfiler with a data path, API key, and base URL for OpenAlex.
        """
        self.api_key = config_manager.get('OPENALEX_API_KEY')
        self.base_url = config_manager.get('OPENALEX_BASE_URL')

        self.seniority_undetermined = config_manager.get('SENIORITY_UNDETERMINED')
        self.seniority_low = config_manager.get('SENIORITY_LOW')
        self.seniority_middle = config_manager.get('SENIORITY_MIDDLE')
        self.seniority_high = config_manager.get('SENIORITY_HIGH')

        self.num_pubs_top_perc = config_manager.get('NUM_PUBS_TOP_PERC_SENIORITY')
        self.num_cits_top_perc = config_manager.get('NUM_CITATIONS_TOP_PERC_SENIORITY')

    def extract_number(self, text):
        """
        Extract the first valid numeric value from a text.
        """
        # check if the input is already an integer
        if type(text) == int:
            return text
        # check for empty or NaN values
        if pd.isna(text) or text.strip() == '':
            return 0
        # check if the entire text is a URL, if so return 0
        if re.match(r'^(https?://)', text.strip()):
            return 0
        # extract all numbers, including those with commas
        numbers = re.findall(r'\b\d{1,10}(?:,\d{3})*|\d+\b', text)
        if numbers:
            # clean commas and convert all numbers to integers
            cleaned_numbers = [int(num.replace(',', '')) for num in numbers]
            # return the largest number found
            return max(cleaned_numbers)
        return 0

    def preprocess_data(self, df_experts):
        """
        Preprocess the experts' dataframe by converting relevant columns to numeric values.
        """
        # apply the extract_number method to the NUMBER_PUBLICATIONS column
        df_experts['NUMBER_PUBLICATIONS'] = df_experts['NUMBER_PUBLICATIONS'].apply(self.extract_number)
        # apply the extract_number method to the NUMBER_CITATIONS column
        df_experts['NUMBER_CITATIONS'] = df_experts['NUMBER_CITATIONS'].apply(self.extract_number)
        return df_experts

    def calculate_thresholds(self, df_experts):
        """
        Calculate thresholds for publications and citations based on percentile values.
        """
        # calculate the publication threshold based on the specified top percentile
        pub_threshold = np.percentile(df_experts['NUMBER_PUBLICATIONS'], self.num_pubs_top_perc)
        # calculate the citation threshold based on the specified top percentile
        cits_threshold = np.percentile(df_experts['NUMBER_CITATIONS'], self.num_cits_top_perc)
        # return the calculated thresholds as a tuple
        return pub_threshold, cits_threshold

    def determine_seniority(self, row, pub_threshold, cits_threshold):
        """
        Determine seniority based on publication count, citation count, and experience.
        """
        # if the expert has no publications and no citations
        if row['NUMBER_PUBLICATIONS'] == 0 and row['NUMBER_CITATIONS'] == 0:
            # check if the expert has any reviewer or panel experience
            if row['EXPERIENCE_REVIEWER'] == 'yes' or row['EXPERIENCE_PANEL'] == 'yes':
                # assign seniority as undetermined if experience exists but no publications or citations
                return self.seniority_undetermined
            else:
                # otherwise, assign seniority as low
                return self.seniority_low
        # if the expert has a high number of publications and reviewer experience
        elif row['NUMBER_PUBLICATIONS'] >= pub_threshold and row['EXPERIENCE_REVIEWER'] == 'yes':
            # assign seniority as high
            return self.seniority_high
        # if the expert has either high publications or reviewer experience
        elif row['NUMBER_PUBLICATIONS'] >= pub_threshold or row['EXPERIENCE_REVIEWER'] == 'yes':
            # check if the expert has panel experience or high citations
            if row['EXPERIENCE_PANEL'] == 'yes' or row['NUMBER_CITATIONS'] >= cits_threshold:
                # assign seniority as middle if additional experience exists
                return self.seniority_middle
            else:
                # otherwise, assign seniority as low
                return self.seniority_low
        # if the expert has both reviewer and panel experience
        elif row['EXPERIENCE_REVIEWER'] == 'yes' and row['EXPERIENCE_PANEL'] == 'yes':
            # assign seniority as middle
            return self.seniority_middle
        else:
            # for all other cases, assign seniority as low
            return self.seniority_low

    def query_openalex_by_name(self, full_name):
        """
        Query OpenAlex by author full name.
        """
        # constructs the url for the authors endpoint in the openalex api
        url = f"{self.base_url}/authors"
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

    def query_openalex_by_orcid(self, orcid):
        """
        Query OpenAlex by author's ORCID.
        """
        # constructs the url for the specific author based on their orcid
        url = f"{self.base_url}/authors/https://orcid.org/{orcid}"
        # sends a get request to the api with the author url
        response = requests.get(url)
        # checks if the request was successful
        if response.status_code == 200:
            # returns the json response
            return response.json()
        # returns none if there was no successful response
        return None
    
    def query_openalex_works(self, author_id):
        """
        Query OpenAlex works by author ID.
        """
        # constructs the url for the works endpoint in the openalex api
        url = f"{self.base_url}/works"
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
    
    def get_author_info(self, full_name=None, orcid=None):
        """
        Get author information by full name or ORCID or both.
        """
        # checks if an orcid is provided
        if orcid:
            # queries openalex by orcid and returns the result
            return self.query_openalex_by_orcid(orcid)
        # checks if a full name is provided if no orcid is given
        elif full_name:
            # queries openalex by full name and returns the result
            return self.query_openalex_by_name(full_name)
        # returns none if neither orcid nor full name is provided
        return None

    def extract_author_details(self, author_info):
        """
        Extract detailed author information from OpenAlex data.
        """
        # checks if author_info is none, indicating no data was returned
        if author_info is None:
            # returns a dictionary with empty or none values if no author information is available
            return {
                "name": None,
                "orcid": None,
                "topics_of_expertise": [],
                "approaches": [],
                "recent_work_titles": None,
                "works_count": None,
                "cited_by_count": None,
            }
    
        # extracts the author id from the author_info object
        author_id = author_info.get("id", "").split("/")[-1]
        # queries openalex for works by the author using the author id
        works_info = self.query_openalex_works(author_id)
    
        # initializes lists and counters for details about the author's works
        recent_work_titles = []
        work_types = []
        
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

            # extracts topics of expertise from topics associated with the author
            "topics_of_expertise": [topic.get("display_name") for topic in author_info.get("topics", [])] if author_info.get("topics") else [],
            
            # extracts approaches from concepts associated with the author
            "approaches": [concept.get("display_name") for concept in author_info.get("x_concepts", [])] if author_info.get("x_concepts") else [],

            # includes recent work titles in the details
            "recent_work_titles": recent_work_titles,

            "works_count": author_info.get("works_count"),
            "cited_by_count": author_info.get("cited_by_count"),
        }

        # returns the dictionary with the author's details
        return details

    def enrich_author_data(self, df, method="both"):
        """
        Enrich author data by querying OpenAlex API.
        """
        # initializes an empty list to store enriched author data
        enriched_data = []
        # iterates over each row in the dataframe
        for index, row in df.iterrows():
            # retrieves the full name and orcid from the current row
            full_name = row['Full Name']
            orcid = row['orcid']
            author_info = None
    
            # retrieves author information based on the specified method
            if method == "orcid" and orcid:
                # queries openalex using orcid if method is set to orcid
                author_info = self.get_author_info(orcid=orcid)
            elif method == "name" and full_name:
                # queries openalex using full name if method is set to name
                author_info = self.get_author_info(full_name=full_name)
            elif method == "both":
                # tries to retrieve author info by orcid first if method is set to both
                if orcid:
                    author_info = self.get_author_info(orcid=orcid)
                # if orcid query fails, attempts retrieval by full name
                if not author_info and full_name:
                    author_info = self.get_author_info(full_name=full_name)
    
            # extracts details about the author using the author_info
            author_details = self.extract_author_details(author_info)
    
            # appends the enriched author details to the list
            enriched_data.append(author_details)
    
        # converts the list of enriched author data into a dataframe and returns it
        return pd.DataFrame(enriched_data)
    
    def compute_completeness(self, data):
        """
        Compute completeness of the dataframe.
        """
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

    def compute_completeness_for_method(self, df, method):
        """
        Compute completeness after enriching author data.
        """
        # enriches the author data in the dataframe using the specified method
        enriched_df = self.enrich_author_data(df, method=method)
        # calculates the completeness percentage of the enriched dataframe
        completeness_percentage = self.compute_completeness(enriched_df)
        # returns the completeness percentage and the enriched dataframe
        return completeness_percentage, enriched_df
    
    def predict_gender_namsor(self, name, api_key):
        """
        Predict gender based on a name using the NamSor API.
        """
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
            headers = {'X-API-KEY': self.api_key}
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

    def enrich_data_with_predicted_gender(self, df, api_key):
        """
        Enrich the dataframe with predicted gender based on names using NamSor API.
        """
        # applies the predict_gender_namsor function to each name in the 'Full Name:' column
        df['gender'] = df['name'].apply(lambda name: self.predict_gender_namsor(name, api_key))

        # reorder columns to place 'name', 'orcid', and 'gender' as the first three columns
        columns = ['name', 'orcid', 'gender'] + [col for col in df.columns if col not in ['name', 'orcid', 'gender']]
        df = df[columns]

        # returns the dataframe with the new gender column
        return df
    
    def classify_recent_works_research_phase(self, df):
        """
        Classify research phases of recent work titles for each author.
        """
        # check if recent_work_titles column exists
        if 'recent_work_titles' not in df.columns:
            print("Column 'recent_work_titles' not found in the input data.")
            return df

        # define a helper function to classify and consolidate research phases for recent works
        def get_unique_research_phases(titles):
            # check if titles is a list
            if not isinstance(titles, list):
                return 'Unknown'

            # classify each title's research phase and store unique values
            research_phases = set()
            for title in titles:
                phase = self.classify_research_phase(title)
                research_phases.add(phase)

            # join unique phases with a semicolon
            return ';'.join(research_phases)

        # apply the helper function to each row to create the 'research_phase' column
        df['research_phase'] = df['recent_work_titles'].apply(get_unique_research_phases)

        # create dummy columns for each unique research phase
        research_phase_dummies = df['research_phase'].str.get_dummies(sep=';')

        # concatenate the dummy columns to the df
        df = pd.concat([df, research_phase_dummies], axis=1)

        # reorder columns: make 'research_phase' the fourth column, followed by the dummy columns
        columns_order = (
            ['name', 'orcid', 'gender', 'research_phase'] +
            research_phase_dummies.columns.tolist() +
            [col for col in df.columns if col not in ['name', 'orcid', 'gender', 'research_phase'] + research_phase_dummies.columns.tolist()]
        )
        df = df[columns_order]

        # return the modified df
        return df

    def add_topics_and_approaches_dummies(self, df):
        """
        Create dummy columns for 'topics_of_expertise' and 'approaches'.
        """
        # check if 'topics_of_expertise' and 'approaches' columns exist
        if 'topics_of_expertise' not in df.columns or 'approaches' not in df.columns:
            print("columns 'topics_of_expertise' or 'approaches' not found in the input data.")
            return df
    
        # helper function to create dummies from a list
        def create_dummies_from_list(column_data):
            if isinstance(column_data, list):
                return ';'.join(column_data)
            return ''  # return empty string if not a list
    
        # apply the helper function
        df['topics_of_expertise'] = df['topics_of_expertise'].apply(create_dummies_from_list)
        df['approaches'] = df['approaches'].apply(create_dummies_from_list)
    
        # create dummy columns
        topics_dummies = df['topics_of_expertise'].str.get_dummies(sep=';')
        approaches_dummies = df['approaches'].str.get_dummies(sep=';')
    
        # create the desired column order
        fixed_columns = [
            'name', 'orcid', 'gender', 'research_phase', 
            'Basic Research', 'Clinical Research', 'Mechanisms of Disease', 
            'Public Health', 'Translational Research',  # these remain unchanged
            'topics_of_expertise'  # topics_of_expertise column
        ]
    
        # concatenate the topic dummy columns and approaches dummy columns
        topics_approaches_df = pd.concat([df[fixed_columns], topics_dummies, df[['approaches']], approaches_dummies], axis=1)
    
        # finally, add 'recent_work_titles', 'works_count' and 'cited_by_count'
        topics_approaches_df = pd.concat([topics_approaches_df, df[['recent_work_titles', 'works_count', 'cited_by_count']]], axis=1)

        return topics_approaches_df

    def get_all_publications(self, author_id):
        """
        Retrieve all publications for an author from the OpenAlex API, handling pagination.
        """
        # constructs the url for the works endpoint in the openalex api
        url = f"{self.base_url}/works"
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

        # sort the works by publication_date (most recent first)
        all_works.sort(key=lambda x: datetime.strptime(x['publication_date'], '%Y-%m-%d'), reverse=True)

        # returns the list of all works retrieved
        return all_works
    
    def get_author_publications(self, full_name=None, orcid=None):
        """
        Retrieve an author's publications based on ORCID and if not successful, full name.
        """
        # initializes author_info as none
        author_info = None
        # checks if an orcid is provided
        if orcid:
            # queries openalex by orcid
            author_info = self.query_openalex_by_orcid(orcid)
        # checks if a full name is provided if no orcid is given
        elif full_name:
            # queries openalex by full name
            author_info = self.query_openalex_by_name(full_name)
        
        # returns none and an empty list if no author information was found
        if author_info is None:
            return None, []
    
        # extracts the author id from the author_info
        author_id = author_info.get("id", "").split("/")[-1]
        # retrieves all publications for the author using the author id
        publications = self.get_all_publications(author_id)
        
        # returns the author's display name (or the full name if display name is not available) and the publications list
        return author_info.get("display_name", full_name), publications
    
    def save_publications_to_csv(self, df, save_path):
        """
        Save each author's publications to a CSV file.
        """
        for index, row in df.iterrows():
            full_name = row.get('name')
            orcid = row.get('orcid')
            author_name, publications = self.get_author_publications(full_name=full_name, orcid=orcid)

            if author_name and publications:
                # construct file path for each reviewer
                filename = f"{author_name.replace(' ', '_').lower()}_publications.csv"
                file_path = os.path.join(save_path, filename)

                # save publications directly to csv
                publications_df = pd.DataFrame(publications)
                publications_df['name'] = author_name

                # reorder columns to make 'name' the first column
                columns = ['name'] + [col for col in publications_df.columns if col != 'name']
                publications_df = publications_df[columns]

                publications_df.to_csv(file_path, index=False)

                #print(f"Publications for {author_name} saved to {file_path}.")
            else:
                print(f"No publications found for {full_name}.")

    def calculate_average_publications_per_author(self, publications_folder):
        """
        Calculate the average number of publications per author based on saved CSV files.
        """
        # initialize an empty list to store publication counts for each author
        publications_counts = []

        # iterate over each file in the publications folder
        for file in os.listdir(publications_folder):
            # check if the file ends with '.csv'
            if file.endswith('_publications.csv'):
                # read the csv file into a dataframe
                df = pd.read_csv(os.path.join(publications_folder, file))
                # count the number of publications (rows) in the dataframe
                num_publications = len(df)
                # add the publication count to the list
                publications_counts.append(num_publications)

        # calculate the average number of publications if the list is not empty, otherwise set to 0
        average_publications = sum(publications_counts) / len(publications_counts) if publications_counts else 0
        return average_publications

    def combine_data(self, input_folder):
        """
        Combine all CSV files in a folder into a single dataframe.
        """
        # initializes an empty list to store dataframes
        data_frames = []
        
        # iterates over each file in the output folder
        for file in os.listdir(input_folder):
            # checks if the file ends with the correct extension 
            if file.endswith('.csv'):
                # reads the CSV file into a dataframe
                df = pd.read_csv(os.path.join(input_folder, file))
                # appends the dataframe to the list
                data_frames.append(df)
        
        # combines all the dataframes in the list into a single dataframe
        combined_df = pd.concat(data_frames, ignore_index=True)
        # returns the combined dataframe
        return combined_df

    def classify_research_phase(self, text):
        """
        Classify the research phase of a given text using a pretrained model.
        """
        research_phase_classifier = pipeline("text-classification", model="SIRIS-Lab/batracio5")
        # check if text is a non-empty string
        if isinstance(text, str) and text.strip():
            # use the research phase classifier to get the label of the text
            return research_phase_classifier(text)[0]['label']
        return 'Unknown'  # default label for invalid input
    
    def classify_publications_by_research_phase(self, author_file, output_folder):
        """
        Classify the research phase for publications in a CSV file and save the results.
        """
        # reads the author file into a dataframe
        df = pd.read_csv(author_file)
        
        # apply research phase classification
        df['research_phase'] = df['title'].apply(self.classify_research_phase)
        
        # save to output folder with updated filename
        author_name = os.path.basename(author_file).replace('_publications.csv', '')
        output_path = os.path.join(output_folder, f"{author_name}_research_phase.csv")
        df.to_csv(output_path, index=False)
        #print(f"Research phase classified for {author_file}, saved to {output_path}")

    def classify_all_publications_by_research_phase(self, input_folder, output_folder):
        """
        Classify research phase for all publication files in a folder.
        """
        # iterates over each file in the input folder and classifies by research phase
        for author_file in os.listdir(input_folder):
            if author_file.endswith('_publications.csv'):
                full_path = os.path.join(input_folder, author_file)
                self.classify_publications_by_research_phase(full_path, output_folder)
    
    def classify_domain(self, text):
        """
        Classify the domain of a given text using a pretrained model.
        """
        domain_classifier = pipeline("text-classification", model="SIRIS-Lab/biomedicine-classifier")
        # check if text is a non-empty string
        if isinstance(text, str) and text.strip():
            # use the domain classifier to get the label of the text
            return domain_classifier(text)[0]['label']
        return 'Unknown'  # default label for invalid input

    def classify_publications_by_domain(self, author_file, output_folder):
        """
        Classify the domain for publications in a CSV file and save the results.
        """
        # reads the author file into a dataframe
        df = pd.read_csv(author_file)
        
        # apply domain classification
        df['domain'] = df['title'].apply(self.classify_domain)
        
        # save to output folder with updated filename
        author_name = os.path.basename(author_file).replace('_research_phase.csv', '')
        output_path = os.path.join(output_folder, f"{author_name}_domain.csv")
        df.to_csv(output_path, index=False)
        #print(f"Domain classified for {author_file}, saved to {output_path}")

    def classify_all_publications_by_domain(self, input_folder, output_folder):
        """
        Classify domain for all publication files in a folder.
        """
        # iterates over each file in the input folder and classifies by domain
        for author_file in os.listdir(input_folder):
            if author_file.endswith('_research_phase.csv'):
                full_path = os.path.join(input_folder, author_file)
                self.classify_publications_by_domain(full_path, output_folder)

    def classify_mental_health(self, text):
        """
        Classify whether a publication is related to mental health using a pretrained model.
        """
        # loads the sciroshot model and tokenizer from hugging face
        model_name = "BSC-LT/sciroshot"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # checks if the title is a valid string
        if not isinstance(text, str) or text.strip() == "":
            return -1  # returns -1 for empty or non-string titles

        # tokenizes the title and prepares the input for the model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # uses the model to predict the class of the title
        with torch.no_grad():
            outputs = model(**inputs)
        
        # applies softmax to get probabilities and selects the predicted class
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()
        
        return predicted_class

    def classify_all_publications_by_mental_health(self, input_folder, output_folder):
        """
        Classify mental health-related publications for all files in a folder.
        """
        # iterates over each file in the input folder
        for author_file in os.listdir(input_folder):
            if author_file.endswith('_domain.csv'):
                # loads the author's publication file into a dataframe
                df = pd.read_csv(os.path.join(input_folder, author_file))

                # checks if the 'title' column exists
                if 'title' not in df.columns:
                    print(f"Error: 'title' column not found in {author_file}")
                    continue

                # applies the mental health classification to each title in the dataframe
                df['mental_health'] = df['title'].apply(self.classify_mental_health)

                # saves the results to the output folder
                author_name = os.path.basename(author_file).replace('_domain.csv', '')
                output_path = os.path.join(output_folder, f"{author_name}_mental_health.csv")
                os.makedirs(output_folder, exist_ok=True)
                df.to_csv(output_path, index=False)
                #print(f"Determined mental health for {author_file}, saved to {output_path}")

    def extract_mesh_terms_string(self, mesh_terms):
        """
        Extract and concatenate descriptor names from a string of MeSH terms.
        """
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

    def rank_mesh_terms_across_all(self, output_folder):
        """
        Rank MeSH terms based on their TF-IDF scores across all documents.
        """
        # loads the combined data from the output folder
        combined_df = self.combine_data(output_folder)
        
        # extracts mesh terms as strings for tf-idf processing
        combined_df['mesh_term'] = combined_df['mesh'].apply(lambda terms: self.extract_mesh_terms_string(terms))
        
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

    def calculate_reviewer_seniority(self, input_folder):
        """
        Calculate the seniority of reviewers based on their publications and citations.
        """
        # initializes a list to store results
        author_seniority_list = []
        
        # iterates over all csv files in the input folder
        for author_file in os.listdir(input_folder):
            if author_file.endswith('_mental_health.csv'):
                # reads the author file into a dataframe
                df = pd.read_csv(os.path.join(input_folder, author_file))
                # extracts the author name from the file name
                author_name = os.path.basename(author_file).replace('_mental_health.csv', '')
                
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
        return author_seniority
    
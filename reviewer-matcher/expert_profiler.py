class ExpertProfiler:
    BASE_URL = "https://api.openalex.org"
    # initialize classifiers once
    research_phase_classifier = pipeline("text-classification", model="SIRIS-Lab/batracio5")
    domain_classifier = pipeline("text-classification", model="SIRIS-Lab/biomedicine-classifier")

    def query_openalex_by_name(full_name):
        url = f"{BASE_URL}/authors"
        params = {"search": full_name}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                return data['results'][0]  # return the first result
        return None

    def query_openalex_by_orcid(orcid):
        url = f"{BASE_URL}/authors/https://orcid.org/{orcid}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    
    def query_openalex_works(author_id):
        url = f"{BASE_URL}/works"
        params = {"filter": f"authorships.author.id:{author_id}", "sort": "publication_date:desc", "per-page": 100}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return None
    
    def get_author_info(full_name=None, orcid=None):
        if orcid:
            return query_openalex_by_orcid(orcid)
        elif full_name:
            return query_openalex_by_name(full_name)
        return None

    def extract_author_details(author_info):
        if author_info is None:
            return {
                "name": None,
                "orcid": None,
                "works_count": None,
                "cited_by_count": None,
                "recent_work_titles": [],
                "work_types": [],
                "areas_of_interest": []
            }
    
        author_id = author_info.get("id", "").split("/")[-1]
        works_info = query_openalex_works(author_id)
    
        recent_work_titles = []
        open_access_count = 0
        work_types = []
        institutions = []
        
        if works_info and "results" in works_info:
            recent_work_titles = [work.get("title") for work in works_info["results"]]
            work_types = [work.get("type") for work in works_info["results"]]
    
        details = {
            "name": author_info.get("display_name"),
            "orcid": author_info.get("orcid"),
            "works_count": author_info.get("works_count"),
            "cited_by_count": author_info.get("cited_by_count"),
            "recent_work_titles": recent_work_titles,
            "work_types": work_types,
            "areas_of_interest": [concept.get("display_name") for concept in author_info.get("x_concepts", [])] 
            if author_info.get("x_concepts") else [],
        }
        
        return details

    def compute_completeness(data):
        non_null_counts = data.notnull().sum()
        total_counts = len(data)
        completeness_percentages = (non_null_counts / total_counts) * 100
        overall_completeness = completeness_percentages.mean()
        return overall_completeness

    def enrich_author_data(df, method="name"):
        enriched_data = []
        for index, row in df.iterrows():
            full_name = row['Full Name:']
            orcid = row['ORCID Number:']
            author_info = None
    
            if method == "orcid" and orcid:
                author_info = get_author_info(orcid=orcid)
            elif method == "name" and full_name:
                author_info = get_author_info(full_name=full_name)
            elif method == "both":
                if orcid:
                    author_info = get_author_info(orcid=orcid)
                if not author_info and full_name:
                    author_info = get_author_info(full_name=full_name)
    
            author_details = extract_author_details(author_info)
    
            # ensure that author_details is always a dictionary with consistent keys
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
    
            enriched_data.append(author_details)
    
        return pd.DataFrame(enriched_data)

    def compute_completeness_for_method(df, method):
        enriched_df = enrich_author_data(df, method=method)
        completeness_percentage = compute_completeness(enriched_df)
        return completeness_percentage, enriched_df

    def get_all_publications(author_id):
        url = f"{BASE_URL}/works"
        params = {
            "filter": f"authorships.author.id:{author_id}",
            "per-page": 200
        }
        all_works = []
        while True:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                all_works.extend(data['results'])
                if 'meta' in data and 'next_cursor' in data['meta'] and data['meta']['next_cursor']:
                    params['cursor'] = data['meta']['next_cursor']
                else:
                    break
            else:
                break
            time.sleep(1)  
        return all_works
    
    def get_author_publications(full_name=None, orcid=None):
        author_info = None
        if orcid:
            author_info = query_openalex_by_orcid(orcid)
        elif full_name:
            author_info = query_openalex_by_name(full_name)
        
        if author_info is None:
            return None, []
    
        author_id = author_info.get("id", "").split("/")[-1]
        publications = get_all_publications(author_id)
        
        return author_info.get("display_name", full_name), publications
    
    def save_publications_to_csv(author_name, publications, save_path):
        df = pd.DataFrame(publications)
        df['author_name'] = author_name
        
        # ensure the save_path directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df.to_csv(save_path, index=False)

    def predict_gender_namsor(name, api_key):
    try:
        # remove titles and honorifics using regular expressions
        titles = ['Dr', 'Prof', 'Prof. ', 'Professor', 'Mr', 'Ms', 'Mrs', 'Miss']
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(title) for title in titles) + r')\.?\b', re.IGNORECASE)
        name = pattern.sub('', name).strip()
        
        first_name, last_name = name.split()[0], name.split()[-1] 
        url = f"https://v2.namsor.com/NamSorAPIv2/api2/json/gender/{first_name}/{last_name}"
        headers = {'X-API-KEY': api_key}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('likelyGender', 'unknown')
        else:
            return 'unknown'
    except Exception as e:
        print(f"Error processing name '{name}': {e}")
        return 'unknown'

    # enrich data with predicted gender
    def enrich_data_with_predicted_gender(df, api_key):
        df['Predicted Gender:'] = df['Full Name:'].apply(lambda name: predict_gender_namsor(name, api_key))
        return df
    
    # save enriched data
    def save_enriched_data(df, folder_path):
        file_path = os.path.join(folder_path, '03_gender_predictions', 'authors_with_gender.csv')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Enriched data saved to {file_path}.")
    
    # calculate metrics manually
    def calculate_metrics(y_true, y_pred):
        results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    
        accuracy = (results['Actual'] == results['Predicted']).mean()
    
        tp = ((results['Predicted'] == 'male') & (results['Actual'] == 'male')).sum()
        fp = ((results['Predicted'] == 'male') & (results['Actual'] == 'female')).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
        fn = ((results['Predicted'] == 'female') & (results['Actual'] == 'male')).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        return accuracy, precision, recall, f1

    # classify the research phase
    def classify_research_phase(text):
        if isinstance(text, str) and text.strip():  # check if text is a non-empty string
            return research_phase_classifier(text)[0]['label']
        return 'Unknown'  # default label for invalid input
    
    # classify the domain
    def classify_domain(text):
        if isinstance(text, str) and text.strip():  # check if text is a non-empty string
            return domain_classifier(text)[0]['label']
        return 'Unknown'  # default label for invalid input
    
    # process publications for a single author file
    def process_publications(author_file, output_folder):
        df = pd.read_csv(author_file)
    
        # apply classification in batches to speed up processing
        df['research_phase'] = df['title'].apply(classify_research_phase)
        df['domain'] = df['title'].apply(classify_domain)
    
        author_name = os.path.basename(author_file).replace('.csv', '')
        output_path = os.path.join(output_folder, f"{author_name}_classified.csv")
        os.makedirs(output_folder, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed {author_file}, saved to {output_path}")
    
    # process all publications in the input folder
    def process_all_publications(input_folder, output_folder):
        for author_file in os.listdir(input_folder):
            if author_file.endswith('.csv'):
                process_publications(os.path.join(input_folder, author_file), output_folder)
    
    # compute descriptive statistics
    def compute_statistics(output_folder):
        all_data = pd.DataFrame()
        
        for classified_file in os.listdir(output_folder):
            if classified_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(output_folder, classified_file))
                all_data = pd.concat([all_data, df], ignore_index=True)
        
        research_phase_counts = all_data['research_phase'].value_counts(normalize=True) * 100
        domain_counts = all_data['domain'].value_counts(normalize=True) * 100
        
        return research_phase_counts, domain_counts

    # load and combine all csv files into a single df
    def load_data(output_folder):
        data_frames = []
        for file in os.listdir(output_folder):
            if file.endswith('_classified.csv'):
                df = pd.read_csv(os.path.join(output_folder, file))
                data_frames.append(df)
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    
    # extract mesh terms as strings from the df
    def extract_mesh_terms_string(mesh_terms):
        if isinstance(mesh_terms, str) and mesh_terms.strip():  # check if a non-empty string
            try:
                # convert the string representation of the list to an actual list of dictionaries
                terms = ast.literal_eval(mesh_terms)
                if isinstance(terms, list) and terms:  # check if a non-empty list
                    # concatenate descriptor names into a single string
                    terms_list = [term['descriptor_name'] for term in terms if 'descriptor_name' in term]
                    return ' '.join(terms_list)
                else:
                    return ''
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing MeSH terms: {e}")
                return ''
        else:
            return ''  # return empty if input is not a valid string
    
    # calculate tf-idf scores and rank mesh terms
    def rank_mesh_terms_across_all(output_folder):
        combined_df = load_data(output_folder)
        
        # extract mesh terms as strings for tf-idf processing
        combined_df['mesh_term'] = combined_df['mesh'].apply(lambda terms: extract_mesh_terms_string(terms))
        
        # filter out rows with empty mesh terms string for tf-idf
        valid_terms_df = combined_df[combined_df['mesh_term'].str.strip() != '']
        
        # check if there are any non-empty mesh terms strings
        if valid_terms_df.empty:
            print("Error: No valid MeSH terms found for TF-IDF vectorization.")
            return None
    
        # initialize the tf-idf vectorizer
        vectorizer = TfidfVectorizer()
    
        # apply tf-idf vectorizer
        tfidf_matrix = vectorizer.fit_transform(valid_terms_df['mesh_term'])
    
        # sum the tf-idf scores for each term
        summed_tfidf = tfidf_matrix.sum(axis=0)
        terms = vectorizer.get_feature_names_out()
    
        # create a df with terms and their corresponding tf-idf scores
        ranked_terms_df = pd.DataFrame(summed_tfidf.T, index=terms, columns=["tfidf_score"])
        ranked_terms_df = ranked_terms_df.sort_values(by="tfidf_score", ascending=False)
    
        return ranked_terms_df

    def calculate_author_seniority(input_folder, output_folder):
        # initialize a list to store results
        author_seniority_list = []
        
        for author_file in os.listdir(input_folder):
            if author_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(input_folder, author_file))
                author_name = os.path.basename(author_file).replace('.csv', '')
                
                # calculate total publications and total citations
                total_publications = len(df)
                total_citations = df['cited_by_count'].sum()
                
                # calculate years active based on publication dates
                df['created_date'] = pd.to_datetime(df['created_date'])
                years_active = (df['created_date'].max() - df['created_date'].min()).days // 365 if total_publications > 0 else 0
                
                # append results to the list
                author_seniority_list.append({
                    'Author': author_name,
                    'Total Publications': total_publications,
                    'Total Citations': total_citations,
                    'Years Active': years_active
                })
        
        # convert the list of results into a df
        author_seniority = pd.DataFrame(author_seniority_list)
        
        # save results to the output folder
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'author_seniority.csv')
        author_seniority.to_csv(output_path, index=False)
        print(f"Author seniority data saved to {output_path}")

    def classify_publications(input_folder, output_folder):
        # load the sciroshot model and tokenizer from hugging face
        model_name = "BSC-LT/sciroshot"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # function to classify a single title
        def classify_title(title):
            if not isinstance(title, str) or title.strip() == "":
                return -1  # return -1 for empty or non-string titles
            inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities).item()
            return predicted_class
    
        # process all publication files in the input folder
        for author_file in os.listdir(input_folder):
            if author_file.endswith('.csv'):  # change this if your files have a different suffix
                df = pd.read_csv(os.path.join(input_folder, author_file))
                if 'title' not in df.columns:
                    print(f"Error: 'title' column not found in {author_file}")
                    continue
    
                # classify each title
                df['mental_health_class'] = df['title'].apply(classify_title)
                
                # save the results
                author_name = os.path.basename(author_file).replace('.csv', '')
                output_path = os.path.join(output_folder, f"{author_name}_mental_health.csv")
                os.makedirs(output_folder, exist_ok=True)
                df.to_csv(output_path, index=False)
                print(f"Processed {author_file}, saved to {output_path}")

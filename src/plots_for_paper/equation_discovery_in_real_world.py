import time

import numpy as np
import requests
import json

from definitions import ROOT_DIR


api_key = "your api key goes here"  # Replace with the actual API key

queries = ["Equation Discovery", "Symbolic Regression"] #"DSR", "PySR", "Operon"
queries = [
    "3ee45877f7f14c8ee4872a7249f74eef7dc2255f", # Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients
    "10.48550/arXiv.2305.01582",  # Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl
    "10.1145/3377929.3398099", # Operon C++: an efficient genetic programming framework for symbolic regression
]
paper_ids = []
url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
fild_dict = {}
for query in queries:
    # query_params = {
    #     "query": query,
    #     "fields": "title,fieldsOfStudy,citationCount,abstract", # authors,abstract,year,
    # }
    # headers = {"x-api-key": api_key}
    url = f"https://api.semanticscholar.org/graph/v1/paper/{query}/citations?fields=title,citationCount,fieldsOfStudy&limit=500&offset=0"
    response = requests.get(url,
                            #params=query_params,
                            #headers=headers
                            ).json()
    paper_title=[]
    citation_count = []
    if len(response['data'])>0:
        for paper in response['data']:
            citing_paper = paper['citingPaper']
            title = citing_paper['title']
            fieldsOfStudy = citing_paper['fieldsOfStudy']
            if not fieldsOfStudy:
                fieldsOfStudy = ['Not given']
            # if fieldsOfStudy == ['Computer Science']:
            #     continue
            if 'citationCount' in citing_paper:
                paper_title.append(title)
                citation_count.append(citing_paper['citationCount'])
                for field in fieldsOfStudy:
                    if field not in fild_dict:
                        fild_dict[field] = set()
                        fild_dict[field].add(title)
                    else:
                        fild_dict[field].add(title)

    time.sleep(1)
sort_index = np.argsort(np.array(citation_count))
citation_count = np.array(citation_count)[sort_index]
paper_title = np.array(paper_title)[sort_index]

for i in range(1,50,1):
    print(f"{paper_title[-i]:<50}: {citation_count[-i]}")

for key, value in fild_dict.items():
    fild_dict[key] = list(value)

save_path = ROOT_DIR / "results/SematicScholar/paper_data.json"
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(fild_dict, f, indent=2)
print("Results saved to {}".format(save_path))


## TREC2020

# Setup
We use elastisearch version 7.17.6 as a database.

[Elastisearch v7.17.6 download](https://www.elastic.co/downloads/past-releases/elasticsearch-7-17-6)

The data is two collections:

MS MARCO Passage Ranking collection [(Direct download)](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz)

and 

TREC CAR paragraph collection v2.0 [(Direct download)](http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz)


Be sure to install the python requirements.
```
pip install -r requirements.txt
```

# Populate database
To populate the database run either "setup_es.py" or "setup_es_multi.py".

If you want to use the multi threaded file (setup_es_multi.py) an extra requirement of the package "ray" is required
```
pip install ray
```
or
```
pip install "ray[default]"
```
for access to a ray dashboard



# Create and evaluate results
## Get results
The results are gathered by running the file "manual_baseline.py".
This will run queries from "2020_manual_evaluation_topics_v1.0.json" and create outputs in the file "manual_results.txt" under the folder results.

## Evaluate
Run the file "evaluate.py", this will print out the results to console.

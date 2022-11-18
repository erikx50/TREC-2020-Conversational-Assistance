# DAT640 Project - TREC 2020 Conversational Assistance

# Setup
We use elastisearch version 7.17.6 as a database.

[Elastisearch v7.17.6 download](https://www.elastic.co/downloads/past-releases/elasticsearch-7-17-6)

The data is two collections:

MS MARCO Passage Ranking collection [(Direct download)](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz)

and 

TREC CAR paragraph collection v2.0 [(Direct download)](http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz)

Extract these files and place them in the data folder in place of the dummy files.
They need the names
```
MS Macro collection.tsv
```
and
```
dedup.articles-paragraphs.cbor
```
for the program to run as is.



Be sure to install the python requirements.
```
pip install -r requirements.txt
```

# Populate database
To populate the database run either "setup_es.py" or "setup_es_multi.py".
```
python setup_es.py
```

If you want to use the multi threaded file (setup_es_multi.py).
This will use all cores on your CPU.
You can modify the script to limit the amount of cores to less than all, say 4/8 cores, if you want to use the computer for other things at the same time.
```
python setup_es_multi.py
```

Ray is part of requirements, but you can additionally install with:
```
pip install "ray[default]"
```
for access to a ray dashboard, optional.



# Create and evaluate results
## Baseline
The baseline results are gathered by running the file "bm25baseline.py".
```
python bm25baseline.py
```
This will run queries from two files under 2020 folder:
```
2020_manual_evaluation_topics_v1.0.json
2020_automatic_evaluation_topics_v1.0.json
```

The results will be written to the files:
```
bm25_manual_results.txt
bm25_automatic_results.txt
```


## BERT and T5
To run the BERT ranker run:
```
python bert_reranker.py
```
This will create four result files under "result" folder:
```
BERT_manual_results.txt
BERT_automatic_results.txt
BERT_reranker_manual_results.txt
BERT_reranker_automatic_results.txt
```
These will be used later by the python file "score.py" to print out information used to evaluate the ranker.

Similalry for T5 run the file mentioned below, then answer wheter you want to run the manual  automatic querys, then select how many documents are to be reranked per query. If unsure use 500.
```
python reranker_T5.py
```


This will create one of two result files under the "result" folder
```
T5_manual_results.txt
T5_automatic_results.txt
```
Depending on the option you select when running the script.
Same as BERT these result files are used when running "score.py"

## Score
Run the file "score.py", this will print out the results to console.
```
python score.py
```
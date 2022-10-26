import imp
from pydoc import importfile

from trec_car import read_data


class Cars:
    def __init__(self):
        self.filePath = "data\paragraphCorpus\dedup.articles-paragraphs.cbor"
        #self.file = open("data\paragraphCorpus\dedup.articles-paragraphs.cbor", "rb")
    

    def next() -> dict:
      # TODO:
      return {}




if __name__ == "__main__":
    cd = Cars() 
    for par in read_data.iter_paragraphs(open("data\paragraphCorpus\dedup.articles-paragraphs.cbor", "rb")):
        totText = ""
        for p in par.bodies:
            
            totText += p.get_text() # text, can be strung togherther to get full body
            # Process this and store in DB
        #TODO: preprocess
        
        #TODO: add to db
        print(totText)

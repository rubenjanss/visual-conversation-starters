import json
import numpy as np
import pandas as pd
import re

def extract_captions(results):
    # Load results from JSON
    captions = results["captions"]
    scores = np.array(results["scores"])
    boxes = np.array(results["boxes"])
    
    # Load results into Pandas DataFrame
    captions_df = pd.DataFrame()
    captions_df["captions"] = captions
    captions_df["scores"] = scores
    captions_df["boxes"] = boxes.tolist()
    
    # Select only captions with score of >= 0.7
    captions_df = captions_df[captions_df["scores"] >= 0.7]
    
    # Drop any discarded captions
    with open("discarded_captions.csv") as f:
        discarded_captions = pd.read_csv(f, names=["caption"])    
        
    captions_df = captions_df[~captions_df["captions"].isin(discarded_captions)]
    
    # Only retain unique captions
    captions_df = captions_df.drop_duplicates('captions')
    
    # Substitute <unk> token
    captions_df["captions"] = captions_df.apply(lambda x: re.sub("the word <unk>", "a word", x["captions"]), axis=1)
    captions_df["captions"] = captions_df.apply(lambda x: re.sub("the number <unk>", "a number", x["captions"]), axis=1)
    captions_df["captions"] = captions_df.apply(lambda x: re.sub("<unk>", "something", x["captions"]), axis=1)
    
    # Join captions together in a single string
    return '. '.join(captions_df["captions"])


def extract_captions_from_file(path):
    with open(path) as f:
        results = json.load(f)
        return extract_captions(results)


def extract_captions_from_string(text):
    results = json.loads(text)
    return extract_captions(results)
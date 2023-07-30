import glob, re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_file_list(file_path="dataset/", format=".dat"):
    file_list=glob.glob(file_path+"*"+format)
    res=pd.DataFrame([], columns=['doc_id', 'terms'])
    for f in file_list:
        with open(f, 'r') as file:
            content=file.read()
            regex=re.findall(r'\.I ([0-9]*)\n\.W\n(.*)', content)
            res=pd.concat([res, pd.DataFrame(regex, columns=['doc_id', 'terms'])], ignore_index=True, axis=0)
    return res


def get_tfidf_repr(df: pd.DataFrame):
    doc_tfidf=TfidfVectorizer(lowercase=False, dtype=np.float32)

    #Computation of the sparse embedding
    sparse_doc=doc_tfidf.fit_transform(df["terms"])
    return sparse_doc
import argparse
from math import trunc 
import random
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Build Subset of IR Dataset in the form of <qid, docid_a, docid_b>')

parser.add_argument('-source', type=str, default=None, help='Dataset Source')
parser.add_argument('-portion', type=int, default=None, help=r'% of dataset to use')
parser.add_argument('-suffix', type=str, default='train', help='Suffix of output name')
parser.add_argument('-out', type=str, help='Output dir for tsv')

def main(args):
    cols = ['qid', 'docid_a', 'doc_id_b']
    types = {col : np.int32 for col in cols}
    with open(args.source, 'r') as f:
        df = pd.read_csv(f, sep='\t', header=None, index=False, names=cols, dtype=types)

    num_choices = trunc(len(df) * (args.portion/100))
    sub_df = df.sample(n=num_choices)

    sub_df.to_csv(args.out + f'triples.{args.suffix}.tsv', sep='\t', index=False, header=False)


if __name__ -- '__main__':
    main(parser.parse_args())
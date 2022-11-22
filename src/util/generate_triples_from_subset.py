import argparse
import pandas as pd
import numpy as np
import ir_datasets

parser = argparse.ArgumentParser(description='Build Subset of IR Dataset in the form of <qid, docid_a, docid_b>')

parser.add_argument('-source', type=str, default=None, help='Dataset Source')
parser.add_argument('-subset', type=str, default=None, help='Query file containing ids')
parser.add_argument('-suffix', type=str, default='train', help='Suffix of output name')
parser.add_argument('-out', type=str, help='Output dir for tsv')

def main(args):
    cols = ['qid', 'docid_a', 'doc_id_b']
    types = {col : np.int32 for col in cols}
    with open(args.source, 'r') as f:
        df = pd.read_csv(f, sep='\t', header=None, index=False, names=cols, dtype=types)
    dataset = ir_datasets.load(args.subset)
    filter = df['qid'].isin([query.qid for query in dataset.queries_iter()])

    filtered_df = df[filter]
    filtered_df.to_csv(args.out + f'triples.{args.suffix}.tsv', sep='\t', index=False, header=False)

if __name__ == '__main__':
    main(parser.parse_args())
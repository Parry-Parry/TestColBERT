import argparse
from math import trunc 
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Build Subset of IR Dataset in the form of <qid, docid_a, docid_b>')

parser.add_argument('-source', type=str, default=None, help='Dataset Source')
parser.add_argument('-type', type=str, default=None, help='Source Type')
parser.add_argument('-portion', type=int, default=None, nargs='+', help=r'% of dataset to use')
parser.add_argument('-suffix', type=str, default='train', help='Suffix of output name')
parser.add_argument('-out', type=str, help='Output dir for tsv')
parser.add_argument('-k', type=int, default=None, help='Exact Number of Samples')


def main(args):
    if args.type == 'tsv':
        cols = ['qid', 'pid+', 'pid-']
        types = {col : str for col in cols}
        with open(args.source, 'r') as f:
            df = pd.read_csv(f, sep='\t', header=None, index_col=False, names=cols, dtype=types)
    else:
        assert args.type == 'jsonl'
        with open(args.source, 'r') as f:
            df = pd.read_json(f, lines=True, orient='records')

    for portion in args.portion:
        if not args.k:
            num_choices = trunc(len(df) * (portion/100))
        else:
            num_choices = args.k
        sub_df = df.sample(n=num_choices)

        sub_df.to_csv(args.out + f'triples.{args.suffix}.tsv', index=False, header='False')


if __name__ == '__main__':
    main(parser.parse_args())
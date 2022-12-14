import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='TSV2JSONL')

parser.add_argument('-source', type=str, default=None, help='Dataset Source')
parser.add_argument('-out', type=str, help='Output dir for JSONL')

def main(args):
    cols = ['qid', 'docid_a', 'doc_id_b']
    types = {col : np.int32 for col in cols}
    with open(args.source, 'r') as f:
        df = pd.read_csv(f, sep='\t', header=None, index_col=False, names=cols, dtype=types)
    df.to_json(args.out, orient='records', lines=True)

if __name__ == '__main__':
    main(parser.parse_args())
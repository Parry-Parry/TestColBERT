import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='New Col')

parser.add_argument('-source', type=str, default=None, help='Dataset Source')
parser.add_argument('-out', type=str, help='Output dir for New file')

def main(args):
    cols = ['qid', 'passage']
    types = {'qid': int, 'passage': str}
    with open(args.source, 'r') as f:
        df = pd.read_csv(f, sep='\t', header=None, index_col=False, names=cols, dtype=types)
    df['arbitrary'] = str(0)
    df.to_json(args.out, orient='records', lines=True, index=False)

if __name__ == '__main__':
    main(parser.parse_args())
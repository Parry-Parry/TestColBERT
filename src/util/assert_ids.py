import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-first', type=int, help='First to check [Smaller JSONL]')
parser.add_argument('-second', type=str, help='Second to check [TSV]')
parser.add_argument('-f_rank', type=int, help='First Column Rank to check')
parser.add_argument('-s_rank', type=int, help='Second Column Rank to check')

def main(args):
    with open(args.first, 'r') as f:
        f_df = pd.read_json(f, orient='records', lines=True)
    with open(args.second, 'r') as f:
        s_df = pd.read_csv(f, sep='\t', header=None, index_col=False)
    
    f_sub = f_df.iloc[:, [args.f_rank]]
    s_sub = s_df.loc[:, [args.s_rank]]

    print('Iterating...')
    counter = 0
    for _, val in f_sub.iteritems():
        if np.any(s_sub[s_sub == val]): counter += 1
    
    print(f'{counter} matches found out of {len(f_sub)} possible cases')
    
if __name__=='__main__':
    main(parser.parse_args())
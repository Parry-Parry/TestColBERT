import argparse
from math import trunc 
import ir_datasets
import random
import pandas as pd

parser = argparse.ArgumentParser(description='Build IR Dataset')

parser.add_argument('-dataset', type=str, default=None, help='Dataset Name')
parser.add_argument('-portion', type=int, default=None, help=r'% of dataset to use')
parser.add_argument('-out', type=str, help='Output dir for tsv')

def main(args):
    ds = ir_datasets.load(args.dataset)
    assert ds.has_doc_pairs()
    triplets = [triplet for triplet in ds.doc_pairs_iter()]

    num_choices = trunc(len(triplets) * (args.portion/100))
    new_triplets = random.choices(triplets, k=num_choices)

    tmp_df = {'qid':[], 'did_a':[], 'did_b':[]}

    for triplet in new_triplets:
        tmp_df['qid'].append(triplet.query_id)
        tmp_df['did_a'].append(triplet.doc_id_a)
        tmp_df['did_b'].append(triplet.doc_id_b)

    tmp_df = pd.DataFrame(tmp_df)
    tmp_df.to_csv(args.out + 'triples.tsv', sep='\t', index=False, header=False)


if __name__ -- '__main__':
    main(parser.parse_args())
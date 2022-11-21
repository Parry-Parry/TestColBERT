import pyterrier as pt
pt.init()
from pyterrier_colbert.indexing import ColBERTIndexer
import argparse 

parser = argparse.ArgumentParser(description='Build Index')

parser.add_argument('-dataset', type=str, help='Name of stored TREC formatted dataset')
parser.add_argument('-dir', type=str, default=None, help='Where to store index')
parser.add_argument('-index_name', type=str, default=None, help='Name of index')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint if not using default')

def main(args):
    dataset = pt.get_dataset("trec-deep-learning-passages")
    if not args.checkpoint: checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
    else: checkpoint = args.checkpoint

    indexer = ColBERTIndexer(checkpoint, args.dir, args.index_name, 64.0, ids=True)
    indexer.index(dataset.get_corpus_iter())

if __name__ == '__main__':
    main(parser.parse_args())




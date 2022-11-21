import pyterrier as pt
pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
import argparse 

parser = argparse.ArgumentParser(description='Build Index')

parser.add_argument('-checkpoint', type=str, help='Model Checkpoint')
parser.add_argument('-index_dir', type=str, default=None, help='Where index is stored')
parser.add_argument('-index_name', type=str, default=None, help='Name of index')
parser.add_argument('-out', type=str, default=None, help='Output Directory / Name of File')

def main(args):
    dataset = pt.get_dataset("trec-deep-learning-passages")

    pytcolbert = ColBERTFactory(args.checkpoint, args.index_dir, args.index_name)
    dense_e2e = pytcolbert.end_to_end()

    bm25 = pt.BatchRetrieve.from_dataset('msmarco_passage', 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
    sparse_colbert = bm25 >> pytcolbert.text_scorer()

    res = pt.Experiment(
    [bm25, sparse_colbert, dense_e2e],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=["map", "ndcg_cut_10"],
    names=["BM25", "BM25 >> ColBERT", "Dense ColBERT"]
    )

    if args.out:
        res.to_csv(args.out)
        return 0
    
    print("No Output Dir Specified printing output...")
    print(res)
    return 1



if __name__ == '__main__':
    main(parser.parse_args())
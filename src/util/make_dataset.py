import argparse 
import ir_datasets
import pickle
import pandas as pd
from transformers import AutoTokenizer
from util.structures import Text, Triple

parser = argparse.ArgumentParser(description='Build IR Dataset')

parser.add_argument('-dataset', type=str, default=None, help='Dataset Name')
parser.add_argument('-portion', type=int, default=None, help=r'% of dataset to use')
parser.add_argument('-tokenizer', type=str, help='What tokenizer to use')
parser.add_argument('-out', type=str, help='Output dir for csv')

def convert_to_pd(ds):
    tmp_df = dict()
    tmp_df['qid'] = []
    tmp_df['q'] = [] 
    tmp_df['d1'] = []
    tmp_df['d2'] = []
    for trip in ds:
        tmp_df['qid'].append(trip.q.id)
        tmp_df['q'].append(trip.q.embedding)
        tmp_df['d1'].append(trip.d1.embedding)
        tmp_df['d2'].append(trip.d2.embedding)
    
    return pd.DataFrame(tmp_df)

def init_embedding_factory(tokenizer, dataset):
    store = dataset.doc_store()
    queries = [query for query in dataset.queries_iter()]
    q_ids = [query.id for query in queries]
    def create_triplet(set):
        q_id, d1_id, d2_id = set.query_id, set.doc_id_a, set.doc_id_b
        d1 = store.get(d1_id)
        d2 = store.get(d2_id)
        query = queries[q_ids.index(q_id)]

        q_obj = Text(q_id, query.text, tokenizer(query.text))
        d1_obj = Text(d1_id, d1.text, tokenizer(d1.text))
        d2_obj = Text(d2_id, d2.text, tokenizer(d2.text))

        return Triple(q_obj, d1_obj, d2_obj)

    return create_triplet

def init_text_factory(dataset):
    store = dataset.doc_store()
    queries = [query for query in dataset.queries_iter()]
    q_ids = [query.id for query in queries]
    def create_triplet(set):
        q_id, d1_id, d2_id = set.query_id, set.doc_id_a, set.doc_id_b
        d1 = store.get(d1_id)
        d2 = store.get(d2_id)
        query = queries[q_ids.index(q_id)]

        return Triple(query.text, d1.text, d2.text)

    return create_triplet

def main(args):
    dataset = ir_datasets.load(args.dataset)
    portion = [triplet for triplet in dataset.docpairs_iter()[:args.portion/100]]
    main = [triplet for triplet in dataset.docpairs_iter()[args.portion/100:]]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    get_text = init_text_factory(dataset)

    portion_ds = list(map(get_text, portion))
    main_ds = list(map(get_text, main))

    tmp_struct = pd.DataFrame({'q':[trip.q for trip in portion_ds], 'd1' : [trip.d1 for trip in portion_ds], 'd2' : [trip.d2 for trip in portion_ds]})
    tmp_struct.to_csv(args.out + '/portion.csv', index=False)

    tmp_struct = pd.DataFrame({'q':[trip.q for trip in main_ds], 'd1' : [trip.d1 for trip in main_ds], 'd2' : [trip.d2 for trip in main_ds]})
    tmp_struct.to_csv(args.out + '/main.csv', index=False)

if __name__ -- '__main__':
    main(parser.parse_args())
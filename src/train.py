from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-ngpu', type=int, help='Number of GPUS')
parser.add_argument('-dir', type=str, help='Storage Dir of Dataset')
parser.add_argument('-out', type=str, help='Where to store checkpoints')
parser.add_argument('-suffix', type=str, help='Suffix of specific triplets')
parser.add_argument('-batch_size', type=int, default=128, help='Batch Size')



def main(args):
    assert args.ngpu > 0
    with Run().context(RunConfig(nranks=args.ngpu, experiment="msmarco")):

        config = ColBERTConfig(
            bsize=args.batch_size,
            root=config.out,
        )
        trainer = Trainer(
            triples=args.dir + f"/triples.{args.suffix}.tsv",
            queries=args.dir + "/queries.tsv",
            collection=args.dir + "/collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")

if __name__=='__main__':
    main(parser.parse_args())
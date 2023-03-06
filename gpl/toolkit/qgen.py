from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
import os
import argparse
from typing import Optional


def qgen(
    data_path,
    output_dir,
    generator_name_or_path="BeIR/query-gen-msmarco-t5-base-v1",
    ques_per_passage=3,
    bsz=32,
    qgen_prefix="qgen",
    device: Optional[str] = None,
):
    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus = GenericDataLoader(data_path).load_corpus()

    #### question-generation model loading
    generator = QGen(model=QGenModel(generator_name_or_path, device=device))

    #### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    #### https://huggingface.co/blog/how-to-generate
    #### Prefix is required to seperate out synthetic queries and qrels from original
    prefix = qgen_prefix

    #### Generating 3 questions per passage.
    #### Reminder the higher value might produce lots of duplicates
    #### Generate queries per passage from docs in corpus and save them in data_path
    try:
        generator.generate(
            corpus,
            output_dir=output_dir,
            ques_per_passage=ques_per_passage,
            prefix=prefix,
            batch_size=bsz,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError(
                f"CUDA out of memory during query generation "
                f"(queries_per_passage: {ques_per_passage}, batch_size_generation: {bsz}). "
                f"Please try smaller `queries_per_passage` and/or `batch_size_generation`."
            )

    if not os.path.exists(os.path.join(output_dir, "corpus.jsonl")):
        os.system(f"cp {data_path}/corpus.jsonl {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument('--device', type=str, required=False, default=None)
    args = parser.parse_args()
    qgen(args.data_path, args.output_dir, device=args.device)

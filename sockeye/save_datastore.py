# Author: ZL
import argparse
import logging
import sys
import time
import torch
from sockeye.model import load_models
from . import inference
from . import arguments
from .utils import read_file, write_file

def count_knn_kvs(path):
    """
    To get the size of knn datastore.
    """
    tgt_lines = read_file(path)
    kv_count = 0
    for tgt_line in tgt_lines:
        # The <EOS> token is also treated as a value.
        kv_count += tgt_line.strip().split(" ").__len__() +1
    return kv_count

def extract_keys_and_values(args,source_sentences,target_sentences):
    """
    Force decode and get keys and values at every timestamp in each sentence.
    """


def main(args):
    """
    Build knn datastore and save to disk.
    """
    source_sentences = read_file(args.datastore_src)
    target_sentences = read_file(args.datastore_tgt)
    keys,values = extract_keys_and_values(args,source_sentences,target_sentences)

if __name__ == "__main__":
    params = arguments.ConfigArgumentParser(description='Translate CLI')
    arguments.add_translate_cli_args(params)
    params.add_argument("--datastore-src",type=str,help="The path of source language data to build knn datastore.\
                         This file should be preprocessed(after tokenize, BPE ...).")
    params.add_argument("--datastore-tgt",type=str,help="The path of source language data to build knn datastore.\
                         This file should be preprocessed(after tokenize, BPE ...).")
    params.add_argument("--save-dir",type=str,help="The path to save the generated keys and values by faiss.")
    params.add_argument("--pretrained-model",type=str,help="The off-the-shell translation model to get knn keys.")
    args = params.parse_args()
    main(args)

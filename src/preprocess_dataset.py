# coding: utf-8
from typing import Any, Dict, Optional
import pickle

from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset
from llamafactory.model import load_tokenizer


def run_sft_preprocess(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, _, _ = get_train_args(args)
    tokenizer_module = load_tokenizer(model_args)
    dataset = get_dataset(model_args=model_args,
                          data_args=data_args,
                          training_args=training_args,
                          stage="sft",
                          **tokenizer_module)
    print(dataset.column_names)
    print('len(dataset): ', len(dataset))


if __name__ == '__main__':
    run_sft_preprocess()

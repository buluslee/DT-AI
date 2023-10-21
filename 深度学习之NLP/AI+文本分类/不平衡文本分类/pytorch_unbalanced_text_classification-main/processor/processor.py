import numpy as np
import os
import pickle
import torch
from torch.utils.data import TensorDataset


def convert_array_to_tensor(inputs, dtype=None):
    if not inputs is np.ndarray:
        inputs = np.array(inputs)
    if dtype:
        inputs = torch.tensor(inputs, dtype=dtype)
    else:
        inputs = torch.tensor(inputs)
    return inputs


class BilstmProcessor:
    def read_data(self, file_path):
        with open(file_path, 'r') as fp:
            raw_examples = fp.read().strip()
        return raw_examples

    def get_examples(self,
                     raw_examples,
                     max_seq_len,
                     tokenizer,
                     pickle_path,
                     label2id,
                     set_type,
                     sep='\t',
                     reverse=False):
        if not os.path.exists(pickle_path):
            total = len(raw_examples.split('\n'))

            input_ids_all = []
            label_ids_all = []
            for i, line in enumerate(raw_examples.split('\n')):
                print("process:{}/{}".format(i, total))
                line = line.split(sep)
                if reverse:
                    text = line[0]
                    label = line[1]
                else:
                    label = line[0]
                    text = line[1]
                input_ids = \
                    self.convert_text_to_feature(text, max_seq_len, tokenizer)
                if i < 3:
                    print(f"{set_type} example-{i}")
                    print(f"text：{text[:max_seq_len]}")
                    print(f"input_ids:{input_ids}")
                    print(f"label：{label}")
                input_ids_all.append(input_ids)
                label_ids_all.append(label2id[label])
            tensorDataset = TensorDataset(
                convert_array_to_tensor(input_ids_all),
                convert_array_to_tensor(label_ids_all),
            )
            with open(pickle_path, 'wb') as fp:
                pickle.dump(tensorDataset, fp)
        else:
            with open(pickle_path, 'rb') as fp:
                tensorDataset = pickle.load(fp)
        return tensorDataset

    def convert_text_to_feature(self, text, max_seq_len, tokenizer):
        text = [i for i in text]
        input_ids = []
        for i, char in enumerate(text):
            if i < max_seq_len:
                input_ids.append(tokenizer['char2id'].get(char, 1))
        if len(input_ids) < max_seq_len:
            input_ids = input_ids + (max_seq_len - len(input_ids)) * [0]

        assert len(input_ids) == max_seq_len

        return input_ids


PROCESSOR = {
    'BilstmProcessor': BilstmProcessor,
}

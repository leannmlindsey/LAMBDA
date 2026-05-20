"""CSV Dataset for generic binary/multiclass classification.

Loads sequences and labels from CSV files with "sequence" and "label" columns.
"""

import pandas as pd
import torch

from src.dataloaders.utils.rc import coin_flip, string_reverse_complement


class CSVDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading DNA sequences from CSV files.

    Expected CSV format:
        sequence,label
        ACGTACGT...,0
        TGCATGCA...,1
    """

    def __init__(
            self,
            csv_path,
            max_length,
            tokenizer=None,
            tokenizer_name=None,
            use_padding=True,
            d_output=2,
            add_eos=False,
            rc_aug=False,
            conjoin_train=False,
            conjoin_test=False,
            return_augs=False,
            return_mask=False,
            split="train",
    ):
        self.csv_path = csv_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.use_padding = use_padding
        self.d_output = d_output
        self.add_eos = add_eos
        self.return_augs = return_augs
        self.return_mask = return_mask
        self.split = split

        assert not (conjoin_train and conjoin_test), "conjoin_train and conjoin_test cannot both be True"
        if (conjoin_train or conjoin_test) and rc_aug:
            print("When using conjoin, we turn off rc_aug.")
            rc_aug = False
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

        # Load CSV file
        self.df = pd.read_csv(csv_path)

        # Validate required columns
        if "sequence" not in self.df.columns:
            raise ValueError(f"CSV file must contain 'sequence' column. Found columns: {list(self.df.columns)}")
        if "label" not in self.df.columns:
            raise ValueError(f"CSV file must contain 'label' column. Found columns: {list(self.df.columns)}")

        self.all_seqs = self.df["sequence"].tolist()
        self.all_labels = self.df["label"].tolist()

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        x = self.all_seqs[idx]
        y = self.all_labels[idx]

        # Apply reverse complement augmentation
        if (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(
            x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]

        # Handle EOS token
        if self.add_eos:
            seq_ids.append(self.tokenizer.sep_token_id)

        # Handle conjoining (forward + reverse complement)
        if self.conjoin_train or (self.conjoin_test and self.split != "train"):
            x_rc = string_reverse_complement(x)
            seq_rc = self.tokenizer(
                x_rc,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else None,
                max_length=self.max_length,
                truncation=True,
            )
            seq_rc_ids = seq_rc["input_ids"]
            if self.add_eos:
                seq_rc_ids.append(self.tokenizer.sep_token_id)
            seq_ids = torch.stack((torch.LongTensor(seq_ids), torch.LongTensor(seq_rc_ids)), dim=1)
        else:
            seq_ids = torch.LongTensor(seq_ids)

        target = torch.LongTensor([y])

        if self.return_mask:
            return seq_ids, target, {"mask": torch.BoolTensor(seq["attention_mask"])}
        else:
            return seq_ids, target

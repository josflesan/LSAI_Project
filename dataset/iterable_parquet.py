import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
import torch

class IterableParquetDataset(IterableDataset):
    def __init__(
    self,
    parquet_file: str,
    tokenizer,
    sequence_length: int,
    bos_token_id: int = 1
    ):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
    def __iter__(self):
        # Reset buffer and index when starting a new iteration
        token_buffer = []
        for doc in self.parquet_ds["text"]:
            encoded = self.tokenizer.encode(str(doc))
            for token_id in encoded:
                token_buffer.append(token_id)
                if len(self.token_buffer) == self.sequence.length + 1:
                    input_ids = torch.LongTensor(self.token_buffer)
                    inputs = input_ids[:, :-1].clone()
                    labels = input_ids[:, 1:]
                    labels[inputs == self.bos_token_id] = -100
                    yield inputs, labels
                    token_buffer = [token_id]
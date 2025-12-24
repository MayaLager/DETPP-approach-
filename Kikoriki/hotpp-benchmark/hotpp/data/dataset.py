import os
import itertools
import random
import torch
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from numbers import Number
from pyarrow.parquet import ParquetFile
from random import Random

from ptls.data_load import read_pyarrow_file
from ptls.data_load.datasets import parquet_file_scan

from .padded_batch import PaddedBatch


def get_nested_value(value):
    if isinstance(value, list):
        if len(value) == 0:
            return None
        return get_nested_value(value[0])
    return value


def get_parquet_length(path):
    with ParquetFile(path) as fp:
        return fp.metadata.num_rows


def to_torch_if_possible(v):
    if v is None:
        v = float("nan")
    try:
        if isinstance(v, np.ndarray):
            if (len(v) > 0) and isinstance(v[0], np.ndarray):
                v = np.stack(v)
            t = torch.from_numpy(v)
        else:
            t = torch.tensor(v)
        if torch.is_floating_point(t):
            t = t.float()
        return t
    except TypeError:
        return v


def parse_fields(fields):
    if fields is None:
        return []
    if isinstance(fields, str):
        return [fields]
    return list(fields)


class HotppDataset(torch.utils.data.IterableDataset):
    """Generate subsequences from parquet file.

    Dataset can contain target labels. Global targets are assigned to each ID and
    local targets are assigned to particular events.

    Args:
        data: Path to a parquet dataset or a list of files.
        min_length: Minimum sequence length. Use 0 to disable subsampling.
        max_length: Maximum sequence length. Disable limit if `None`.
        position: Sample position (`random` or `last`).
        fields: A list of fields to keep in data. Other fields will be discarded.
        drop_nans: A list of fields to skip nans for.
        add_seq_fields: A dictionary with additional constant fields.
        global_target_fields: The name of the target field or a list of fields. Global targets are assigned to sequences.
        local_targets_fields: The name of the target field or a list of fields. Local targets are assigned to individual events.
        local_targets_indices_field: The name of the target field or a list of fields. Local targets are assigned to individual events.
    """
    def __init__(self, data,
                 min_length=0, max_length=None,
                 position="random",
                 min_required_length=None,
                 fields=None,
                 id_field="id",
                 timestamps_field="timestamps",
                 drop_nans=None,
                 add_seq_fields=None,
                 global_target_fields=None,
                 local_targets_fields=None,
                 local_targets_indices_field=None):
        super().__init__()
        if isinstance(data, str):
            self.filenames = list(sorted(parquet_file_scan(data)))
        elif isinstance(data, list):
            self.filenames = data
        else:
            raise ValueError(f"Unknown data type: {type(data)}")
        if not self.filenames:
            raise RuntimeError("Empty dataset")
        self.total_length = sum(map(get_parquet_length, self.filenames))
        self.min_length = min_length
        self.max_length = max_length
        self.position = position
        self.min_required_length = min_required_length
        self.id_field = id_field
        self.timestamps_field = timestamps_field
        self.drop_nans = parse_fields(drop_nans)
        self.add_seq_fields = add_seq_fields
        self.global_target_fields = parse_fields(global_target_fields)

        if local_targets_fields and not local_targets_indices_field:
            raise ValueError("Need indices fol local targets.")
        self.local_targets_fields = parse_fields(local_targets_fields)
        self.local_targets_indices_field = local_targets_indices_field

        if fields is not None:
            known_fields = [id_field, timestamps_field] + list(self.global_target_fields) + list(self.local_targets_fields)
            if local_targets_indices_field is not None:
                known_fields = known_fields + [local_targets_indices_field]
            fields = list(sorted(set(fields) | set(known_fields)))
        self.fields = fields

    def shuffle_files(self, rnd=None):
        """Make a new dataset with shuffled partitions."""
        rnd = rnd if rnd is not None else random.Random()
        filenames = list(self.filenames)
        rnd.shuffle(filenames)
        return HotppDataset(filenames,
                            min_length=self.min_length, max_length=self.max_length,
                            position=self.position, min_required_length=self.min_required_length,
                            fields=self.fields, id_field=self.id_field, timestamps_field=self.timestamps_field,
                            drop_nans=self.drop_nans, global_target_fields=self.global_target_fields,
                            local_targets_fields=self.local_targets_fields, local_targets_indices_field=self.local_targets_indices_field)

    def is_seq_feature(self, name, value, batch=False):
        """Check whether feature is sequential using its name and value.

        Args:
            batch: Whether the value is a batch of features.
        """
        if (name == self.id_field) or (name in self.global_target_fields):
            return False
        if isinstance(value, list):
            ndim = 1
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            ndim = value.ndim
        else:
            ndim = 0
        return ndim > int(batch)

    def process(self, features):
        if self.id_field not in features:
            raise ValueError("Need ID feature")
        if self.timestamps_field not in features:
            raise ValueError("Need timestamps feature")
        if (self.min_length > 0) or (self.max_length is not None):
            if self.local_targets_fields:
                raise NotImplementedError("Future work: subsequence local targets.")
            # Select subsequences.
            length = len(features[self.timestamps_field])
            max_length = min(length, self.max_length or length)
            min_length = min(length, self.min_length if self.min_length > 0 else max_length)
            out_length = random.randint(min_length, max_length)
            if self.position == "random":
                offset = random.randint(0, length - out_length)
            elif self.position == "last":
                offset = length - out_length
            else:
                raise ValueError(f"Unknown position: {self.position}.")
            features = {k: (v[offset:offset + out_length] if self.is_seq_feature(k, v) else v)
                        for k, v in features.items()}
            assert len(features[self.timestamps_field]) == out_length
        features[self.timestamps_field] = features[self.timestamps_field].float()
        return features  # Tensors.

    def __len__(self):
        return self.total_length

    def __iter__(self):
        for filename in self.filenames:
            for rec in read_pyarrow_file(filename, use_threads=True):
                if (self.min_required_length is not None) and (len(rec[self.timestamps_field]) < self.min_required_length):
                    continue
                if self.fields is not None:
                    rec = {field: rec[field] for field in self.fields}
                features = {k: to_torch_if_possible(v) for k, v in rec.items()}
                skip = False
                for field in self.drop_nans:
                    if not features[field].isfinite().all():
                        skip = True
                        break
                if skip:
                    continue
                yield self.process(features)

    def _make_batch(self, by_name, batch_size, seq_feature_name=None):
        # Compute lengths.
        if seq_feature_name is not None:
            lengths = torch.tensor(list(map(len, by_name[seq_feature_name])))
        else:
            lengths = torch.zeros(batch_size, dtype=torch.long)

        # Add padding.
        features = {}
        for k, vs in by_name.items():
            if self.is_seq_feature(k, vs[0]):
                features[k] = torch.nn.utils.rnn.pad_sequence(vs, batch_first=True)  # (B, L, *).
            else:
                try:
                    features[k] = torch.stack(vs)  # (B, *).
                except TypeError:
                    features[k] = vs
        if not features:
            return None
        batch = PaddedBatch(features, lengths,
                            seq_names={k for k, v in features.items()
                                       if self.is_seq_feature(k, v, batch=True)})
        if self.add_seq_fields is not None:
            b, l = batch.shape
            payload = dict(batch.payload)
            for k, v in self.add_seq_fields.items():
                payload[k] = torch.full((b, l), v, device=batch.device)
            batch = PaddedBatch(payload, batch.seq_lens,
                                seq_names=set(batch.seq_names) | set(self.add_seq_fields))
        return batch

    def collate_fn(self, batch):
        batch_size = len(batch)
        by_name = defaultdict(list)
        for features in batch:
            for name, value in features.items():
                by_name[name].append(value)

        # Check batch size consistency.
        for name, values in by_name.items():
            if len(values) != batch_size:
                raise ValueError(f"Missing values for feature {name}")

        # Pop targets.
        targets_by_name = {name: by_name.pop(name) for name in
                           itertools.chain(self.global_target_fields, self.local_targets_fields)}

        # Make PaddedBatch objects.
        features = self._make_batch(by_name, batch_size, self.timestamps_field)
        targets = self._make_batch(targets_by_name, batch_size, self.local_targets_indices_field)
        return features, targets


class ShuffledDistributedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, rank=None, world_size=None, cache_size=None, seed=0):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.cache_size = cache_size
        self.seed = seed
        self.epoch = 0

    def _get_context(self):
        dataset = self.dataset
        rank = int(os.environ.get("RANK", self.rank if self.rank is not None else 0))
        world_size = int(os.environ.get("WORLD_SIZE", self.world_size if self.world_size is not None else 1))
        global_seed = self.seed + self.epoch

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker = 0
            num_workers = 1

        total_workers = world_size * num_workers
        worker_id = rank * num_workers + worker
        return dataset, worker_id, total_workers, rank, world_size, global_seed

    def __iter__(self):
        dataset, worker_id, total_workers, rank, world_size, global_seed = self._get_context()
        if (worker_id is None) and (total_workers == 1):
            worker_id = 0
        for i, item in enumerate(self._iter_shuffled(dataset, global_seed)):
            if (i - worker_id) % total_workers == 0:
                yield item

    def _iter_shuffled(self, dataset, seed):
        if self.cache_size is None:
            yield from dataset
        else:
            rnd = Random(seed)
            cache = []
            for item in dataset.shuffle_files(rnd):
                cache.append(item)
                if len(cache) >= self.cache_size:
                    rnd.shuffle(cache)
                    yield from cache
                    cache = []
            if len(cache) > 0:
                rnd.shuffle(cache)
                yield from cache


class HotppCSVDataset(torch.utils.data.IterableDataset):
    """
    CSV version of HotppDataset.

    Expected columns:
      - id
      - timestamps
      - labels
    plus optional target columns (global_target_fields), e.g. target_label, target_timestamp

    timestamps/labels can be:
      - python-like lists already (rare in CSV)
      - JSON strings: "[1,2,3]"
    """

    def __init__(self, data,
                 min_length=0, max_length=None,
                 position="random",
                 min_required_length=None,
                 fields=None,
                 id_field="id",
                 timestamps_field="timestamps",
                 drop_nans=None,
                 add_seq_fields=None,
                 global_target_fields=None,
                 local_targets_fields=None,
                 local_targets_indices_field=None):
        super().__init__()
        if isinstance(data, str):
            self.csv_path = data
        else:
            raise ValueError("HotppCSVDataset expects a single .csv path (string).")

        self.min_length = min_length
        self.max_length = max_length
        self.position = position
        self.min_required_length = min_required_length
        self.id_field = id_field
        self.timestamps_field = timestamps_field
        self.drop_nans = parse_fields(drop_nans)
        self.add_seq_fields = add_seq_fields
        self.global_target_fields = parse_fields(global_target_fields)

        if local_targets_fields and not local_targets_indices_field:
            raise ValueError("Need indices for local targets.")
        self.local_targets_fields = parse_fields(local_targets_fields)
        self.local_targets_indices_field = local_targets_indices_field

        self.fields = fields

        # We don't know true length without reading file; keep as None-ish.
        self.total_length = None

    def __len__(self):
        # Optional: could compute once, but IterableDataset doesn't require it.
        raise TypeError("HotppCSVDataset length is unknown (IterableDataset).")

    @staticmethod
    def _maybe_json_list(x):
        if isinstance(x, list):
            return x
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if isinstance(x, str):
            s = x.strip()
            # try JSON
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try:
                    return json.loads(s)
                except Exception:
                    pass
            # fallback: single value string
            return [x]
        return [x]

    def process(self, features):
        # same logic as HotppDataset.process
        if self.id_field not in features:
            raise ValueError("Need ID feature")
        if self.timestamps_field not in features:
            raise ValueError("Need timestamps feature")

        if (self.min_length > 0) or (self.max_length is not None):
            if self.local_targets_fields:
                raise NotImplementedError("Subsequence local targets not supported.")
            length = len(features[self.timestamps_field])
            max_length = min(length, self.max_length or length)
            min_length = min(length, self.min_length if self.min_length > 0 else max_length)
            out_length = random.randint(min_length, max_length)
            if self.position == "random":
                offset = random.randint(0, length - out_length)
            elif self.position == "last":
                offset = length - out_length
            else:
                raise ValueError(f"Unknown position: {self.position}.")
            features = {k: (v[offset:offset + out_length] if self.is_seq_feature(k, v) else v)
                        for k, v in features.items()}
            assert len(features[self.timestamps_field]) == out_length

        features[self.timestamps_field] = features[self.timestamps_field].float()
        return features

    def is_seq_feature(self, name, value, batch=False):
        if (name == self.id_field) or (name in self.global_target_fields):
            return False
        if isinstance(value, list):
            ndim = 1
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            ndim = value.ndim
        else:
            ndim = 0
        return ndim > int(batch)

    def __iter__(self):
        df = pd.read_csv(self.csv_path)

        # keep only requested fields if given
        if self.fields is not None:
            df = df[self.fields]

        # ensure required exist
        if self.id_field not in df.columns:
            raise ValueError(f"Missing column {self.id_field} in {self.csv_path}")
        if self.timestamps_field not in df.columns:
            raise ValueError(f"Missing column {self.timestamps_field} in {self.csv_path}")

        for row in df.itertuples(index=False):
            rec = row._asdict()

            # parse list-like fields
            rec[self.timestamps_field] = self._maybe_json_list(rec[self.timestamps_field])
            # common sequential label field name in hotpp is "labels"
            if "labels" in rec:
                rec["labels"] = self._maybe_json_list(rec["labels"])

            # enforce min_required_length on timestamps
            if (self.min_required_length is not None) and (len(rec[self.timestamps_field]) < self.min_required_length):
                continue

            # to torch tensors
            features = {k: to_torch_if_possible(v) for k, v in rec.items()}

            # drop NaNs check
            skip = False
            for field in self.drop_nans:
                if not features[field].isfinite().all():
                    skip = True
                    break
            if skip:
                continue

            yield self.process(features)

    # reuse same collate_fn as HotppDataset:
    def _make_batch(self, by_name, batch_size, seq_feature_name=None):
        if seq_feature_name is not None:
            lengths = torch.tensor(list(map(len, by_name[seq_feature_name])))
        else:
            lengths = torch.zeros(batch_size, dtype=torch.long)

        features = {}
        for k, vs in by_name.items():
            if self.is_seq_feature(k, vs[0]):
                features[k] = torch.nn.utils.rnn.pad_sequence(vs, batch_first=True)
            else:
                try:
                    features[k] = torch.stack(vs)
                except TypeError:
                    features[k] = vs
        if not features:
            return None

        batch = PaddedBatch(features, lengths,
                            seq_names={k for k, v in features.items()
                                       if self.is_seq_feature(k, v, batch=True)})

        if self.add_seq_fields is not None:
            b, l = batch.shape
            payload = dict(batch.payload)
            for k, v in self.add_seq_fields.items():
                payload[k] = torch.full((b, l), v, device=batch.device)
            batch = PaddedBatch(payload, batch.seq_lens,
                                seq_names=set(batch.seq_names) | set(self.add_seq_fields))
        return batch

    def collate_fn(self, batch):
        batch_size = len(batch)
        by_name = defaultdict(list)
        for features in batch:
            for name, value in features.items():
                by_name[name].append(value)

        for name, values in by_name.items():
            if len(values) != batch_size:
                raise ValueError(f"Missing values for feature {name}")

        targets_by_name = {name: by_name.pop(name) for name in
                           itertools.chain(self.global_target_fields, self.local_targets_fields)}

        features = self._make_batch(by_name, batch_size, self.timestamps_field)
        targets = self._make_batch(targets_by_name, batch_size, self.local_targets_indices_field)
        return features, targets



class HotppEventsCSVDataset(torch.utils.data.IterableDataset):
    """
    Read event-level CSV:
      columns: user_id, item_id, timestamp
    and yield sequence-level records:
      {"id": user, "timestamps": Tensor[L], "labels": Tensor[L], ...optional targets...}

    Works streaming if CSV is sorted by (user_id, timestamp).
    If not sorted, you can set assume_sorted=False (loads all and sorts in memory).
    """

    def __init__(
        self,
        data: str,
        min_length=0, max_length=None,
        position="random",
        min_required_length=None,
        fields=None,                 # ignored for event-level; we know required cols
        id_field="id",
        timestamps_field="timestamps",
        drop_nans=None,
        add_seq_fields=None,
        global_target_fields=None,   # ignored here (event-level csv doesn’t carry targets)
        local_targets_fields=None,
        local_targets_indices_field=None,
        # CSV-specific:
        user_col="user_id",
        item_col="item_id",
        time_col="timestamp",
        chunksize=2_000_000,
        assume_sorted=True,
    ):
        super().__init__()
        if not isinstance(data, str):
            raise ValueError("HotppEventsCSVDataset expects a single CSV path string.")
        self.csv_path = data

        # keep same knobs as HotppDataset
        self.min_length = min_length
        self.max_length = max_length
        self.position = position
        self.min_required_length = min_required_length
        self.id_field = id_field
        self.timestamps_field = timestamps_field
        self.drop_nans = parse_fields(drop_nans)
        self.add_seq_fields = add_seq_fields

        if local_targets_fields or local_targets_indices_field:
            raise NotImplementedError("Event-level CSV dataset: local targets not supported.")

        # CSV specifics
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.chunksize = chunksize
        self.assume_sorted = assume_sorted

    def __len__(self):
        raise TypeError("Length is unknown for streaming CSV IterableDataset.")

    def is_seq_feature(self, name, value, batch=False):
        # In our sequence-level output, id is non-seq; timestamps/labels are seq.
        if name == self.id_field:
            return False
        if isinstance(value, list):
            ndim = 1
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            ndim = value.ndim
        else:
            ndim = 0
        return ndim > int(batch)

    def process(self, features):
        # same subsequence sampling logic as HotppDataset.process
        if self.id_field not in features:
            raise ValueError("Need ID feature")
        if self.timestamps_field not in features:
            raise ValueError("Need timestamps feature")

        if (self.min_length > 0) or (self.max_length is not None):
            length = len(features[self.timestamps_field])
            max_length = min(length, self.max_length or length)
            min_length = min(length, self.min_length if self.min_length > 0 else max_length)
            out_length = random.randint(min_length, max_length)

            if self.position == "random":
                offset = random.randint(0, length - out_length)
            elif self.position == "last":
                offset = length - out_length
            else:
                raise ValueError(f"Unknown position: {self.position}.")

            features = {
                k: (v[offset:offset + out_length] if self.is_seq_feature(k, v) else v)
                for k, v in features.items()
            }
            assert len(features[self.timestamps_field]) == out_length

        features[self.timestamps_field] = features[self.timestamps_field].float()

        if self.add_seq_fields is not None:
            l = len(features[self.timestamps_field])
            for k, v in self.add_seq_fields.items():
                features[k] = torch.full((l,), v)

        return features

    def _emit_user(self, user, ts_list, it_list):
        if user is None:
            return None
        if (self.min_required_length is not None) and (len(ts_list) < self.min_required_length):
            return None

        features = {
            self.id_field: to_torch_if_possible(user),
            self.timestamps_field: to_torch_if_possible(np.asarray(ts_list, dtype=np.float32)),
            "labels": to_torch_if_possible(np.asarray(it_list, dtype=np.int64)),
        }

        # drop_nans check
        for field in self.drop_nans:
            if field in features and isinstance(features[field], torch.Tensor):
                if not features[field].isfinite().all():
                    return None

        return self.process(features)

    def __iter__(self):
        usecols = [self.user_col, self.item_col, self.time_col]

        if not self.assume_sorted:
            # simple but memory-heavy: load all and sort
            df = pd.read_csv(self.csv_path, usecols=usecols)
            df = df.dropna()
            df = df.sort_values([self.user_col, self.time_col], kind="mergesort")
            yield from self._iter_sorted_df(df)
            return

        # streaming: require sorted by (user_id, timestamp)
        reader = pd.read_csv(self.csv_path, usecols=usecols, chunksize=self.chunksize)
        cur_user = None
        cur_ts = []
        cur_it = []

        for chunk in reader:
            chunk = chunk.dropna()
            # do NOT sort chunk — that breaks user continuity unless file already sorted
            for u, it, t in zip(chunk[self.user_col].values,
                                chunk[self.item_col].values,
                                chunk[self.time_col].values):
                if cur_user is None:
                    cur_user = u

                if u != cur_user:
                    out = self._emit_user(cur_user, cur_ts, cur_it)
                    if out is not None:
                        yield out
                    cur_user = u
                    cur_ts = []
                    cur_it = []

                cur_ts.append(float(t))
                cur_it.append(int(it) if isinstance(it, (int, np.integer)) else int(str(it)) if str(it).isdigit() else hash(str(it)) % (2**31))

        out = self._emit_user(cur_user, cur_ts, cur_it)
        if out is not None:
            yield out

    def _iter_sorted_df(self, df: pd.DataFrame):
        for user, grp in df.groupby(self.user_col, sort=False):
            ts = grp[self.time_col].astype(float).tolist()
            it_raw = grp[self.item_col].tolist()
            it = []
            for x in it_raw:
                if isinstance(x, (int, np.integer)):
                    it.append(int(x))
                else:
                    sx = str(x)
                    it.append(int(sx) if sx.isdigit() else hash(sx) % (2**31))
            out = self._emit_user(user, ts, it)
            if out is not None:
                yield out

    # reuse same collate_fn logic as HotppDataset
    def _make_batch(self, by_name, batch_size, seq_feature_name=None):
        if seq_feature_name is not None:
            lengths = torch.tensor(list(map(len, by_name[seq_feature_name])))
        else:
            lengths = torch.zeros(batch_size, dtype=torch.long)

        features = {}
        for k, vs in by_name.items():
            if self.is_seq_feature(k, vs[0]):
                features[k] = torch.nn.utils.rnn.pad_sequence(vs, batch_first=True)
            else:
                try:
                    features[k] = torch.stack(vs)
                except TypeError:
                    features[k] = vs

        batch = PaddedBatch(features, lengths,
                            seq_names={k for k, v in features.items()
                                       if self.is_seq_feature(k, v, batch=True)})

        return batch

    def collate_fn(self, batch):
        batch_size = len(batch)
        by_name = defaultdict(list)
        for features in batch:
            for name, value in features.items():
                by_name[name].append(value)

        features = self._make_batch(by_name, batch_size, self.timestamps_field)
        targets = None
        return features, targets

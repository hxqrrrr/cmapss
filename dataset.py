# dataset.py (Engine-wise Holdout aligned)
import os
import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple, List, Sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch


def _xu_window_for(fd: str) -> int:
    fd = fd.upper()
    if fd == "FD001": return 31
    if fd == "FD002": return 30
    if fd == "FD003": return 60
    if fd == "FD004": return 50
    raise ValueError(f"Unknown fault mode: {fd}")


def _xu_14_sensor_cols_0based() -> List[int]:
    # 1-based sensors: 2,3,4,7,8,9,11,12,13,14,15,17,20,21
    # file 0-based col = 4 + sensor_id
    chosen = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
    return [k + 4 for k in chosen]


class CMAPSSDataset(Dataset):
    """
    Per-unit sliding windows (no cross-unit).
    Engine-wise Holdout ready:
      - split ∈ {"train","val","test"}
      - "train" / "val" both read official train_FDxxx.txt; "test" reads test_FDxxx.txt
      - scaler.fit only on TRAIN SUBSET units; "val"/"test" MUST reuse that scaler (no leakage)
    Preset 'xu2023': 14 sensors + MinMax + piecewise-125 + fixed window + stride=1
    """
    def __init__(
        self,
        data_dir: str,
        fault_mode: str,
        split: Literal["train", "val", "test"] = "train",
        preset: Literal["xu2023", "custom"] = "xu2023",
        # custom options:
        window_size: Optional[int] = None,
        stride: int = 1,
        features: Literal["14sensors", "sensors", "all"] = "14sensors",
        norm: Literal["minmax", "zscore", "none"] = "minmax",
        label_mode: Literal["piecewise_125", "natural"] = "piecewise_125",
        rul_clip: Optional[int] = None,
        scaler: Optional[object] = None,
        # engine-wise selection
        unit_whitelist: Optional[Sequence[int]] = None,
        unit_blacklist: Optional[Sequence[int]] = None,
        # rebalancing (applies to TRAIN ONLY)
        keep_natural_rul_le: Optional[int] = None,
        keep_last_k_per_unit: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.fault_mode = fault_mode.upper()
        self.split = split.lower()
        assert self.split in {"train", "val", "test"}, "split must be 'train'|'val'|'test'"

        # columns (0-based)
        self.ID_COL = 0
        self.CYCLE_COL = 1
        self.SETTING_COLS = [2, 3, 4]
        self.SENSOR_COLS = list(range(5, 26))

        # preset
        if preset == "xu2023":
            window_size = _xu_window_for(self.fault_mode)
            stride = 1
            features = "14sensors"
            norm = "minmax"
            label_mode = "piecewise_125"
            rul_clip = 125 if rul_clip is None else rul_clip
            if keep_natural_rul_le is None:
                keep_natural_rul_le = 140
            if keep_last_k_per_unit is None and self.split == "train":
                keep_last_k_per_unit = 150

        self.window_size = int(window_size) if window_size is not None else 128
        self.stride = int(stride)
        self.features_mode = features
        self.norm = norm
        self.label_mode = label_mode
        self.rul_clip = rul_clip

        # For strict holdout: disable rebalancing on VAL/TEST
        if self.split in {"val", "test"} and (keep_natural_rul_le is not None or keep_last_k_per_unit is not None):
            print("[WARN] Rebalancing is TRAIN-only. Ignoring keep_* for split", self.split)
            keep_natural_rul_le = None
            keep_last_k_per_unit = None
        self.keep_natural_rul_le = keep_natural_rul_le
        self.keep_last_k_per_unit = keep_last_k_per_unit

        # file paths
        self.train_path = os.path.join(self.data_dir, f"train_{self.fault_mode}.txt")
        self.test_path  = os.path.join(self.data_dir, f"test_{self.fault_mode}.txt")
        self.rul_path   = os.path.join(self.data_dir, f"RUL_{self.fault_mode}.txt")

        train_df = pd.read_csv(self.train_path, header=None, sep=r"\s+").astype(float)
        test_df  = pd.read_csv(self.test_path,  header=None, sep=r"\s+").astype(float)
        rul_df   = pd.read_csv(self.rul_path,   header=None, sep=r"\s+").astype(float)

        # feature columns
        if self.features_mode == "14sensors":
            feat_cols = _xu_14_sensor_cols_0based()
        elif self.features_mode == "sensors":
            feat_cols = self.SENSOR_COLS
        elif self.features_mode == "all":
            feat_cols = self.SETTING_COLS + self.SENSOR_COLS
        else:
            if self.fault_mode in ["FD002", "FD004"]:
                feat_cols = self.SETTING_COLS + self.SENSOR_COLS
            else:
                feat_cols = _xu_14_sensor_cols_0based()
        self.feat_cols = feat_cols

        # choose base df by split
        if self.split in {"train", "val"}:
            base_df = train_df.copy()
        else:
            base_df = test_df.copy()

        # engine-wise filter
        if unit_blacklist:
            base_df = base_df[~base_df.iloc[:, self.ID_COL].isin(unit_blacklist)]
        if unit_whitelist:
            base_df = base_df[base_df.iloc[:, self.ID_COL].isin(unit_whitelist)]

        # scaler
        if self.norm == "minmax":
            self.scaler = scaler if scaler is not None else MinMaxScaler()
        elif self.norm == "zscore":
            self.scaler = scaler if scaler is not None else StandardScaler()
        else:
            self.scaler = None

        # STRICT anti-leakage:
        # - TRAIN: if scaler is None -> fit on TRAIN SUBSET ONLY (after unit filter)
        # - VAL/TEST: scaler MUST be provided (no fit allowed)
        if self.scaler is not None:
            if self.split == "train":
                if scaler is None:
                    # fit on TRAIN subset features only
                    self.scaler.fit(base_df.iloc[:, self.feat_cols].values)
            else:
                if scaler is None:
                    raise RuntimeError(
                        f"[Holdout safety] split='{self.split}' requires a pre-fitted scaler from TRAIN subset."
                    )

        # apply transform using the (pre)fitted scaler
        if self.scaler is not None:
            base_df.loc[:, self.feat_cols] = self.scaler.transform(base_df.iloc[:, self.feat_cols].values)
            # also transform the other partition for convenience (not used here)
            train_df.loc[:, self.feat_cols] = self.scaler.transform(train_df.iloc[:, self.feat_cols].values)
            test_df.loc[:,  self.feat_cols] = self.scaler.transform(test_df.iloc[:,  self.feat_cols].values)

        # store
        self._train_df, self._test_df, self._rul_df = train_df, test_df, rul_df
        self._df = base_df  # filtered & transformed df for this split

        # per-unit caches
        self.units = sorted(self._df.iloc[:, self.ID_COL].unique().astype(int).tolist())
        self.unit_feats, self.unit_cycles, self.unit_last_cycle = {}, {}, {}
        for uid in self.units:
            df_u = self._df[self._df.iloc[:, self.ID_COL] == uid]
            self.unit_feats[uid]  = df_u.iloc[:, self.feat_cols].to_numpy(dtype=np.float32)
            self.unit_cycles[uid] = df_u.iloc[:, self.CYCLE_COL].to_numpy(dtype=np.int32)
            self.unit_last_cycle[uid] = int(self.unit_cycles[uid].max())

        # test RUL mapping (unit order assumed ascending)
        self.test_rul_map = {}
        if self.split == "test":
            test_units_sorted = sorted(self._test_df.iloc[:, self.ID_COL].unique().astype(int).tolist())
            if len(test_units_sorted) != len(rul_df):
                # safety check (rare but useful)
                print("[WARN] Test units count and RUL rows mismatch. Make sure ordering matches.")
            for i, uid in enumerate(test_units_sorted):
                self.test_rul_map[uid] = float(rul_df.iloc[i, 0])

        # build sample indices [(uid, start)]
        self.sample_index: List[Tuple[int, int]] = []
        W, S = self.window_size, self.stride
        for uid in self.units:
            n = len(self.unit_cycles[uid])
            if n >= W:
                all_starts = list(range(0, n - W + 1, S))
                if self.split == "train":
                    filtered_starts = []
                    for start in all_starts:
                        end_cycle = int(self.unit_cycles[uid][start + W - 1])
                        natural_rul = self.unit_last_cycle[uid] - end_cycle
                        if self.keep_natural_rul_le is not None and natural_rul > self.keep_natural_rul_le:
                            continue
                        filtered_starts.append(start)
                    if self.keep_last_k_per_unit is not None and len(filtered_starts) > self.keep_last_k_per_unit:
                        filtered_starts = filtered_starts[-self.keep_last_k_per_unit:]
                    starts_to_use = filtered_starts
                else:
                    starts_to_use = all_starts
                for start in starts_to_use:
                    self.sample_index.append((uid, start))

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        uid, start = self.sample_index[idx]
        W = self.window_size
        feats = self.unit_feats[uid]
        cycles = self.unit_cycles[uid]

        x = feats[start: start + W]  # (W, C)
        end_cycle = int(cycles[start + W - 1])
        last_cycle = self.unit_last_cycle[uid]

        # labels
        if self.split in {"train", "val"}:
            natural = float(last_cycle - end_cycle)
            y = min(125.0, natural) if self.label_mode == "piecewise_125" else natural
        else:  # test
            base_rul = self.test_rul_map[uid]
            y = float(base_rul + (last_cycle - end_cycle))
            if self.label_mode == "piecewise_125":
                y = min(125.0, y)

        if self.rul_clip is not None:
            y = float(min(self.rul_clip, max(0.0, y)))

        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0, drop_last=False, pin_memory=False) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_features(self) -> int:
        return len(self.feat_cols)

    def get_scaler(self):
        return self.scaler

    # diagnostics
    def analyze_training_labels(self):
        if self.split not in {"train", "val"}:
            print("Only for train/val splits.")
            return
        ys = []
        for uid, start in self.sample_index:
            end_cycle = self.unit_cycles[uid][start + self.window_size - 1]
            natural = self.unit_last_cycle[uid] - int(end_cycle)
            y = min(125.0, natural) if self.label_mode == "piecewise_125" else float(natural)
            ys.append(y)
        ys = np.array(ys)
        print(f"{self.split} 标签分布：min/median/max = {ys.min():.1f}/{np.median(ys):.1f}/{ys.max():.1f}")
        print(f"标签==125 的比例：{np.mean(ys==125)*100:.1f}%")
        print(f"标签>100 的比例：{np.mean(ys>100)*100:.1f}%")
        print(f"标签<=50 的比例：{np.mean(ys<=50)*100:.1f}%")
        print(f"总样本数：{len(ys)}")

    def analyze_last_window_labels(self):
        last_y = []
        for uid in self.units:
            idxs = [i for i, (u, _) in enumerate(self.sample_index) if u == uid]
            if idxs:
                _, y = self[idxs[-1]]
                last_y.append(float(y.item()))
        if last_y:
            import numpy as np
            print(f"最后窗口 RUL：min/median/max = {min(last_y):.1f}/{np.median(last_y):.1f}/{max(last_y):.1f}")
            print(f"最后窗口样本数：{len(last_y)}")
        else:
            print("未找到最后窗口样本")

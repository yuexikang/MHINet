"""RGB512 manifests use source-image identity, not legacy geographic filenames."""
from collections import Counter
import json
from pathlib import Path

from mhinet.dataio.data import _IndexEntry
from mhinet.pretraining.data import SharedPairDataset


class RGB512Dataset(SharedPairDataset):
    """Retain the existing image/GT loader with an explicit RGB manifest index.

    The original RGB512 split is by source image. It does not establish spatial
    independence between neighbouring FLAIR tiles; report that limitation rather
    than inventing geographic coordinates or changing the frozen split.
    """
    grouping_policy = "source_dataset + source_relative_path (fallback source_path); source-image isolation, not geographic-neighbour isolation"

    def __init__(self, manifest, *, image_size=512, max_pairs=None):
        if image_size != 512:
            raise ValueError("RGB512Dataset only accepts 512")
        self.manifest = Path(manifest).resolve()
        self.split_root = self.manifest.parent
        self.image_size = 512
        self.include_overlap_mask = False
        self.index = []; self.source_counts = Counter()
        identities = set()
        with self.manifest.open("rb") as stream:
            while True:
                offset = stream.tell(); line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("size_A") != [512, 512] or row.get("size_B") != [512, 512]:
                    raise ValueError("RGB512 manifests require stored 512x512 images on both sides")
                dataset = row.get("source_dataset")
                original = row.get("source_relative_path") or row.get("source_path")
                if not dataset or not original:
                    raise ValueError("RGB512 row lacks its original source identity")
                if row.get("split") != self.split_root.name:
                    raise ValueError("Manifest row split disagrees with its directory")
                pair = str(row["pair_id"])
                if pair in identities:
                    raise ValueError(f"Duplicate RGB512 pair ID: {pair}")
                identities.add(pair)
                group = f"source:{dataset}:{Path(original).as_posix()}"
                self.index.append(_IndexEntry(offset, pair, group, group))
                self.source_counts[dataset] += 1
                if max_pairs is not None and len(self.index) >= max_pairs:
                    break

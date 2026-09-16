"""Read both mutual-visibility masks without changing the legacy trainer."""
import numpy as np
import torch
from PIL import Image
from mhinet.dataio.data import HomographyPairDataset


class SharedPairDataset(HomographyPairDataset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        record = self._read_record(self.index[index].offset)
        for side in ('A', 'B'):
            name = f'mask_{side}_overlap'
            with Image.open(self.split_root / record[name]) as image:
                array = np.asarray(image.convert('L').resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST)).copy()
            result[name] = torch.from_numpy(array).float()[None] / 255
        return result

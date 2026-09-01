import os

import cv2
import numpy
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """Bi-temporal change detection dataset.

    Expected directory structure:
        <dataset_root>/
        ├── A/            # pre-change images
        ├── B/            # post-change images
        ├── label/        # ground-truth change masks
        └── list/         # train.txt / val.txt / test.txt

    List files contain one filename per line.  Most datasets include the
    extension (e.g. ``train_00001.png``).  CDD-CD-256 is unusual: list files
    contain bare numbers (``00000``), actual files are ``train_00000.jpg`` /
    ``val_00000.jpg`` / ``test_00000.jpg``.  This class auto-probes the
    ``A/`` directory to resolve the real on-disk name.
    """

    _EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    def __init__(self, dataset, file_root='data/', transform=None, list_name=None):
        self.dataset = dataset
        self.file_root = file_root
        self.transform = transform

        split = list_name or dataset
        list_path = os.path.join(file_root, 'list', f'{split}.txt')

        # legacy fallback: <root>/<dataset>/list/<dataset>.txt
        legacy_list_path = os.path.join(file_root, dataset, 'list', f'{dataset}.txt')

        if os.path.isfile(list_path):
            self.list_path = list_path
            base_root = file_root
        elif os.path.isfile(legacy_list_path):
            self.list_path = legacy_list_path
            base_root = os.path.join(file_root, dataset)
        else:
            raise FileNotFoundError(
                f'List file not found: {list_path}  or  {legacy_list_path}'
            )

        with open(self.list_path, 'r', encoding='utf-8') as f:
            raw_names = [x.strip() for x in f if x.strip()]

        # ------------------------------------------------------------------
        # Resolve actual on-disk filenames
        # ------------------------------------------------------------------
        self.file_list = []
        for name in raw_names:
            resolved = self._resolve_filename(base_root, name, split)
            self.file_list.append(resolved)

        self.pre_images = [os.path.join(base_root, 'A', x) for x in self.file_list]
        self.post_images = [os.path.join(base_root, 'B', x) for x in self.file_list]
        self.gts = [os.path.join(base_root, 'label', x) for x in self.file_list]

    # ------------------------------------------------------------------
    def _resolve_filename(self, base_root, name, split):
        """Return the on-disk filename with extension.

        Resolution order (first match wins):
        1. *name* already has an extension → trust it as-is.
        2. *name* as-is (no-extension edge case).
        3. ``{split}_{name}.ext`` — CDD-style prefix (e.g. ``train_00000.jpg``).
        4. ``{name}.ext`` — bare name + extension.
        """
        # 1. recognised extension already present
        if os.path.splitext(name)[1].lower() in self._EXTENSIONS:
            return name

        # 2. try as-is
        if os.path.isfile(os.path.join(base_root, 'A', name)):
            return name

        # 3. CDD-style: "{split}_{name}.ext"
        for ext in self._EXTENSIONS:
            candidate = f'{split}_{name}{ext}'
            if os.path.isfile(os.path.join(base_root, 'A', candidate)):
                return candidate

        # 4. bare "{name}.ext"
        for ext in self._EXTENSIONS:
            candidate = f'{name}{ext}'
            if os.path.isfile(os.path.join(base_root, 'A', candidate)):
                return candidate

        raise FileNotFoundError(
            f'Cannot find image file for "{name}" (split={split}) '
            f'in {os.path.join(base_root, "A")}'
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image = cv2.imread(self.pre_images[idx], cv2.IMREAD_COLOR)
        post_image = cv2.imread(self.post_images[idx], cv2.IMREAD_COLOR)
        label = cv2.imread(self.gts[idx], cv2.IMREAD_GRAYSCALE)

        if pre_image is None:
            raise FileNotFoundError(self.pre_images[idx])
        if post_image is None:
            raise FileNotFoundError(self.post_images[idx])
        if label is None:
            raise FileNotFoundError(self.gts[idx])

        img = numpy.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            img, label = self.transform(img, label)

        return img, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}

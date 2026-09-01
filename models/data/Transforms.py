import numpy as np
import torch
import random
import cv2
import numpy


class AmpMix(object):
    """Amplitude-invariant augmentation (Run12).

    Replace spatial image with another image's amplitude spectrum in FFT domain,
    reducing illumination/contrast pseudo-changes while preserving structure.

    Args:
        prob: probability of applying this augmentation (default 0.5).
        same_image_pool: if True, swap amplitude between T1 and T2 of the same pair;
                         if False, swap with a random image from the dataset (not implemented).
    """

    def __init__(self, prob=0.5, same_image_pool=True):
        self.prob = prob
        self.same_image_pool = same_image_pool

    def __call__(self, image, label):
        if random.random() > self.prob:
            return [image, label]

        # Defensive checks: AmpMix operates on raw [0, 255] uint8 images
        if image.ndim != 3 or image.shape[2] != 6:
            raise ValueError(
                f'AmpMix expects [H, W, 6] uint8 input, got shape {image.shape}'
            )
        if image.dtype != np.uint8:
            raise TypeError(
                f'AmpMix expects uint8 input (raw pixel values [0, 255]), '
                f'got dtype {image.dtype}. Ensure AmpMix is placed BEFORE Normalize.'
            )
        if image.min() < 0 or image.max() > 255:
            raise ValueError(
                f'AmpMix input out of range [0, 255]: min={image.min()}, max={image.max()}'
            )

        # image: [H, W, 6] uint8, [T1_B, T1_G, T1_R, T2_B, T2_G, T2_R]
        h, w = image.shape[:2]
        t1 = image[:, :, 0:3].astype(np.float32)
        t2 = image[:, :, 3:6].astype(np.float32)

        if self.same_image_pool:
            # Swap amplitude between T1 and T2
            t1_mixed = self._amp_swap(t1, t2)
            t2_mixed = self._amp_swap(t2, t1)
        else:
            # For cross-sample swap, need dataset-level implementation
            # Currently fallback to same-pair swap
            t1_mixed = self._amp_swap(t1, t2)
            t2_mixed = self._amp_swap(t2, t1)

        image_mixed = np.concatenate([t1_mixed, t2_mixed], axis=2).astype(np.uint8)
        return [image_mixed, label]

    def _amp_swap(self, spatial_img, amp_source_img):
        """Replace spatial_img's amplitude with amp_source_img's amplitude.

        Args:
            spatial_img: [H, W, 3] float32, provides phase.
            amp_source_img: [H, W, 3] float32, provides amplitude.

        Returns:
            [H, W, 3] float32, clipped to [0, 255].
        """
        result = np.zeros_like(spatial_img)
        for c in range(3):
            f_spatial = np.fft.fft2(spatial_img[:, :, c])
            f_amp_source = np.fft.fft2(amp_source_img[:, :, c])

            amp_new = np.abs(f_amp_source)
            phase_keep = np.angle(f_spatial)

            f_mixed = amp_new * np.exp(1j * phase_keep)
            img_mixed = np.fft.ifft2(f_mixed).real

            result[:, :, c] = np.clip(img_mixed, 0, 255)

        return result


class Scale(object):
    """Resize image and label to a fixed size"""

    def __init__(self, wi, he):
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        img = cv2.resize(img, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]


class Resize(object):
    """Resize with random short-side scaling and max-size constraint"""

    def __init__(self, min_size, max_size, strict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.strict = strict

    def get_size(self, image_size):
        w, h = image_size

        if not self.strict:
            size = random.choice(self.min_size)

            if self.max_size is not None:
                min_org = float(min(w, h))
                max_org = float(max(w, h))
                if max_org / min_org * size > self.max_size:
                    size = int(round(self.max_size * min_org / max_org))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                return (int(size * h / w), size)
            else:
                return (size, int(size * w / h))
        else:
            if w < h:
                return (self.max_size, self.min_size[0])
            else:
                return (self.min_size[0], self.max_size)

    def __call__(self, image, label):
        size = self.get_size(image.shape[:2])
        image = cv2.resize(image, size)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return (image, label)


class RandomCropResize(object):
    """Random crop followed by resize back to original size"""

    def __init__(self, crop_area):
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop

        return [img, label]


class RandomFlip(object):
    """Random horizontal and vertical flip"""

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return [image, label]


class RandomExchange(object):
    """Randomly swap pre- and post-event images (for bi-temporal input)"""

    def __call__(self, image, label):
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = numpy.concatenate((post_img, pre_img), axis=2)
        return [image, label]


class Normalize(object):
    """Normalize image using dataset mean and std"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32) / 255.0
        label = np.ceil(label / 255)

        for i in range(6):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]

        return [image, label]


class GaussianNoise(object):
    """Add Gaussian noise to image"""

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, image, label):
        noise = np.random.normal(0, self.std, size=image.shape)
        image = image + noise.astype(np.float32)
        return [image, label]


class ToTensor(object):
    """Convert numpy arrays to PyTorch tensors.

    Args:
        scale: downsample factor for label (default 1 = no downsampling).
        color_order: 'legacy' (old bug: whole-array reverse swaps T1/T2) or
                     'fixed' (correct per-image BGR→RGB). Default 'fixed'.
    """

    def __init__(self, scale=1, color_order='fixed'):
        self.scale = scale
        if color_order not in ('legacy', 'fixed'):
            raise ValueError(f"color_order must be 'legacy' or 'fixed', got {color_order!r}")
        self.color_order = color_order

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (w, h))
            label = cv2.resize(
                label,
                (int(w / self.scale), int(h / self.scale)),
                interpolation=cv2.INTER_NEAREST
            )

        # color_order='legacy': old behavior (whole-array reverse, swaps T1/T2)
        # color_order='fixed': per-image BGR→RGB (correct)
        if self.color_order == 'legacy':
            image = image[:, :, ::-1].copy()      # [B1,G1,R1,B2,G2,R2] → [R2,G2,B2,R1,G1,B1]
        else:
            # Correct: reverse each 3-channel image independently
            t1 = image[:, :, 0:3][:, :, ::-1]     # BGR→RGB for T1
            t2 = image[:, :, 3:6][:, :, ::-1]     # BGR→RGB for T2
            image = np.concatenate([t1, t2], axis=2).copy()

        image = image.transpose((2, 0, 1))    # HWC → CHW

        image_tensor = torch.from_numpy(image)
        label_tensor = torch.LongTensor(
            np.array(label, dtype=np.int_)
        ).unsqueeze(0)

        return [image_tensor, label_tensor]


class Compose(object):
    """Compose multiple transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

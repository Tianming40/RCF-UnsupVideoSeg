# From the original objectness
import mmcv
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from numpy import random
from mmcv.image.colorspace import convert_color_factory

class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 interpolation='bilinear',
                 override_scale=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            #assert len(self.img_scale) == 1
            pass
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.override_scale = override_scale

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
            #scale, scale_idx = self.random_sample_ratio(results['img'][0].shape[:2], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        resized_imgs=[]
        if self.keep_ratio:
            if results['scale'] == (-1, -1):
                img = results['img']
            else:
                for _img in results['img']:
                    #print("img before _img.shape={}".format(_img.shape))
                    _img, scale_factor = mmcv.imrescale(_img, results['scale'], return_scale=True, interpolation=self.interpolation)
                    #print("img _img.shape={}".format(_img.shape))
                    resized_imgs.append(_img)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = resized_imgs[0].shape[:2]
            h, w = results['img'][0].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            for _img in results['img']:
                _img, w_scale, h_scale = mmcv.imresize(
                    _img, results['scale'], return_scale=True, interpolation=self.interpolation)
                resized_imgs.append(_img)

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = resized_imgs
        results['img_shape'] = resized_imgs[0].shape
        results['pad_shape'] = resized_imgs[0].shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            for i in range(len(results[key])):
                if self.keep_ratio:
                    # If it's not 1, then we have more than 1 frame and need to resize all of the frames.
                    if results['scale'] == (-1, -1):
                        gt_seg = results[key][i]
                    else:
                        #print("anno before _img.shape={}".format(results[key][i].shape))
                        gt_seg = mmcv.imrescale(
                            results[key][i], results['scale'], interpolation='nearest')
                        #print("anno _img.shape={}".format(gt_seg.shape))
                else:
                    gt_seg = mmcv.imresize(
                        results[key][i], results['scale'], interpolation='nearest')
                #results['gt_semantic_seg'] = gt_seg
                results[key][i] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """
        # If override_scale is set, we generate scale according to args to replace the scale in results['scale'].
        if ('scale' not in results) or self.override_scale:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str



class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
        flow_fields (Sequence[str], optional): names (within seg_fields) that
            hold optical-flow vector data (last dim = (x, y) displacement),
            as opposed to plain scalar/label maps (e.g. pl_masks). Mirroring
            an array only rearranges pixel POSITIONS; a flow field's VECTOR
            VALUES must also be corrected -- horizontal flip negates the x
            (first) channel, vertical flip negates the y (second) channel.
            Fields not listed here (default: none) are mirrored positionally
            only, matching the original behaviour exactly -- this is correct
            for plain masks, which have no directional semantics.
    """

    def __init__(self, flip_ratio=None, direction='horizontal', flow_fields=()):
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flow_fields = set(flow_fields)
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            #results['img'] = mmcv.imflip(results['img'], direction=results['flip_direction'])

            flipped_list = []
            for _img in results['img']:
                flip_img = mmcv.imflip(_img, direction=results['flip_direction'])
                flipped_list.append(flip_img)
            results['img'] = flipped_list

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                for i in range(len(results[key])):
                    flipped = mmcv.imflip(
                        results[key][i], direction=results['flip_direction']).copy()
                    if key in self.flow_fields:
                        # Mirroring only rearranges pixel positions -- a flow
                        # vector's VALUE must also be negated along the
                        # mirrored axis (horizontal flip -> negate x/channel 0,
                        # vertical flip -> negate y/channel 1), or the
                        # supervision points in the wrong direction for every
                        # flipped sample.
                        chan = 0 if results['flip_direction'] == 'horizontal' else 1
                        flipped[..., chan] *= -1
                    results[key][i] = flipped
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'



class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for img_idx in range(len(results['img'])):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results['img'][img_idx], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results['img'][img_idx], self.size_divisor, pad_val=self.pad_val)
            results['img'][img_idx] = padded_img
        # The statistics follow the last image (the padding should be the same for image pairs)
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            for i in range(len(results[key])):
                results[key][i] = mmcv.impad(
                    results[key][i],
                    shape=results['pad_shape'][:2],
                    pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str



class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        normed_list = []
        for _img in results['img']:
            _img = mmcv.imnormalize(_img, self.mean, self.std, self.to_rgb)
            normed_list.append(_img)
        results['img'] = np.asarray(normed_list)
        #results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str



class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        cropped=[]


        for i, img in enumerate(results['img']):
            if img.shape[0] < self.crop_size[0] or img.shape[1] < self.crop_size[1]:
                print("img.shape={}".format(img.shape))
                print("self.crop_size={}".format(self.crop_size))
            if img.shape[0] < self.crop_size[0]:
                results['img'][i] = mmcv.imrescale(img, (2000, self.crop_size[0]))

        for key in results.get('seg_fields', []):
            for i in range(len(results[key])):
                if results[key][i].shape[0] < self.crop_size[0]:
                    results[key][i] = mmcv.imrescale(results[key][i], (2000, self.crop_size[0]))

        crop_bbox = self.get_crop_bbox(results['img'][0])
        for img in results['img']:
            assert self.cat_max_ratio == 1.
            # crop the image
            #print("before img.shape={}".format(img.shape))
            img = self.crop(img, crop_bbox)
            #if img.shape[-3:-1] != self.crop_size:
            #    print('Warn:', img.shape)
            #    img = img
            #    print('after Warn:', img.shape)

            #print("after img.shape={}".format(img.shape))
            img_shape = img.shape
            cropped.append(img)

        results['img'] = cropped

        # crop semantic seg
        for key in results.get('seg_fields', []):
            for i in range(len(results[key])):
                # print("before results[key][i].shape={}".format(results[key][i].shape))
                results[key][i] = self.crop(results[key][i], crop_bbox)
                # print("after results[key][i].shape={}".format(results[key][i].shape))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

class CenterCrop(RandomCrop):
    def get_crop_bbox(self, img):
        """Get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = margin_h // 2
        offset_w = margin_w // 2
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2


class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'

rgb2hsv = convert_color_factory('rgb', 'hsv')
hsv2rgb = convert_color_factory('hsv', 'rgb')

class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.

    Changes to the original PhotoMetricDistortion:
    1. Support multiple images in imgs (each pair is augmented the same)
    2. Assume BGR instead of RGB
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert_one_img(self, img, alpha=1, beta=0):
        """Multiply with alpha and add beta with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def convert(self, img, **kwargs):
        return [self.convert_one_img(img_item, **kwargs) for img_item in img]

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation_one_img(self, img, alpha):
        img = rgb2hsv(img)
        img[:, :, 1] = self.convert_one_img(img[:, :, 1], alpha=alpha)
        img = hsv2rgb(img)

        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            alpha = random.uniform(self.saturation_lower,
                                   self.saturation_upper)
            img = [self.saturation_one_img(img_item, alpha) for img_item in img]
            
        return img

    def hue_one_img(self, img, delta):
        img = rgb2hsv(img)
        img[:, :, 0] = (img[:, :, 0].astype(int) + delta) % 180
        img = hsv2rgb(img)
        
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            delta = random.uniform(-self.hue_delta, self.hue_delta)
            img = [self.hue_one_img(img_item, delta) for img_item in img]
        
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        # img should be a list (of usually two items, each is an array that represent an image in the frame)
        # Each item has range from 0 to 255 (not 1.0).
        img = results['img']

        # print(1, np.array(img).max(), np.array(img).min())

        # random brightness
        img = self.brightness(img)

        # print(2, np.array(img).max(), np.array(img).min())

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # print(3, np.array(img).max(), np.array(img).min())

        # random saturation
        img = self.saturation(img)

        # print(4, np.array(img).max(), np.array(img).min())

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        # print(np.array(img).max(), np.array(img).min())

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_semantic_seg".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'

def load_pipeline_item(item):
    # Note that this changes the item
    if isinstance(item, dict):
        item_type = item.pop('type')
        item_obj = globals()[item_type](**item)
        item['type'] = item_type
        return item_obj
    return item

# This is for a list of items
class ApplyIndividually(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        return [self.transform(item) for item in data]

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self.transform})"

from torchvision import transforms

class NumpyToTensor(object):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        # It's 3 dimensional because we apply this in each image.
        for key in self.keys:
            for key_idx in range(len(data[key])):
                data[key][key_idx] = torch.tensor(
                    data[key][key_idx].transpose(2, 0, 1).astype(np.float32) / 255.
                )

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.keys})"

class AnnotationTransform(object):
    def __call__(self, data):
        # data['ann']: one PIL image
        # The RGB from annotation has 3 channels but they are the same, so we select the first channel.
        data['ann'] = np.array(data['ann'])[..., 0]
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class AnnResize(object):
    """Resize data['ann'] (PIL image) to a fixed (h, w) before AnnotationTransform.
    Needed when batch_size > 1 and images have varying original sizes, since
    data.py intentionally excludes 'ann' from seg_fields.
    """
    def __init__(self, size):
        self.h, self.w = size[0], size[1]

    def __call__(self, data):
        if 'ann' in data and data['ann'] is not None:
            ann = data['ann']
            if hasattr(ann, 'resize'):  # PIL Image
                data['ann'] = ann.resize((self.w, self.h), Image.NEAREST)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size=({self.h}, {self.w}))"

class AttnTransform(object):
    def __call__(self, data):
        # data['attn']: one numpy array
        # The mask does not have channel dimension so we need to unsqueeze the first dimension.
        # Different from annotation, we need to resize and crop attention.
        for idx in range(len(data['attn'])):
            data['attn'][idx] = torch.tensor(data['attn'][idx][None, ...])
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class FlowTransform(object):
    def __init__(self, flow_fields, scale_flow=False):
        super().__init__()
        self.flow_fields = flow_fields
        self.scale_flow = scale_flow
    
    def __call__(self, data):
        # data with key: numpy array with dimension (480, 854, 2)
        # We do not need to divide by 255 in the flow.
        for key in self.flow_fields:
            # Need to implement if len(data[key]) != 1
            assert len(data[key]) == 1, f"{len(data[key])} != 1"
            flow = data[key][0]
            if self.scale_flow:
                # Image pairs in the same data item are scaled the same
                # scale_factor: w, h
                scale_factor = data['scale_factor'][:2]
                # flow: [H, W, (x, y)]
                flow *= scale_factor
            data[key][0] = torch.tensor(np.transpose(flow, (2, 0, 1)))
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flow_fields={self.flow_fields})"

class TorchNormalize(object):
    # From transforms.Normalize
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data):
        for key_idx in range(len(data['img'])):
            data['img'][key_idx] = TF.normalize(data['img'][key_idx], self.mean, self.std, self.inplace)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class PLTransform(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        for idx in range(len(data['pl_masks'])):
            data['pl_masks'][idx] = torch.tensor(data['pl_masks'][idx]) / 255.
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class ResolutionAwareCrop:
    """Resize + RandomCrop with per-resolution settings.

    configs: list of dicts with keys:
        max_short_side  (int)   — apply this entry if image short side <= value
        resize_short    (int)   — target short side for Resize
        crop_size       (list)  — [h, w] crop size
        split_wide      (bool)  — optional; randomly split wide images (AR>1.5)
                                  left/right before resize, making them portrait

    Entries are matched in ascending max_short_side order; the last entry
    acts as the default (largest / catch-all).

    Example config (720p + 1080p with split):
        - {max_short_side: 576,   resize_short: 400, crop_size: [384, 384]}
        - {max_short_side: 99999, resize_short: 400, crop_size: [384, 384], split_wide: true}
    """
    def __init__(self, configs):
        self.configs = sorted(configs, key=lambda c: c['max_short_side'])

    @staticmethod
    def _split_wide(results):
        """Randomly take left or right half; updates img + all seg_fields."""
        imgs = results['img']
        h, w = imgs[0].shape[:2]
        half_w = w // 2
        start = 0 if random.random() < 0.5 else half_w
        end = half_w if start == 0 else w

        results['img'] = [img[:, start:end] for img in imgs]
        for key in results.get('seg_fields', []):
            results[key] = [arr[:, start:end] for arr in results[key]]
        return results

    @staticmethod
    def _half_crop(results, crop_size):
        """Crop from left or right half (50/50). Flow stays fully consistent
        because we crop img and seg_fields from the same region of the full image."""
        h, w = results['img'][0].shape[:2]
        crop_h, crop_w = crop_size
        half_w = w // 2

        if random.random() < 0.5:
            x_lo, x_hi = 0, half_w          # left half
        else:
            x_lo, x_hi = half_w, w          # right half

        margin_y = max(h - crop_h, 0)
        margin_x = max((x_hi - x_lo) - crop_w, 0)
        y0 = np.random.randint(0, margin_y + 1)
        x0 = x_lo + np.random.randint(0, margin_x + 1)

        def crop(arr):
            return arr[y0:y0 + crop_h, x0:x0 + crop_w]

        results['img'] = [crop(img) for img in results['img']]
        for key in results.get('seg_fields', []):
            results[key] = [crop(arr) for arr in results[key]]
        return results

    def __call__(self, results):
        h, w = results['img'][0].shape[:2]
        short = min(h, w)
        cfg = self.configs[-1]
        for c in self.configs:
            if short <= c['max_short_side']:
                cfg = c
                break
        if cfg.get('split_wide', False) and w / h > 1.5:
            results = self._split_wide(results)
        results = Resize(img_scale=(9999, cfg['resize_short']), ratio_range=(0.96, 1.0))(results)
        if cfg.get('half_crop', False):
            results = self._half_crop(results, tuple(cfg['crop_size']))
        else:
            results = RandomCrop(crop_size=tuple(cfg['crop_size']), cat_max_ratio=1.0)(results)
        return results

    def __repr__(self):
        return f"ResolutionAwareCrop(configs={self.configs})"


class ResolutionGroupedBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that groups samples by resolution and interleaves batches.

    Each batch contains only same-resolution samples → no padding needed.
    Batches from different groups are shuffled together so the model sees
    all resolutions throughout training without distribution shift.

    Args:
        dataset: VideoDataset instance (needs seq_frames_path_all, seq_len_cumsum, num_seq)
        batch_size: samples per batch
        resolution_configs: same list as ResolutionAwareCrop — each entry has max_short_side
    """
    def __init__(self, dataset, batch_size, resolution_configs):
        from PIL import Image as _Image

        configs = sorted(resolution_configs, key=lambda c: c['max_short_side'])
        self.batch_size = batch_size
        self.groups = [[] for _ in range(len(configs))]

        for seq_idx in range(dataset.num_seq):
            first_path = dataset.seq_frames_path_all[seq_idx][0]
            w, h = _Image.open(first_path).size  # PIL: (width, height)
            short = min(h, w)

            group_idx = len(configs) - 1
            for cfg_idx, cfg in enumerate(configs):
                if short <= cfg['max_short_side']:
                    group_idx = cfg_idx
                    break

            start = int(dataset.seq_len_cumsum[seq_idx])
            end = int(dataset.seq_len_cumsum[seq_idx + 1])
            self.groups[group_idx].extend(range(start, end))

        sizes = [len(g) for g in self.groups]
        print(f"[ResolutionGroupedBatchSampler] group sizes={sizes}, "
              f"batches/epoch={sum(s // batch_size for s in sizes)}")

    def __iter__(self):
        import random
        all_batches = []
        for group in self.groups:
            indices = list(group)
            random.shuffle(indices)
            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                all_batches.append(indices[i:i + self.batch_size])
        random.shuffle(all_batches)  # interleave across resolutions
        yield from all_batches

    def __len__(self):
        return sum(len(g) // self.batch_size for g in self.groups)


def pad_collate(batch):
    """Collate that zero-pads all spatial tensors to the max (H, W) in the batch.

    Required when resolution_crop_configs produces different crop sizes per item.
    """
    import torch
    from torch.utils.data.dataloader import default_collate

    max_h = max_w = 0
    for item in batch:
        for val in item.values():
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        max_h = max(max_h, v.shape[-2])
                        max_w = max(max_w, v.shape[-1])

    if max_h == 0:
        return default_collate(batch)

    def _pad(t):
        if isinstance(t, torch.Tensor) and t.dim() >= 2:
            h, w = t.shape[-2], t.shape[-1]
            if h < max_h or w < max_w:
                out = torch.zeros(*t.shape[:-2], max_h, max_w, dtype=t.dtype)
                out[..., :h, :w] = t
                return out
        return t

    padded = []
    for item in batch:
        padded.append({k: [_pad(v) for v in val] if isinstance(val, list) else val
                       for k, val in item.items()})
    return default_collate(padded)


class Transform(object):
    """
    This is the transform used by v1 and v2 (v2.1).
    Training and evaluation have different shape (training is rectangular).
    """
    def __init__(self, training, strong_aug=False, has_flow=True, has_attn=False, has_pl=False,
                 scale_flow=False, test_crop_size=None, resolution_crop_configs=None,
                 resize_short=400, crop_size=None):
        # v1 by default uses weak_aug
        self.training = training
        self.has_pl = has_pl
        normalize_kwargs = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        if self.training:
            if resolution_crop_configs:
                resize_crop = [ResolutionAwareCrop(resolution_crop_configs)]
            else:
                _crop = tuple(crop_size) if crop_size else (384, 384)
                resize_crop = [
                    Resize(img_scale=(9999, resize_short), ratio_range=(0.96, 1.0)),
                    RandomCrop(crop_size=_crop, cat_max_ratio=1.0),
                ]
            self.group_transform = transforms.Compose([
                *([AttnTransform()] if has_attn else []),
                *resize_crop,
                *([
                    RandomFlip(flip_ratio=0.5, direction='horizontal',
                              flow_fields=['gt_fw_flows', 'gt_bw_flows'] if has_flow else ()),
                    PhotoMetricDistortion()
                ] if strong_aug else []),
                *([FlowTransform(['gt_fw_flows', 'gt_bw_flows'], scale_flow=scale_flow)] if has_flow else []),
                *([PLTransform()] if has_pl else []),
                NumpyToTensor(['img']),
                TorchNormalize(**normalize_kwargs)
            ])
        else:
            # test_crop_size: fixed-size resize (keep_ratio=False) applied BEFORE
            # AnnotationTransform so that both img and ann are resized together.
            if test_crop_size is not None:
                test_resize = Resize(img_scale=tuple(test_crop_size), keep_ratio=False)
            else:
                test_resize = Resize(img_scale=(9999, resize_short), ratio_range=(0.98, 0.98))
            self.group_transform = transforms.Compose([
                test_resize,
                *([AnnResize(test_crop_size)] if test_crop_size is not None else []),
                AnnotationTransform(),
                NumpyToTensor(['img']),
                TorchNormalize(**normalize_kwargs)
            ])
        
    def __call__(self, data):
        data['img'] = [np.asarray(item) for item in data['imgs']]
        # Resize and RandomCrop uses img
        data = self.group_transform(data)
        data['imgs'] = data.pop('img')

        # Remove the item with None
        data.pop('scale_idx')
        return data

    def __repr__(self):
        return str(self.group_transform)

def get_transform(args, training):
    transform_kwargs = args.train_transform_kwargs if training else args.test_transform_kwargs
    transform_cls = getattr(args, "transform_cls", "Transform")
    return globals()[transform_cls](training=training, **transform_kwargs)

if __name__ == "__main__":
    transform = get_transform(training=True)
    print(transform)

    transform = get_transform(training=False)
    print(transform)

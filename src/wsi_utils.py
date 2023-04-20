import math
import openslide
import numpy as np
import tensorflow as tf
from matplotlib import cm
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
from PIL import Image, ImageFilter, ImageDraw


class SlideManager:
    def __init__(self, tile_size, level=0, overlap=1, verbose=0):
        """
        # SlideManager provides an easy way to generate a cropList object.
        # This object is not tied to a particular slide and can be reused to crop many slides using the same settings.
        """
        self.tile_size = tile_size
        self.overlap = int(1/overlap)
        self.verbose = verbose
        self.level = level

    def __generateSections__(self,
                             x_start,
                             y_start,
                             width,
                             height,
                             downscaling_factor,
                             filepath):
        side = self.tile_size
        step = int(side / self.overlap)
        self.__sections__ = []

        n_tiles = 0
        # N.B. Tiles are considered in the 0 level
        for y in range(int(math.floor(height / step))):
            for x in range(int(math.floor(width / step))):
                # x * step + side is right margin of the given tile
                if x * step + side > width or y * step + side > height:
                    continue
                n_tiles += 1
                self.__sections__.append(
                    {'top': y_start + step * y, 'left': x_start + step * x,
                     'size': math.floor(side / downscaling_factor)})
        if self.verbose:
            print("-"*len("{} stats:".format(filepath)))
            print("{} stats:".format(filepath))
            print("step: {}".format(step))
            print("y: {}".format(y_start))
            print("x: {}".format(x_start))
            print("slide width {}".format(width))
            print("slide height {}".format(height))
            print("downscaling factor: {}".format(downscaling_factor))
            print("# of tiles:{}".format(n_tiles))
            print("-" * len("{} stats:".format(filepath)))

    def crop(self, filepath_slide, label=None):
        slide = openslide.OpenSlide(filepath_slide)
        downscaling = slide.level_downsamples[0]
        if 'openslide.bounds-width' in slide.properties.keys():
            bounds_width = int(slide.properties['openslide.bounds-width'])
            bounds_height = int(slide.properties['openslide.bounds-height'])
            bounds_x = int(slide.properties['openslide.bounds-x'])
            bounds_y = int(slide.properties['openslide.bounds-y'])
        else:
            bounds_width = slide.level_dimensions[self.level][0]
            bounds_height = slide.level_dimensions[self.level][1]
            bounds_x = 0
            bounds_y = 0

        self.__generateSections__(bounds_x,
                                  bounds_y,
                                  bounds_width,
                                  bounds_height,
                                  downscaling,
                                  filepath_slide)
        indexes = self.__sections__
        for index in indexes:
            index['filepath_slide'] = filepath_slide
            index['level'] = self.level
            index['label'] = label
        return indexes


class DatasetManager:
    def __init__(self,
                 filepaths,
                 labels,
                 tile_size,
                 scaling_factor_tissue=1/50,
                 annotations=(),
                 tile_new_size=None,
                 num_classes=None,
                 overlap=1,
                 channels=3,
                 one_hot=True,
                 tissue_percentage_min=0.75,
                 eosin_percentage_min=0.9,
                 random_sampling_fraction=1.0,
                 verbose=0):

        # Note that we work always at level 0. Increasing the tile size at same level
        # will provide tiles at a bigger magnification factor without being tied to
        # the downscaling levels of the given slide (see openslide).
        self.level = 0
        self.scaling_factor_tissue = scaling_factor_tissue
        if tile_new_size:
            self.new_size = tile_new_size
        else:
            self.new_size = tile_size
        self.tile_size = tile_size
        self.one_hot = one_hot
        if len(annotations) > 0:
            self.annotations = annotations
        self.overlap = overlap
        self.tissue_percentage_min = tissue_percentage_min
        self.eosin_percentage_min = eosin_percentage_min
        if num_classes is None:
            self.num_classes = len(set(labels))
        else:
            self.num_classes = num_classes
        self.channels = channels
        self.section_manager = SlideManager(tile_size, overlap=self.overlap, level=self.level, verbose=verbose)
        self.tile_placeholders = list()
        self.tile_placeholder_list = list()
        self.slide_list = list()
        self.tissue_segmentations = dict()
        self.tissue_segmentations['grayscale'] = list()
        self.tissue_segmentations['hed'] = list()
        self.random_sampling_fraction = random_sampling_fraction
        if random_sampling_fraction < 1:
            print("Performing random sampling.")
        for filepath, label in zip(filepaths, labels):
            tile_placeholder = self.section_manager.crop(
                filepath,
                label=label)
            tile_placeholder, mask_grayscale, mask_hed = self.compute_tissue_percentage(tile_placeholder)
            self.tissue_segmentations['grayscale'].append(mask_grayscale)
            self.tissue_segmentations['hed'].append(mask_hed)
            if random_sampling_fraction < 1:
                self.tile_placeholder_list.append(self._random_sampling_by_slide(
                    self._filter_tissue(tile_placeholder)))
                self.tile_placeholders += self._random_sampling_by_slide(self._filter_tissue(tile_placeholder))
            else:
                self.tile_placeholder_list.append(self._filter_tissue(tile_placeholder))
                self.tile_placeholders += self._filter_tissue(tile_placeholder)
            self.slide_list.append(openslide.OpenSlide(filepath))

        print("Found {} tissue tiles belonging to {} slides".format(len(self.tile_placeholders),
                                                                    len(filepaths)))

    def _random_sampling_by_slide(self, tile_placeholder):
        size = int(np.floor(self.random_sampling_fraction * len(tile_placeholder)))
        return np.random.choice(tile_placeholder, size, replace=False).tolist()
    def _filter_tissue(self, tile_placeholder):
        tissue = list(filter(lambda x: x["tissue_percentage"] >= self.tissue_percentage_min, tile_placeholder))
        eosin = list(filter(lambda x: x["eosin_percentage"] >= self.eosin_percentage_min, tissue))
        return eosin

    def _to_image(self, x):
        slide = openslide.OpenSlide(self.tile_placeholders[x.numpy()]['filepath_slide'])
        pil_object = slide.read_region([self.tile_placeholders[x.numpy()]['left'],
                                        self.tile_placeholders[x.numpy()]['top']],
                                       self.tile_placeholders[x.numpy()]['level'],
                                       [self.tile_placeholders[x.numpy()]['size'],
                                        self.tile_placeholders[x.numpy()]['size']])
        pil_object = pil_object.convert('RGB')
        pil_object = pil_object.resize(size=(self.new_size, self.new_size))
        label = self.tile_placeholders[x.numpy()]['label']
        im_size = pil_object.size
        img = tf.reshape(tf.cast(pil_object.getdata(), dtype=tf.uint8), (im_size[0], im_size[1], 3))
        return tf.image.convert_image_dtype(img, dtype=tf.float32), tf.cast(label, tf.float32)

    def _to_one_hot(self, image, label):
        return image, tf.cast(tf.one_hot(tf.cast(label, tf.int32),
                                         self.num_classes,
                                         name='label', axis=-1),
                              tf.float32)

    def _fixup_shape(self, image, label):
        """
        Tensor.shape is determined at graph build time (tf.shape(tensor) gets you the runtime shape).
        In tf.numpy_function/tf.py_function “don’t build a graph for this part, just run it in python”.
        So none of the code in such functions runs during graph building, and TensorFlow does not know the shape in there.
        With the function _fixup_shape we set the shape of the tensors.
        """
        image.set_shape([self.new_size,
                         self.new_size,
                         self.channels])
        if self.one_hot:
            label.set_shape([self.num_classes])
        else:
            label.set_shape([])
        return image, label

    def make_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices([i for i in range(len(self.tile_placeholders))])
        dataset = dataset.map(lambda x: tf.py_function(self._to_image, [x], Tout=[tf.float32, tf.float32]),
                              num_parallel_calls=8)
        if self.one_hot:
            dataset = dataset.map(self._to_one_hot)
        dataset = dataset.map(lambda x, y: self._fixup_shape(x, y))
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def compute_tissue_percentage(self, tile_placeholder):
        slide = openslide.OpenSlide(tile_placeholder[0]['filepath_slide'])
        if 'openslide.bounds-width' in slide.properties.keys():
            # Here to consider only the rectangle bounding the non-empty region of the slide, if available.
            # These properties are in the level 0 reference frame.
            bounds_width = int(slide.properties['openslide.bounds-width'])
            bounds_height = int(slide.properties['openslide.bounds-height'])
            bounds_x = int(slide.properties['openslide.bounds-x'])
            bounds_y = int(slide.properties['openslide.bounds-y'])

            region_lv0 = (bounds_x,
                          bounds_y,
                          bounds_width,
                          bounds_height)
        else:
            # If bounding box of the non-empty region of the slide is not available
            # Slide dimensions of given level reported to level 0
            region_lv0 = (0, 0, slide.level_dimensions[0][0], slide.level_dimensions[0][1])

        region_lv0 = [round(x) for x in region_lv0]
        region_lv_selected = [round(x * self.scaling_factor_tissue) for x in region_lv0]
        slide_image_grayscale = slide.get_thumbnail((region_lv_selected[2], region_lv_selected[3])).convert('L')
        slide_image_rgb = slide.get_thumbnail((region_lv_selected[2], region_lv_selected[3]))

        # H&E segmentation
        hed = sk_color.rgb2hed(slide_image_rgb)
        eosin = np.array(hed[:, :, 1])
        otsu_thresh = sk_filters.threshold_otsu(eosin)
        mask_eosin = np.zeros_like(eosin, dtype=int)
        mask_eosin[eosin > otsu_thresh] = 255
        mask_eosin = sk_morphology.remove_small_holes(mask_eosin, area_threshold=100, connectivity=8)
        mask_eosin = sk_morphology.remove_small_objects(mask_eosin, connectivity=2, min_size=100)

        # Grayscale segmentation
        otsu_thresh = sk_filters.threshold_otsu(np.array(slide_image_grayscale))
        mask_grayscale = np.zeros_like(slide_image_grayscale)
        mask_grayscale[slide_image_grayscale > otsu_thresh] = 255
        mask_grayscale = sk_morphology.remove_small_holes(mask_grayscale, area_threshold=100, connectivity=8)
        mask_grayscale = sk_morphology.remove_small_objects(mask_grayscale, connectivity=2, min_size=100)
        # Complement
        mask_grayscale = np.invert(mask_grayscale)

        for tile in tile_placeholder:
            top = math.ceil(tile['top'] * self.scaling_factor_tissue)
            left = math.ceil(tile['left'] * self.scaling_factor_tissue)
            side = math.ceil(tile['size'] * self.scaling_factor_tissue)
            top -= region_lv_selected[1]
            left -= region_lv_selected[0]
            side_x = side
            side_y = side
            if top < 0:
                side_y += top
                top = 0
            if left < 0:
                side_x += left
                left = 0
            if side_x > 0 and side_y > 0:
                # Grayscale
                portion = mask_grayscale[top:top + side_y, left:left + side_x]
                tissue_percentage = portion.sum() / (portion.shape[0] * portion.shape[1])
                tile['tissue_percentage'] = tissue_percentage
                # H&E
                portion = mask_eosin[top:top + side_y, left:left + side_x]
                eosin_percentage = portion.sum() / (portion.shape[0] * portion.shape[1])
                tile['eosin_percentage'] = eosin_percentage
        return tile_placeholder, mask_grayscale, mask_eosin

def set_label_from_mask(tile_placeholder,
                        mask,
                        scaling_factor,
                        label_value):
    """
    Assign label to tile in tile_placeholders based on a binary mask.
    If mask[x, y] != 0 the corresponding tile (see np.ravel_multi_index) sets 'label' key to label_value
    """
    tile_size = scaling_factor * tile_placeholder[0]['size']
    for x_coord in range(mask.shape[0]):
        for y_coord in range(mask.shape[1]):
            if mask[x_coord, y_coord] != 0:
                matrix_index = np.array([int(x_coord / tile_size), int(y_coord / tile_size)])
                linear_index = np.ravel_multi_index(matrix_index, (int(mask.shape[0] / tile_size),
                                                                   int(mask.shape[1] / tile_size)))
                tile_placeholder[linear_index]['label'] = label_value


def set_mask_from_label(tile_placeholder,
                        scaling_factor,
                        label_value):
    """
    Build a binary mask where tile_placeholder[linear_index]['label'] == label_value.
    """
    filepath_slide = tile_placeholder[0]['filepath_slide']
    slide = openslide.OpenSlide(filepath_slide)
    if 'openslide.bounds-width' in slide.properties.keys():
        # Here to consider only the rectangle bounding the non-empty region of the slide, if available.
        # These properties are in the level 0 reference frame.
        bounds_width = int(slide.properties['openslide.bounds-width'])
        bounds_height = int(slide.properties['openslide.bounds-height'])
        bounds_x = int(slide.properties['openslide.bounds-x'])
        bounds_y = int(slide.properties['openslide.bounds-y'])

        region_lv0 = (bounds_x,
                      bounds_y,
                      bounds_width,
                      bounds_height)

    else:
        # If bounding box of the non-empty region of the slide is not available
        size = slide.level_dimensions[0]
        # Slide dimensions of given level reported to level 0
        region_lv0 = (0, 0, size[0], size[1])
    region_lv0 = [round(x) for x in region_lv0]
    region_lv0_scaled = [round(x * scaling_factor) for x in region_lv0]
    mask = np.zeros((region_lv0_scaled[3], region_lv0_scaled[2], 3), dtype=np.uint8)
    tile_size = scaling_factor * tile_placeholder[0]['size']
    for x_coord in range(mask.shape[0]):
        for y_coord in range(mask.shape[1]):
            matrix_index = np.array([int(x_coord / tile_size), int(y_coord / tile_size)])
            linear_index = np.ravel_multi_index(matrix_index, (int(mask.shape[0] / tile_size),
                                                               int(mask.shape[1] / tile_size)))
            if tile_placeholder[linear_index]['label'] == label_value:
                mask[x_coord, y_coord] = 255
    return mask


def get_labelled_image(tile_placeholder,
                       scaling_factor,
                       labels2colors):
    """
    Produces an image with different color corresponding to different labels.
    labels2colors is a dict mapping label to color, e.g.:
    {'first_label': [255, 0, 0],
     'second_label': [0, 255, 0]}
    """
    filepath_slide = tile_placeholder[0]['filepath_slide']
    slide = openslide.OpenSlide(filepath_slide)
    if 'openslide.bounds-width' in slide.properties.keys():
        # Here to consider only the rectangle bounding the non-empty region of the slide, if available.
        # These properties are in the level 0 reference frame.
        bounds_width = int(slide.properties['openslide.bounds-width'])
        bounds_height = int(slide.properties['openslide.bounds-height'])
        bounds_x = int(slide.properties['openslide.bounds-x'])
        bounds_y = int(slide.properties['openslide.bounds-y'])

        region_lv0 = (bounds_x,
                      bounds_y,
                      bounds_width,
                      bounds_height)

    else:
        # If bounding box of the non-empty region of the slide is not available
        size = slide.level_dimensions[0]
        # Slide dimensions of given level reported to level 0
        region_lv0 = (0, 0, size[0], size[1])
    region_lv0 = [round(x) for x in region_lv0]
    region_lv0_scaled = [round(x * scaling_factor) for x in region_lv0]
    output = np.zeros((region_lv0_scaled[3], region_lv0_scaled[2], 3), dtype=np.uint8)
    tile_size = scaling_factor * tile_placeholder[0]['size']
    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            matrix_index = np.array([int(x / tile_size), int(y / tile_size)])
            linear_index = np.ravel_multi_index(matrix_index, (int(output.shape[0] / tile_size),
                                                               int(output.shape[1] / tile_size)))
            output[x, y, 0] = labels2colors[tile_placeholder[linear_index]['label']][0]
            output[x, y, 1] = labels2colors[tile_placeholder[linear_index]['label']][1]
            output[x, y, 2] = labels2colors[tile_placeholder[linear_index]['label']][2]

    return output

def get_heatmap(tile_placeholders,
                slide,
                class_to_map=0,
                num_classes=1,
                scaling_factor=1 / 100,
                threshold=0.5,
                tile_placeholders_mapping_key='fake2showTissue',
                colormap=cm.get_cmap('Blues')):

    """
    Builds a 3 channel map.
    The first three channels represent the sum of all the probabilities of the crops which contain that pixel
    belonging to classes 0-1, the fourth hold the number of crops which contain it.
    """

    if 'openslide.bounds-width' in slide.properties.keys():
        # Here to consider only the rectangle bounding the non-empty region of the slide, if available.
        # These properties are in the level 0 reference frame.
        bounds_width = int(slide.properties['openslide.bounds-width'])
        bounds_height = int(slide.properties['openslide.bounds-height'])
        bounds_x = int(slide.properties['openslide.bounds-x'])
        bounds_y = int(slide.properties['openslide.bounds-y'])

        region_lv0 = (bounds_x,
                      bounds_y,
                      bounds_width,
                      bounds_height)
    else:
        # If bounding box of the non-empty region of the slide is not available
        # Slide dimensions of given level reported to level 0
        region_lv0 = (0, 0, slide.level_dimensions[0][0], slide.level_dimensions[0][1])

    region_lv0 = [round(x) for x in region_lv0]
    region_lv_selected = [round(x * scaling_factor) for x in region_lv0]
    probabilities = np.zeros((region_lv_selected[3], region_lv_selected[2], 3))
    for tile in tile_placeholders:
        top = math.ceil(tile['top'] * scaling_factor)
        left = math.ceil(tile['left'] * scaling_factor)
        side = math.ceil(tile['size'] * scaling_factor)
        top -= region_lv_selected[1]
        left -= region_lv_selected[0]
        side_x = side
        side_y = side
        if top < 0:
            side_y += top
            top = 0
        if left < 0:
            side_x += left
            left = 0
        if side_x > 0 and side_y > 0:
            try:
                probabilities[top:top + side_y, left:left + side_x, 0:num_classes] = np.array(
                    tile[tile_placeholders_mapping_key][class_to_map])
            except KeyError:
                raise KeyError(f"Class {class_to_map} not found.")

    probabilities = probabilities * 255
    probabilities = probabilities.astype('uint8')

    map_ = probabilities[:, :, class_to_map]
    map_ = Image.fromarray(map_).filter(ImageFilter.GaussianBlur(3))
    map_ = np.array(map_) / 255
    map_[map_ < threshold] = 0
    segmentation = (map_ * 255).astype('uint8')
    map_ = colormap(np.array(map_))
    roi_map = Image.fromarray((map_ * 255).astype('uint8'))
    roi_map.putalpha(75)

    slide_image = slide.get_thumbnail((region_lv_selected[2], region_lv_selected[3]))
    slide_image = slide_image.convert('RGBA')
    slide_image.alpha_composite(roi_map)
    slide_image.convert('RGBA')
    return slide_image, segmentation

def locate_tiles(tile_placeholders,
                 slide,
                 scaling_factor=1/50):

    if 'openslide.bounds-width' in slide.properties.keys():
        # Here to consider only the rectangle bounding the non-empty region of the slide, if available.
        # These properties are in the level 0 reference frame.
        bounds_width = int(slide.properties['openslide.bounds-width'])
        bounds_height = int(slide.properties['openslide.bounds-height'])
        bounds_x = int(slide.properties['openslide.bounds-x'])
        bounds_y = int(slide.properties['openslide.bounds-y'])

        region_lv0 = (bounds_x,
                      bounds_y,
                      bounds_width,
                      bounds_height)
    else:
        # If bounding box of the non-empty region of the slide is not available
        # Slide dimensions of given level reported to level 0
        region_lv0 = (0, 0, slide.level_dimensions[0][0], slide.level_dimensions[0][1])

    region_lv0 = [round(x) for x in region_lv0]
    region_lv_selected = [round(x * scaling_factor) for x in region_lv0]
    slide_image = slide.get_thumbnail((region_lv_selected[2], region_lv_selected[3]))
    slide_image.putalpha(180)
    draw = ImageDraw.Draw(slide_image)

    for tile in tile_placeholders:
        top = math.ceil(tile['top'] * scaling_factor)
        left = math.ceil(tile['left'] * scaling_factor)
        side = math.ceil(tile['size'] * scaling_factor)
        top -= region_lv_selected[1]
        left -= region_lv_selected[0]
        side_x = side
        side_y = side
        if top < 0:
            side_y += top
            top = 0
        if left < 0:
            side_x += left
            left = 0
        if side_x > 0 and side_y > 0:
            draw.rectangle([left, top, left + side, top + side], outline="firebrick", width=1)

    return slide_image

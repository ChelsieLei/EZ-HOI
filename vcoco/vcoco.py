"""
V-COCO dataset in Python3

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import itertools
import numpy as np

from typing import Optional, List, Callable, Tuple, Any, Dict
from pocket.data import ImageDataset, DataSubset

class VCOCOSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._keep[self.pool[idx]]]
    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        """Return the size (width, height) of an image"""
        return self.load_image(os.path.join(
            self._root,
            self.filename(self._keep[self.pool[idx]])
        )).size


class VCOCO(ImageDataset):
    """
    V-COCO dataset

    Parameters:
    -----------
    root: str
        Root directory where images are saved.
    anno_file: str
        Path to json annotation file.
    transform: callable
        A function/transform that  takes in an PIL image and returns a transformed version.
    target_transform: callble
        A function/transform that takes in the target and transforms it.
    transforms: callable
        A function/transform that takes input sample and its target as entry and 
        returns a transformed version.
    """
    def __init__(self, root: str, anno_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        super().__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = None
        self.num_action_cls = 24

        self._anno_file = anno_file

        # Compute metadata
        self._compute_metatdata(anno)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._keep)

    def __getitem__(self, i: int) -> Tuple[Any, Any]:
        """
        Parameters:
        -----------
        i: int
            The index to an image.
        
        Returns:
        --------
        image: Any
            Input Image. By default, when relevant transform arguments are None,
            the image is in the form of PIL.Image.
        target: Any
            The annotation associated with the given image. By default, when
            relevant transform arguments are None, the taget is a dict with the
            following keys:
                boxes_h: List[list]
                    Human bouding boxes in a human-object pair encoded as the top
                    left and bottom right corners
                boxes_o: List[list]
                    Object bounding boxes corresponding to the human boxes
                actions: List[int]
                    Ground truth action class for each human-object pair
                objects: List[int]
                    Object category index for each object in human-object pairs. The
                    indices follow the 80-class standard, where 0 means background and
                    1 means person.
        """
        image = self.load_image(os.path.join(
            self._root, self.filename(i)
        ))
        # print("-------", self._root)
        target = self._anno[self._keep[i]].copy()
        target.pop('file_name')
        return self._transforms(image, target),self.filename(i)

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._anno_file)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def actions(self) -> List[str]:
        """Return the list of actions"""
        return self._actions

    @property
    def objects(self) -> List[str]:
        """Return the list of objects"""
        return self._objects

    @property
    def present_objects(self) -> List[int]:
        """Return the list of objects that are present in the dataset partition"""
        return self._present_objects

    @property
    def num_instances(self) -> List[int]:
        """Return the number of human-object pairs for each action class"""
        return self._num_instances

    @property
    def action_to_object(self) -> List[list]:
        """Return the list of objects for each action"""
        return self._action_to_object

    @property
    def object_to_action(self) -> Dict[int, list]:
        """Return the list of actions for each object"""
        object_to_action = {obj: [] for obj in list(range(1, 81))}
        for act, obj in enumerate(self._action_to_object):
            for o in obj:
                if act not in object_to_action[o]:
                    object_to_action[o].append(act)
        return object_to_action

    def image_id(self, idx: int) -> int:
        """Return the COCO image ID"""
        return self._image_ids[self._keep[idx]]

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._anno[self._keep[idx]]['file_name']
    def split(self, ratio: float):
        perm = np.random.permutation(len(self._keep))
        n = int(len(perm) * ratio)
        return VCOCOSubset(self, perm[:n]), VCOCOSubset(self, perm[n:])

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self.load_image(os.path.join(
            self._root,
            self.filename(idx)
        )).size

    def _compute_metatdata(self, f: dict) -> None:
        self._anno = f['annotations']
        self._actions = f['classes']
        self._objects = f['objects']
        self._image_ids = f['images']
        self._action_to_object = f['action_to_object']

        keep = list(range(len(f['images'])))
        num_instances = [0 for _ in range(len(f['classes']))]
        valid_objects = [[] for _ in range(len(f['classes']))]
        for i, anno_in_image in enumerate(f['annotations']):
            # Remove images without human-object pairs
            if len(anno_in_image['actions']) == 0:
                keep.remove(i)
                continue
            for act, obj in zip(anno_in_image['actions'], anno_in_image['objects']):
                num_instances[act] += 1
                if obj not in valid_objects[act]:
                    valid_objects[act].append(obj)

        objects = list(itertools.chain.from_iterable(valid_objects))
        self._present_objects = np.unique(np.asarray(objects)).tolist()
        self._num_instances = num_instances
        self._keep = keep
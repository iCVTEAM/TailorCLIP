from fileinput import filename
import os
import subprocess
import json
from typing import Any, Dict, List, Optional, Set
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

from chinopie.datasets.multilabel import MultiLabelLocalDataset

from ... import logging
_logger=logging.get_logger(__name__)

def prepare_vg(root: str, phase: str, include_segmentations:bool=False):
    raise NotImplemented


def get_vg_labels(root: str) -> List[str]:
    with open(os.path.join(root, "vg_category_200_labels_index.json"), "r") as f:
        annotations = json.load(f)
    label_sets:list[str]=[]
    for i in annotations:
        label_sets.extend(annotations[i])
    return list(map(lambda x:str(x),sorted(list(set(label_sets)))))


class VG200Dataset(MultiLabelLocalDataset):
    img_list: List[Any]
    one_hot: bool

    def __init__(
        self,
        root: str,
        preprocess: Any,
        extra_preprocess:Optional[Any]=None,
        phase: str = "train",
        negatives_as_neg1=False,
        prepreprocess=None
    ):

        if os.path.exists(os.path.join(root, f"vg200_{phase}.cache.pth")):
            cache=torch.load(os.path.join(root, f"vg200_{phase}.cache.pth"))
            super().__init__(
                cache["img_paths"],
                cache["num_labels"],
                cache["annotations"],
                cache["labels"],
                preprocess,
                extra_preprocess,
                negatives_as_neg1,
                prepreprocess
            )
            return

        labels=get_vg_labels(root)
        num_labels=len(labels)
        _logger.info(f"[dataset] VG200 classification {phase} phase, {num_labels} classes: {labels[:40]}...")

        with open(os.path.join(root,"vg_category_200_labels_index.json"),"r") as f:
            t=json.load(f)
            _raw_list=[{
                'id':int(k.split('.')[0]),
                'filename':k,
                'labels':v
            } for k,v in t.items()]
            _raw_list.sort(key=lambda x:x['id'])
        _len=len(_raw_list)
        if phase=='train':
            _raw_list=_raw_list[:int(_len*0.7)]
        elif phase=='val':
            _raw_list=_raw_list[int(_len*0.7):]
        else:
            raise RuntimeError(f"don't know phase `{phase}`")
        
        img_paths=[]
        for i in _raw_list:
            possible_path=[
                os.path.join(root, f"VG_100K", i['filename']),
                os.path.join(root, f"VG_100K_2", i['filename'])
            ]
            rgb_image=None
            for path in possible_path:
                if os.path.exists(path):
                    rgb_image=path
                    break
            assert rgb_image is not None, f"image {filename} not found"
            img_paths.append(rgb_image)

        annotations:list[list[int]]=[]
        for img in _raw_list:
            annotations.append(img['labels'])

        _logger.info(
            f"[dataset] VG200 classification {phase} phase, {num_labels} classes, {len(img_paths)} images"
        )

        torch.save({
            "img_paths": img_paths,
            "annotations": annotations,
            "labels": labels,
            "num_labels": num_labels
        }, os.path.join(root, f"vg200_{phase}.cache.pth"))

        super().__init__(
            img_paths,
            num_labels,
            annotations,
            labels,
            preprocess,
            extra_preprocess,
            negatives_as_neg1,
            prepreprocess
        )
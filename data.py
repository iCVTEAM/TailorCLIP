import random
from typing import Any, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import default_collate
from torchvision import transforms

from PIL import Image
from chinopie.filehelper import GlobalFileHelper
from chinopie.datasets.multilabel import MultiLabelLocalDataset
from utils import clamp_as_hard_label

def custom_collate(x):
    overlay_indices=[v['overlay_indices'] for v in x]
    for v in x:del v['overlay_indices']
    res=default_collate(x)
    res['overlay_indices']=overlay_indices
    return res

class RandomInterpolationDataset(Dataset):
    def __init__(self,dataset:MultiLabelLocalDataset,overlay_dataset:MultiLabelLocalDataset,num_overlay:int,scales:List[float],p:List[float],preprocess:Any,extra_preprocess:Any):
        self._dataset=dataset
        self._overlay_dataset=overlay_dataset
        self._scales=scales
        assert num_overlay>=1
        self._num_overlay=num_overlay
        self._p=p
        self._preprocess=preprocess
        self._extra_preprocess=extra_preprocess
        
        # replace the transforms with identical one
        self._dataset._preprocess=lambda x:x.copy()
        # replace the transforms with identical one
        self._overlay_dataset._preprocess=lambda x:x.copy()
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index):
        item=self._dataset[index]
        p=random.random()
        num_overlay=-1
        for k,v in enumerate(self._p):
            if p<=sum(self._p[:k+1]):
                num_overlay=k+1
                break
        assert num_overlay>=1,"invalid possibilities"
        overlay_indices=[random.randint(0,len(self._overlay_dataset)-1) for _ in range(num_overlay)]

        image:Image.Image=item['image'] # type: ignore
        overlay_targets=[]
        for overlay_index in overlay_indices:
            overlay_item=self._overlay_dataset[overlay_index]
            overlay_targets.append(overlay_item['target'])

            max_width,max_height=image.width*self._scales[num_overlay-1],image.height*self._scales[num_overlay-1]
            overlay_image:Image.Image=overlay_item['image'] # type: ignore
            real_scale=min(max_width/overlay_image.width,max_height/overlay_image.height)
            overlay_width,overlay_height=int(overlay_image.width*real_scale),int(overlay_image.height*real_scale)
            overlay_image=overlay_image.resize((overlay_width,overlay_height))
            rand_x=random.randint(0,image.width-overlay_width)
            rand_y=random.randint(0,image.height-overlay_height)
            image.paste(overlay_image,(rand_x,rand_y))

        tensored_image=self._preprocess(image)
        extra_image=self._extra_preprocess(image)

        # handle hard label
        overlay_targets=torch.stack(overlay_targets,dim=0)
        hard_target=clamp_as_hard_label(torch.cat([item['target'].unsqueeze(0),overlay_targets],dim=0)).sum(dim=0).clamp_max(1)
        

        return {
            'index':index,
            'overlay_indices':overlay_indices,
            'image':tensored_image,
            'extra_image':extra_image,
            'target':item['target'],
            'overlay_target':overlay_targets.max(dim=0)[0],
            'hard_target':hard_target,
            'name':item['name'],
        }


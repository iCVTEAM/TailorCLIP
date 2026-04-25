from typing import Any, Tuple
from torch import Tensor,nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image,ImageDraw
from clip.model import CLIP,VisionTransformer,ModifiedResNet
from clip_nopooling.model import VisionTransformer as ViT_no_pooling, ModifiedResNet as ResNet_no_pooling
from torch.nn import functional as F
import torch
from chinopie import logger

class RandomMask(nn.Module):
    def __init__(self,mask_size:int,p:float):
        super().__init__()
        self.mask_size=mask_size
        self.p=p
    
    def forward(self,image:Image.Image):
        assert image.mode=='RGB'
        w,h=image.size
        w//=self.mask_size
        h//=self.mask_size
        mask=(torch.rand((1,w,h))<=self.p).float()
        mask=resize_mask(mask,image.size[0])

        mask=mask.repeat(3,1,1)
        mask_image=transforms.ToPILImage()(mask.float())

        array=np.array(image)*(np.array(mask_image)==0)
        return Image.fromarray(array,mode='RGB')


def enlarge_to_448(model:CLIP, original_size:int, target_size:int,patch_size:int):
    # positional embedding is stored in format:
    # 0, 1, 2, 3, 4, 5,
    # 6, 7, 8, 9, ...., <end token>
    pe=model.visual
    grid_length=original_size//patch_size
    grid_token_size=grid_length**2
    if isinstance(pe,VisionTransformer) or isinstance(pe,ViT_no_pooling):
        assert pe.positional_embedding.size(0)==grid_token_size+1
        grid_tokens=pe.positional_embedding.data[:grid_token_size].clone().reshape(grid_length,grid_length,-1)
        eot=pe.positional_embedding[grid_token_size].clone()
    elif isinstance(pe,ModifiedResNet) or isinstance(pe,ResNet_no_pooling):
        assert pe.attnpool.positional_embedding.size(0)==grid_token_size+1
        grid_tokens=pe.attnpool.positional_embedding.data[:grid_token_size].clone().reshape(grid_length,grid_length,-1)
        eot=pe.attnpool.positional_embedding[grid_token_size].clone()
    else:
        raise RuntimeError(f"don't known how to enlarge {type(pe)}")

    grid_tokens=grid_tokens.permute(2,0,1).unsqueeze(0) # (bs, channel, h, w)
    target_grid_length=target_size//patch_size
    expanded_grid_tokens:Tensor=F.interpolate(grid_tokens,(target_grid_length,target_grid_length),mode='bilinear')
    expanded_grid_tokens=expanded_grid_tokens.squeeze(0).permute(1,2,0)
    expanded_grid_tokens=expanded_grid_tokens.flatten(0,1)
    assert expanded_grid_tokens.size(0)==(target_size//patch_size)**2
    assert expanded_grid_tokens.size(1)==eot.size(0)

    res=torch.cat([expanded_grid_tokens,eot.unsqueeze(0)],dim=0)
    if isinstance(model.visual,VisionTransformer) or isinstance(model.visual,ViT_no_pooling):
        model.visual.positional_embedding=nn.parameter.Parameter(res)
    elif isinstance(model.visual,ModifiedResNet) or isinstance(model.visual,ResNet_no_pooling):
        model.visual.attnpool.positional_embedding=nn.parameter.Parameter(res)
    else:
        raise RuntimeError(f"don't known how to enlarge {type(pe)}")
    logger.info(f"enlarge positional embedding in CLIP with linear interpolation to fit {target_size} resolution")


def resize_mask(masks: Tensor, new_size: int):
    assert len(masks.size()) == 3
    hw = masks.size(1)
    if new_size < hw:
        assert hw // new_size * new_size == hw

        masks = F.avg_pool2d(masks, hw // new_size, stride=hw // new_size)
        return masks
    elif new_size == hw:
        return masks
    else:
        batch_size = masks.size(0)
        patch_num = masks.size(1)
        patch_hw = new_size // patch_num
        assert patch_hw * patch_num == new_size
        masks = (
            masks.unsqueeze(-1)
            .expand(-1, -1, -1, patch_hw)
            .reshape(batch_size, patch_num, -1)
            .unsqueeze(2)
            .expand(-1, -1, patch_hw, -1)
            .reshape(batch_size, new_size, new_size)
        )
        return masks

def generate_binary_mask(masks: Tensor, threshold: float = 0.5):
    """
    return a bool tensor where 1 indicates high activation
    """
    binary_masks = masks.clone()
    binary_masks[binary_masks >= threshold] = 1.0
    binary_masks[binary_masks < threshold] = 0.0

    return binary_masks.bool()

def generate_masked_image_by_pixel_mask(images: Tensor, masks: Tensor):
    assert images.dtype == torch.float, masks.dtype == torch.float
    assert images.device == masks.device
    assert len(images.size()) == 4
    assert len(masks.size()) == 3

    batch_size = images.size(0)
    image_channel = images.size(1)
    image_hw = images.size(2)
    assert batch_size == masks.size(0)
    assert image_hw == images.size(3)

    binary_mask = masks.clone()
    channel = binary_mask.reshape(batch_size, 1, image_hw, image_hw)
    binary_mask = channel.repeat(1, 3, 1, 1)
    minnum = images.view(batch_size, -1).min(dim=1, keepdim=True)[0]

    masked = images - binary_mask * 1926.0
    masked = torch.max(
        masked,
        minnum.repeat(1, image_channel * image_hw * image_hw).view(
            batch_size, image_channel, image_hw, image_hw
        ),
    )
    return masked

def generate_cam_image(images: Tensor, masks: Tensor):
    heatmaps: Any = [
        cv2.applyColorMap(np.uint8(255 * x.cpu()), cv2.COLORMAP_JET) for x in masks # type: ignore
    ]
    heatmap = np.float32(heatmaps) / 255.0
    heatmap = heatmap[..., ::-1] # type: ignore

    cam = heatmap + np.float32(images.permute(0, 2, 3, 1).cpu())
    return recover_to_readable_image(
        torch.tensor(cam, dtype=torch.float).permute(0, 3, 1, 2).contiguous()
    ), torch.tensor((heatmap * 255.0).astype(np.uint8), dtype=torch.uint8)

def recover_to_readable_image(image: Tensor) -> Tensor:
    batch_size = image.size(0)
    image_channel = image.size(1)
    image_hw = image.size(2)
    assert image.size(2) == image.size(3)

    image = image.view(batch_size, -1)
    image -= image.min(dim=1, keepdim=True)[0]
    image /= image.max(dim=1, keepdim=True)[0]
    image = image.view(batch_size, image_channel, image_hw, image_hw)
    image *= 255.0
    masked = image.to(dtype=torch.uint8)
    return masked


def rollback_normalization(x:Tensor,mean:Tuple[float,float,float],std:Tuple[float,float,float])->Tensor:
    rev_mean=list(map(lambda a: -a[0]/a[1],zip(mean,std)))
    rev_std=list(map(lambda x:1/x,std))
    return transforms.Normalize(rev_mean,rev_std)(x)

def clamp_as_hard_label(logit:Tensor):
    logit=logit.clone()
    original_dim=logit.dim()
    if logit.dim()==1:
        logit=logit.unsqueeze(0)
    
    for kth in range(logit.size(0)):
        id=logit[kth].argmax()
        logit[kth]*=0
        logit[kth][id]=1
    
    if original_dim==1:logit=logit.squeeze(0)

    return logit
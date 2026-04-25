from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import chinopie
from chinopie import logger

import clip
from clip.model import CLIP,ModifiedResNet
from utils import enlarge_to_448

class ClipWithConstantPrompts(nn.Module):
    def __init__(self,model:CLIP,prompt_ids:Tensor,cache_prompts:bool=True):
        super().__init__()

        self.visual=model.visual
        self.logit_scale=model.logit_scale
        self.prompt_ids=nn.Parameter(prompt_ids,requires_grad=False)

        if cache_prompts:
            self.text_features:Optional[nn.Parameter]=nn.Parameter(model.encode_text(prompt_ids.to(self.visual.conv1.weight.device)).detach())
        else:
            self.text_features=None
            self.token_embedding=model.token_embedding
            self.transformer=model.transformer
            self.ln_final=model.ln_final
            self.positional_embedding=model.positional_embedding
            self.text_projection=model.text_projection


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image:Tensor,eject_feature:bool=False):
        image_features = self.encode_image(image)
        if self.text_features!=None:
            text_features = self.text_features
        else:
            text_features = self.encode_text(self.prompt_ids.detach().to(image.device))

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # shape = [global_batch_size, global_batch_size]
        if not eject_feature:
            return logits_per_image
        else:
            return logits_per_image,image_features

def make_model(model_type:str,labels:List[str]):
    assert model_type in ['RN101','RN50x64','ViT-L/14','ViT-L/14@336px']
    clip_model,transforms=clip.load(model_type,device='cpu')
    if model_type=='RN101':
        enlarge_to_448(clip_model,224,448,32)
    elif model_type=='ViT-L/14@336px':
        enlarge_to_448(clip_model,336,448,14)
    # generate prompts
    prompts=clip.tokenize([f"a photo contains {x}" for x in labels])
    # build model
    model=ClipWithConstantPrompts(clip_model,prompts,cache_prompts=True)
    return model
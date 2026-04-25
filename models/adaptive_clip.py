from typing import List, Any, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from chinopie import logger
import clip_nopooling as clip
from clip_nopooling.model import CLIP,ModifiedResNet, VisionTransformer
from .position_embedding import build_position_encoding

from itertools import cycle


def sigmoid_inverse(p: Tensor):
    epsilon = torch.tensor(1e-5)
    p.clamp_(max=1 - epsilon, min=epsilon)
    return (p / (1 - p)).log()


class SpatialFeatureAdapter(nn.Module):
    """
    CLIP-Adater
    """

    def __init__(self, dim_feat: int, dim_space: int, alpha: float):
        """
        alpha: the portion of newly learned features
        """
        super().__init__()
        self.layer1 = nn.Conv2d(dim_feat, dim_space, 1)
        self.layer2 = nn.Conv2d(dim_space, dim_feat, 1)

        self.alpha = alpha

    def forward(self, x: Tensor):
        y = self.layer1(x)
        y = F.relu(y)
        y = self.layer2(y)

        res = self.alpha * y + (1.0 - self.alpha) * x
        return res

class PositionalEncoder(nn.Module):
    def __init__(self,dim_feat:int,num_layer:int) -> None:
        super().__init__()

        self.dim_feat=dim_feat
        self.position_embedding = build_position_encoding(self.dim_feat)
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.dim_feat, 8, batch_first=True,activation=F.leaky_relu,dim_feedforward=256),
            num_layer,
            nn.LayerNorm(self.dim_feat, eps=1e-5),
        )
    
    def forward(self,x:Tensor):
        batch_size, channel, H, W = x.size()
        pos_embedding = self.position_embedding(x)
        x = x + pos_embedding
        # (bs, channel, H, W) -> (bs, *, channel)
        x = x.flatten(2, 3).permute(0, 2, 1)
        x = self.encoder(src=x)
        return x.permute(0,2,1).reshape(batch_size,channel,H,W)
    
    def get_learnable_params(self,lr:float,lr_transformer:float):
        return [
            {
                "params":self.position_embedding.parameters(),
                "lr":lr,
            },
            {
                "params":self.encoder.parameters(),
                "lr":lr_transformer,
            }
        ]

class PositionalDecoder(nn.Module):
    def __init__(self,dim_feat:int,dim_output:int,num_layer:int) -> None:
        super().__init__()
        self.dim_feat=dim_feat
        self.dim_output=dim_output

        self.position_embedding = build_position_encoding(self.dim_feat)

        self.decoder=nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.dim_feat,8,dim_feedforward=256,batch_first=True,activation=F.leaky_relu),
            num_layer,
            nn.LayerNorm(self.dim_feat,eps=1e-5)
        )

        self.embedding=nn.Embedding(dim_output,dim_feat)
    
    def forward(self,x:Tensor):
        batch_size, channel, H, W = x.size()
        # build input
        pos_embedding = self.position_embedding(x)
        x = x + pos_embedding
        # (bs, channel, H, W) -> (bs, *, channel)
        x = x.flatten(2, 3).permute(0, 2, 1)
        
        #build token: (#labels,#dim)
        tokens:Tensor=self.embedding(torch.arange(0,self.dim_output,dtype=torch.int,device=x.device))
        tokens=tokens.unsqueeze(0).repeat(batch_size,1,1) # (#bs, #label, #dim)

        x = self.decoder(memory=x,tgt=tokens) # x: (#bs, #label, #dim)
        return x
    
    def get_learnable_params(self,lr:float,lr_transformer:float):
        return [
            {
                "params":self.embedding.parameters(),
                "lr":lr,
            },
            {
                "params":self.position_embedding.parameters(),
                "lr":lr,
            },
            {
                "params":self.decoder.parameters(),
                "lr":lr_transformer,
            }
        ]


class AdaptiveClip(nn.Module):
    def __init__(
        self,
        clip:CLIP,
        num_labels:int,
        image_adapter_alpha: float,
    ) -> None:
        super().__init__()
        # init params
        self.dim_feat = 256
        self.num_labels=num_labels

        self._dim_resnet_feat = 2048
        self._dim_space = 2048 // 16

        self.image_encoder = clip.visual
        self.image_adapter = SpatialFeatureAdapter(
            self._dim_resnet_feat, self._dim_space, image_adapter_alpha
        )
        self.image_conv = nn.Conv2d(self._dim_resnet_feat, self.dim_feat, 1)
        self.caption_encoder14 = PositionalEncoder(self.dim_feat,2)
        self.norm=nn.BatchNorm2d(self.dim_feat)
        self.caption_fc = nn.Linear(self.dim_feat, self.num_labels)
        

    def get_learnable_params(
        self, lr: float, lr_transformer: float, lr_backbone: float
    ):
        normal_layers = [
            self.image_adapter.parameters(),
            self.image_conv.parameters(),
            self.caption_fc.parameters(),
        ]
        return [
            *[{"params": a, "lr": b} for a, b in zip(normal_layers, cycle([lr]))],
            *self.caption_encoder14.get_learnable_params(lr,lr_transformer),
            {
                "params": self.image_encoder.parameters(),
                "lr": lr_backbone,
            },
        ]


    def encode_image(self, x: Tensor):
        assert isinstance(self.image_encoder,ModifiedResNet) or isinstance(self.image_encoder,VisionTransformer)
        x = self.image_encoder(x)
        x = self.image_adapter(x)
        x = self.image_conv(x)

        return x


    def build_caption_memory(self, image_features: Tensor)->Tensor:
        t:Tensor = self.caption_encoder14(image_features)
        t = self.norm(t)
        image_features = (t+image_features).flatten(2,3).permute(0,2,1).contiguous()
        return image_features


    def decode_caption(self, memory: Tensor)->Tensor:
        # logits: (bs, #token, #label)
        logits:Tensor = self.caption_fc(memory)

        return logits.sum(dim=1)


    def forward(self, image: Tensor):
        # encode image features
        image_features = self.encode_image(image)
        memory = self.build_caption_memory(image_features)
        caption_logits = self.decode_caption(memory)
        return caption_logits


class AdaptiveClipDecodeStyle(nn.Module):
    def __init__(
        self,
        clip:CLIP,
        num_labels:int,
        image_adapter_alpha: float,
    ) -> None:
        super().__init__()
        # init params
        self.dim_feat = 256
        self.num_labels=num_labels
        if isinstance(clip.visual,ModifiedResNet):
            feat_dim=2048
            hidden_dim=2048//16
        elif isinstance(clip.visual,VisionTransformer):
            feat_dim=768
            hidden_dim=384

        self._dim_resnet_feat = feat_dim
        self._dim_space = hidden_dim

        self.image_encoder = clip.visual
        self.image_adapter = SpatialFeatureAdapter(
            self._dim_resnet_feat, self._dim_space, image_adapter_alpha
        )
        self.image_conv = nn.Conv2d(self._dim_resnet_feat, self.dim_feat, 1)
        self.caption_decoder=PositionalDecoder(self.dim_feat,num_labels,2)
        self.norm=nn.BatchNorm2d(self.dim_feat)
        self.caption_fc = nn.Linear(self.dim_feat, self.num_labels)
        

    def get_learnable_params(
        self, lr: float, lr_transformer: float, lr_backbone: float
    ):
        normal_layers = [
            self.image_adapter.parameters(),
            self.image_conv.parameters(),
            self.caption_fc.parameters(),
        ]
        return [
            *[{"params": a, "lr": b} for a, b in zip(normal_layers, cycle([lr]))],
            *self.caption_decoder.get_learnable_params(lr,lr_transformer),
            {
                "params": self.image_encoder.parameters(),
                "lr": lr_backbone,
            },
        ]


    def encode_image(self, x: Tensor):
        assert isinstance(self.image_encoder,ModifiedResNet) or isinstance(self.image_encoder,VisionTransformer)
        x = self.image_encoder(x) # (bs, dim_feat, H, W)
        x = self.image_adapter(x) # (bs, dim_feat, H, W)
        x = self.image_conv(x) # (bs, dim_clip, H, W)
        x = self.norm(x)

        return x


    def decode_caption(self, image_feat: Tensor):
        # logits: (bs, #label, #dim)
        label_protos = self.caption_decoder(image_feat)
        # logits: (bs, #label, #label)
        logits:Tensor = self.caption_fc(label_protos)

        num_label=logits.size(1)
        logits=logits[:,torch.arange(0,num_label),torch.arange(0,num_label)]

        return logits,label_protos


    def forward(self, image: Tensor,eject_feature:bool=False):
        # encode image features
        image_features = self.encode_image(image)
        caption_logits,label_protos = self.decode_caption(image_features)
        if not eject_feature:
            return caption_logits
        else:
            return caption_logits,image_features,label_protos
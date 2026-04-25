from typing import Any, Dict, List, Optional, Tuple
import math
import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.optim import AdamW
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
import chinopie
from chinopie import ModuleRecipe,EvaluationRecipe, TrainBootstrap
from chinopie import logger
from chinopie.probes import AverageMeter,AveragePrecisionMeter
from chinopie.modelhelper import HyperparameterManager, ModelStaff
from chinopie.filehelper import GlobalFileHelper
from chinopie.datasets.multilabel import VOC2012Dataset,VOC2007Dataset,COCO2014Dataset,NusWideDataset,MultiLabelLocalDataset
from chinopie.datasets.fakeset import FakeRandomSet

import clip
import clip_nopooling
import utils
from models.adaptive_clip import AdaptiveClip,AdaptiveClipDecodeStyle
from models.zero import make_model,ClipWithConstantPrompts
from models.estimator import Estimator
import label_names
import losses
import data
from data import RandomInterpolationDataset
import randaugment

CLIP_MEAN=(0.48145466, 0.4578275, 0.40821073)
CLIP_STD=(0.26862954, 0.26130258, 0.27577711)

transform_zeroshot = transforms.Compose(
    [
        transforms.Resize((448,448),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((448,448)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN,CLIP_STD),
    ]
)

transform_zeroshot_estimating = transforms.Compose(
    [
        transforms.RandomResizedCrop((448,448),scale=(0.1,0.2),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN,CLIP_STD),
    ]
)

transform_train = transforms.Compose(
    [
        transforms.Resize((448,448),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((448,448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN,CLIP_STD),
    ]
)

transform_train_hard = transforms.Compose(
    [
        transforms.Resize((448,448),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((448,448)),
        transforms.RandomHorizontalFlip(),
        randaugment.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN,CLIP_STD),
    ]
)


SUPPORTED_BACKBONES = [
    "RN101",
    "ViT-L/14@336px",
    "ViT-L/14",
    "RN50x64",
]
SUPPORTED_PUSH_EXTREME_MODES = [
    "soft",
    "hard",
    "linear",
    "identity",
]

class ConstantClipWithEstimator(nn.Module):
    def __init__(self,clip:ClipWithConstantPrompts,estimator:Estimator):
        super().__init__()

        self.clip=clip
        self.estimator=estimator
    
    def forward(self,image:Tensor,ids:Tensor):
        return self.clip(image),self.estimator(ids)


class ZeroRecipe(EvaluationRecipe):
    # model:ConstantClipWithEstimator
    def __init__(self,dataset_name:str,trainset:MultiLabelLocalDataset,valset:MultiLabelLocalDataset):
        super().__init__()

        self.dataset_name=dataset_name
        self.trainset=trainset
    
    def ask_hyperparameter(self, hp):
        self.image_backbone=hp.suggest_category('image_backbone_stage_1',SUPPORTED_BACKBONES)
    
    def prepare(self, staff: ModelStaff):
        batch_size = 64 # hp.suggest_int('batch_size',4,32) # FIXME:
        trainloader = DataLoader(self.trainset, batch_size) # NO SHUFFLE!
        staff.reg_dataset(self.trainset, trainloader, self.trainset, trainloader)

        # generate prompts
        if self.dataset_name.find('voc')!=-1:
            assert label_names.VOC_LABELS==self.trainset.get_defined_labels() # type: ignore
            clip_model=make_model(self.image_backbone,label_names.VOC_BETTER_LABELS)
        elif self.dataset_name.find('coco')!=-1:
            coco_labels,coco_better_labels=label_names.get_coco_better_labels_in_alphabet_order()
            assert coco_labels==self.trainset.get_defined_labels(), f"{coco_labels}!={self.trainset.get_defined_labels()}" # type: ignore
            clip_model=make_model(self.image_backbone,coco_better_labels)
        elif self.dataset_name.find('nus')!=-1:
            clip_model=make_model(self.image_backbone,self.trainset.get_defined_labels()) # TODO: maybe we can design better prompt for NUS
        else:
            raise NotImplementedError(f"unknown dataset `{self.dataset_name}`")
        
        model=ConstantClipWithEstimator(clip_model,Estimator(torch.zeros_like(self.trainset.get_all_labels(),dtype=torch.float)))
        chinopie.freeze_model(model.clip)
        staff.reg_model(model)

    
    def set_optimizers(self, model:ClipWithConstantPrompts) -> Optimizer:
        return AdamW(
            model.parameters(),
            lr=0,
        )
    
    def forward(self, data) -> Any:
        return self.model(data['image'],data['index'])
    
    def cal_loss(self, data, output) -> Tensor:
        return F.binary_cross_entropy_with_logits(F.softmax(output[0][:,:self.trainset._num_labels],dim=-1),data['target'].float())
    
    def before_epoch(self):
        self.aps={
            'train':AveragePrecisionMeter(dev=self.dev),
            'val':AveragePrecisionMeter(dev=self.dev),
        }


    def after_iter(self, data, output, phase: str):
        if phase=='val':
            self.raw_model.estimator.update_smoothing(data['index'].to(self.dev),output[0].to(self.dev).softmax(dim=-1).detach(),1,1e10)

        target=torch.zeros_like(output[0],device=output[0].device)
        target[:,:self.trainset._num_labels]=data['target']
        self.aps[phase].add(torch.softmax(output[0],dim=-1),target,data['name'])


    def report_score(self, phase: str) -> float:
        res=self.aps[phase].value()
        logger.info(f"aps: {res}")
        return res[:self.trainset._num_labels].mean().item()


    def export_custom_state(self) -> Dict[str, Any] | None:
        return {
            'estimator':self.raw_model.estimator.state_dict()
        }

class ZeroEstimatorRecipe(EvaluationRecipe):
    # model:ConstantClipWithEstimator
    def __init__(self,dataset_name:str,trainset:MultiLabelLocalDataset,valset:MultiLabelLocalDataset):
        super().__init__()

        self.dataset_name=dataset_name
        self.trainset=trainset
    
    def ask_hyperparameter(self, hp: HyperparameterManager):
        self.image_backbone=hp.suggest_category('image_backbone_stage_1',SUPPORTED_BACKBONES)
        
        self.batch_size = hp.suggest_int('batch_size',4,32)

        self.estimator_alpha=hp.suggest_float('estimator_alpha',1e-3,1,log=True)
        self.estimator_beta=hp.suggest_float('estimator_beta',1e-2,1)
    
    def prepare(self, staff: ModelStaff):
        trainloader = DataLoader(self.trainset, self.batch_size,shuffle=True)
        staff.reg_dataset(self.trainset, trainloader, self.trainset, trainloader)

        # generate prompts
        if self.dataset_name.find('voc')!=-1:
            assert label_names.VOC_LABELS==self.trainset.get_defined_labels() # type: ignore
            clip_model=make_model(self.image_backbone,label_names.VOC_BETTER_LABELS)
        elif self.dataset_name.find('coco')!=-1:
            coco_labels,coco_better_labels=label_names.get_coco_better_labels_in_alphabet_order()
            assert coco_labels==self.trainset.get_defined_labels() # type: ignore
            clip_model=make_model(self.image_backbone,coco_better_labels)
        elif self.dataset_name.find('nus')!=-1:
            clip_model=make_model(self.image_backbone,self.trainset.get_defined_labels()) # TODO: maybe we can design better prompt for NUS
        else:
            raise NotImplementedError(f"unknown dataset `{self.dataset_name}`")
        # load estimator
        model=ConstantClipWithEstimator(clip_model,Estimator(torch.zeros((len(self.trainset),self.trainset._num_labels))))
        if staff.prev_files and len(staff.prev_files)>=1:
            ckpt=torch.load(staff.prev_files[-1].get_best_checkpoint_slot(),'cpu')
            # load estimator
            model.estimator.load_state_dict(ckpt['custom']['estimator'])

            aps=AveragePrecisionMeter(dev=self.dev)
            aps.targets=self.trainset.get_all_labels()
            aps.scores=model.estimator.annotations.data.sigmoid().detach().clone()
            logger.info(f"estimator loaded predefined labels. mAP: {aps.value().mean().item()}")
            logger.info(f"predefined labels:\n{model.estimator.annotations.data}")
        else:
            logger.info("estimator started from 0")
        chinopie.freeze_model(model.clip)
        staff.reg_model(model)

    
    def set_optimizers(self, model:ClipWithConstantPrompts) -> Optimizer:
        return AdamW(
            model.parameters(),
            lr=0,
        )
    
    def forward(self, data) -> Any:
        return self.model(data['image'],data['index'])
    
    def cal_loss(self, data, output) -> Tensor:
        return F.binary_cross_entropy_with_logits(F.softmax(output[1],dim=-1),data['target'].float())
    
    def before_epoch(self):
        self.aps={
            'train':AveragePrecisionMeter(dev=self.dev),
            'val':AveragePrecisionMeter(dev=self.dev),
        }


    def after_iter(self, data, output, phase: str):
        # self.raw_model.estimator.update_smoothing(data['index'].to(self.dev),output[0].to(self.dev).softmax(dim=-1).detach(),self.estimator_alpha,self.estimator_beta)
        self.raw_model.estimator.update_max(data['index'].to(self.dev),output[0].to(self.dev).softmax(dim=-1).detach(),self.estimator_beta)

        target=torch.zeros_like(output[1],device=output[1].device)
        target[:,:self.trainset._num_labels]=data['target']
        self.aps[phase].add(torch.sigmoid(output[1]),target,data['name'])


    def report_score(self, phase: str) -> float:
        res=self.aps[phase].value()
        logger.info(f"aps: {res}")
        return res[:self.trainset._num_labels].mean().item()


    def export_custom_state(self) -> Dict[str, Any] | None:
        return {
            'estimator':self.raw_model.estimator.state_dict(),
        }


def kl_loss(output:Tensor,target:Tensor,T:float):
    return F.kl_div(F.log_softmax(output/T, dim=-1), F.softmax(target/T, dim=-1), reduction='batchmean') * T * T


class AdaptiveClipWithEstimator(nn.Module):
    def __init__(self,clip:AdaptiveClip|AdaptiveClipDecodeStyle,estimator:Estimator):
        super().__init__()

        self.clip=clip
        self.estimator=estimator
    
    def forward(self,image:Tensor,ids:Tensor):
        return self.clip(image),self.estimator(ids)

class BasicFinetuneRecipe(ModuleRecipe):
    def __init__(self,trainset:MultiLabelLocalDataset,valset:MultiLabelLocalDataset):
        super().__init__()

        self.trainset=trainset
        self.valset=valset
    
    def ask_hyperparameter(self, hp: HyperparameterManager):

        self.image_backbone=hp.suggest_category('image_backbone_stage_2',SUPPORTED_BACKBONES)
        self.push_extreme_mode=hp.suggest_category('push_extreme_mode',SUPPORTED_PUSH_EXTREME_MODES)

        self.batch_size=hp.suggest_int('batch_size',2,64,log=True)

        self.lambda_hard=hp.suggest_float('lambda_hard',1e-1,1e1,log=True)



        self.lr=hp.suggest_float('lr',1e-5,1e-2,log=True) # 1e-3 (cliptest)
        self.lr_transformer=hp.suggest_float('lr_transformer',1e-6,1e-4,log=True) # 5e-5 (cliptest)

    def prepare(self, staff: ModelStaff):
        assert staff.prev_files
        ckpt=torch.load(staff.prev_files[-1].get_best_checkpoint_slot(),map_location='cpu')['custom']
        # init estimator with softmax logits
        estimator=Estimator(torch.zeros_like(self.trainset.get_all_labels()))
        estimator.load_state_dict(ckpt['estimator'])
        logger.info(f"[BasicFinetune] push_extreme mode `{self.push_extreme_mode}` (pos=1, neg=0)")
        estimator.push_extreme(self.push_extreme_mode,1,0)

        trainloader=DataLoader(self.trainset,batch_size=self.batch_size,shuffle=True)
        valloader=DataLoader(self.valset,batch_size=self.batch_size)
        staff.reg_dataset(self.trainset,trainloader,self.valset,valloader)

        logger.info(f"[BasicFinetune] loading CLIP backbone `{self.image_backbone}`")
        clip_model,_=clip_nopooling.load(self.image_backbone,device='cpu')
        # no need to interpolate the positional embedding, as it is removed in adaptive_clip
        clip_model=AdaptiveClipDecodeStyle(clip_model,self.trainset._num_labels,0.5)
        model=AdaptiveClipWithEstimator(clip_model,estimator)
        # freeze backbone
        chinopie.freeze_model(model.clip.image_encoder)
        staff.reg_model(model)



    def set_optimizers(self, model:AdaptiveClipWithEstimator) -> Optimizer:
        
        return AdamW(
            model.clip.get_learnable_params(lr=self.lr,lr_transformer=self.lr_transformer,lr_backbone=0), # config of backbone is useless, being frozen already
            lr=self.lr,
        )
    
    def forward_train(self, data) -> Any:
        images=torch.cat([data['image'],data['extra_image']])
        return self.model(images,data['index'])
    
    def forward(self, data) -> Any:
        images=torch.cat([data['image'],data['extra_image']])
        return self.model(images,torch.zeros_like(data['index']))
    
    def cal_loss_train(self, data, output) -> Tensor:
        bs=data['image'].size(0)
        return (F.binary_cross_entropy_with_logits(output[0][:bs],target=torch.softmax(output[1].detach().to(self.dev),dim=-1))
                +self.lambda_hard*F.binary_cross_entropy_with_logits(output[0][bs:],target=torch.softmax(output[1].detach().to(self.dev),dim=-1)))

    def cal_loss(self, data, output) -> Tensor:
        bs=data['image'].size(0)
        return F.binary_cross_entropy_with_logits(output[0][:bs],target=data['target'].float())
    
    def before_epoch(self):
        self.aps={
            'train':AveragePrecisionMeter(dev=self.dev),
            'val':AveragePrecisionMeter(dev=self.dev),
        }
    
    def after_iter(self, data, output, phase: str):
        bs=data['image'].size(0)
        self.aps[phase].add(output[0][:bs],data['target'],data['name'])
    
    def report_score(self, phase: str) -> float:
        aps=self.aps[phase].value()
        logger.info(f"aps: {aps}")
        return aps.mean().item()

class FinetuneRecipe(ModuleRecipe):
    # model:AdaptiveClipWithEstimator
    def __init__(self,trainset:MultiLabelLocalDataset,valset:MultiLabelLocalDataset):
        super().__init__()

        self.trainset=trainset
        self.valset=valset

        self.train_preprocess=self.trainset._preprocess
        self.train_extra_preprocess=self.trainset._extra_preprocess

        self.partialbce=losses.PartialAsymmetricBCE()
    
    def ask_hyperparameter(self, hp: HyperparameterManager):
        self.image_backbone=hp.suggest_category('image_backbone_stage_2',SUPPORTED_BACKBONES)
        self.push_extreme_mode=hp.suggest_category('push_extreme_mode',SUPPORTED_PUSH_EXTREME_MODES)
        self.push_extreme_beta=hp.suggest_float('push_extreme_beta',1e-3,1)
        self.topk=hp.suggest_int('topk',1,1000)

        self.overlay_num=hp.suggest_int('overlay_num',1,3,step=1)
        self.overlay_scale=hp.suggest_float('overlay_scale',0,1,step=0.1)
        self.overlay_p=hp.suggest_float('overlay_p',0.1,1)

        self.batch_size=hp.suggest_int('batch_size',2,24,log=True)

        # other params
        self.lambda_rev=hp.suggest_float('lambda_rev',1e-1,1e1,log=True)
        self.lambda_hard=hp.suggest_float('lambda_hard',1e-1,1e1,log=True)
        self.lambda_avg=hp.suggest_float('lambda_avg',1e-1,1e1,log=True)
        self.alpha=hp.suggest_float('alpha',1,5)

        self.lr=hp.suggest_float('lr',1e-5,1e-2,log=True) # 1e-3 (cliptest)
        self.lr_transformer=hp.suggest_float('lr_transformer',1e-6,1e-4,log=True) # 5e-5 (cliptest)
        self.lr_estimator=hp.suggest_float('lr_estimator',1e-4,1e-2,log=True)
        self.lr_backbone=hp.suggest_float('lr_backbone',1e-6,1e-4,log=True)

        return super().ask_hyperparameter(hp)

    def prepare(self, staff: ModelStaff):
        assert staff.prev_files
        ckpt=torch.load(staff.prev_files[-1].get_best_checkpoint_slot(),map_location='cpu')['custom']
        # init estimator with softmax logits
        estimator=Estimator(torch.zeros_like(self.trainset.get_all_labels()))
        estimator.load_state_dict(ckpt['estimator'])
        # data clean
        # sample_maxlogits=[]
        # for k,sample in enumerate(estimator.annotations.data):
        #     sample_maxlogits.append((sample.max(),k))
        # sample_maxlogits.sort(reverse=True)
        # retained_ids=list(map(lambda x:x[1],sample_maxlogits[:5000]))
        # self.trainset=self.trainset.filter_by_ids(retained_ids)
        # estimator.annotations=nn.Parameter(estimator.annotations.data[retained_ids])
        # push extreme pos and neg
        logger.info(f"[Finetune] push_extreme mode `{self.push_extreme_mode}` (pos_beta={self.push_extreme_beta}, neg=0)")
        estimator.push_extreme(self.push_extreme_mode,pos=self.push_extreme_beta,neg=0)
        logger.info(f"pseudo labels:\n{estimator.annotations.data}")
        # test precision
        aps=AveragePrecisionMeter(dev=self.dev)
        aps.scores=estimator.annotations.detach()
        aps.targets=self.trainset.get_all_labels()
        logger.info(f"estimator aps:{aps.value().mean().item()}")
        acc_top1=0
        for (l,t) in zip(aps.scores,aps.targets):
            maxl=l.argmax()
            if t[maxl]==1:acc_top1+=1
        logger.info(f"top1 acc: {acc_top1/aps.scores.size(0)}")

        # select top-k as the overlay dataset
        selected_ids=[]
        for class_idx in range(self.trainset._num_labels):
            v,k=estimator.annotations[:,class_idx].detach().sort(descending=True) # The probe must be collected on unshuffled dataset
            selected_ids.extend(k[:self.topk].tolist())
        self.selected_ids=torch.tensor(list(set(selected_ids)),dtype=torch.int) # to support complex indexing
        logger.info(f"use {len(self.selected_ids)} images as overlay dataset")
        overlay_dataset=self.trainset.filter_by_ids(self.selected_ids)
        # build the rand dataset
        ps,t=[],1
        for _ in range(self.overlay_num-1):
            ps.append(t*self.overlay_p)
            t-=t*self.overlay_p
        ps.append(t)
        random_trainset=RandomInterpolationDataset(self.trainset,
                                                   overlay_dataset,
                                                   num_overlay=self.overlay_num,
                                                   scales=[(self.overlay_scale**2/k)**0.5 for k in range(1,self.overlay_num+1)],
                                                   p=ps,
                                                   preprocess=self.train_preprocess,
                                                   extra_preprocess=self.train_extra_preprocess)
        # reg dataset
        trainloader=DataLoader(random_trainset,batch_size=self.batch_size,shuffle=True,collate_fn=data.custom_collate,pin_memory=True)
        valloader=DataLoader(self.valset,batch_size=self.batch_size,pin_memory=True)
        staff.reg_dataset(random_trainset,trainloader,self.valset,valloader)

        # reg main model
        logger.info(f"[Finetune] loading CLIP backbone `{self.image_backbone}`")
        clip_model,_=clip_nopooling.load(self.image_backbone,device='cpu')
        if self.image_backbone=='ViT-L/14@336px':
            utils.enlarge_to_448(clip_model,336,448,14)
        elif self.image_backbone=='RN101':
            utils.enlarge_to_448(clip_model,224,448,32)
        # no need to interpolate the positional embedding, as it is removed in adaptive_clip
        model=AdaptiveClipDecodeStyle(clip_model,self.trainset._num_labels,0.5)
        union_model=AdaptiveClipWithEstimator(model,estimator)
        # freeze backbone. No freezing backbone leads to performance drop
        chinopie.freeze_model(union_model.clip.image_encoder)
        staff.reg_model(union_model)

        


    def set_optimizers(self, model:AdaptiveClipWithEstimator) -> Optimizer:
        return AdamW(
            [
                {
                    "params":model.estimator.parameters(),
                    "lr":self.lr_estimator,
                },
                *model.clip.get_learnable_params(lr=self.lr,lr_transformer=self.lr_transformer,lr_backbone=self.lr_backbone)
            ],
            lr=self.lr,
        )
    
    def set_scheduler(self, optimizer: Optimizer) -> Optional[LRScheduler]:
        # TODO: find this
        # return torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
        pass
    
    def forward_train(self, data) -> Any:
        images=torch.cat([data['image'],data['extra_image']])
        main_ids:Tensor=data['index'] # (bs)
        overlay_ids=self.selected_ids[[x for indices in data['overlay_indices'] for x in indices]]
        ids=torch.cat([main_ids,overlay_ids.to(main_ids.device)]).detach()

        return self.model(images,ids)
    
    def forward(self, data) -> Any:
        images=torch.cat([data['image'],data['extra_image']])
        main_ids=torch.zeros_like(data['index'])

        return self.model(images,main_ids)
    
    def reform_pseudo_logits(self,data,esti_output:Tensor,strategy:str):
        bs=data['image'].size(0)
        main_logits=esti_output[:bs]
        hard_targets=utils.clamp_as_hard_label(main_logits)

        overlay_logits=torch.zeros_like(main_logits)
        ptr=bs
        for k,v in enumerate(data['overlay_indices']):
            this_target=esti_output[ptr:ptr+len(v)]
            if strategy=='max':
                overlay_logits[k]=this_target.max(dim=0)[0]
            elif strategy=='sum':
                overlay_logits[k]=this_target.sum(dim=0)
            else:
                raise NotImplementedError()
            hard_targets[k]+=utils.clamp_as_hard_label(this_target).sum(dim=0)
            ptr+=len(v)
        assert ptr==esti_output.size(0)
        return main_logits,overlay_logits,hard_targets.clamp_max(1)

    
    # def cal_loss_train1(self,data,output,mix_method:str)->Tensor:
    #     """
    #     METHOD1. Just to simply test if the mixing strategy is usefull or not.
    #     """
    #     target,overlay_target=self.reform_pseudo_logits(data,output[1],'max')
    #     target=target.sigmoid()
    #     overlay_target=overlay_target.sigmoid()

    #     if mix_method=='sum':
    #         fused_target=(target+overlay_target).clamp_max(1)
    #     elif mix_method=='max':
    #         fused_target=torch.max(target,overlay_target)
    #     else:
    #         raise NotImplementedError(f"unknown mix method `{mix_method}`")
        
    #     return F.binary_cross_entropy_with_logits(output[0],target=fused_target) 

    
    def cal_loss_train_on_weak_strong_augs(self,data,output,strategy:str)->Tensor:
        bs=data['image'].size(0)
        target,overlay_target,hard_target=self.reform_pseudo_logits(data,output[1],strategy)
        if strategy=='max':
            fused_target=torch.max(target,overlay_target)
        elif strategy=='sum':
            fused_target=target+overlay_target
        else:
            raise NotImplementedError()

        loss_1=(F.binary_cross_entropy_with_logits(output[0][:bs],fused_target.detach().sigmoid())
                +self.lambda_hard*F.binary_cross_entropy_with_logits(output[0][bs:],fused_target.detach().sigmoid()))
        loss_2=F.binary_cross_entropy_with_logits(fused_target,output[0][:bs].detach().sigmoid())
        loss_3=losses.loss_avg_annotation(output[1].sigmoid(),self.alpha)+losses.loss_avg_annotation(output[0].sigmoid(),self.alpha)
        return loss_1+self.lambda_rev*loss_2+self.lambda_avg*loss_3

    
    def cal_loss_train(self, data, output) -> Tensor:
        return self.cal_loss_train_on_weak_strong_augs(data,output,'max')
        

    def cal_loss_val(self, data, output) -> Tensor:
        bs=data['image'].size(0)
        return F.binary_cross_entropy_with_logits(output[0][:bs],target=data['target'].float())
    

    def cal_loss(self, data, output) -> Tensor:
        assert False
    

    def before_epoch(self):
        self.aps={
            'train':AveragePrecisionMeter(dev=self.dev),
            'esti':AveragePrecisionMeter(dev=self.dev),
            'val':AveragePrecisionMeter(dev=self.dev),
        }
    

    def after_iter(self, data, output, phase: str):
        bs=data['image'].size(0)
        if phase=='train':
            self.aps[phase].add(output[0][:bs],(data['target']+data['overlay_target']).clamp_max(1),data['name'])
            self.aps['esti'].add(output[1][:bs],data['target'],data['name'])
            # save_image(make_grid(torch.cat([data['image'],data['extra_image']])),'preview.png')
        else:
            self.aps[phase].add(output[0][:bs],data['target'],data['name'])
    
    def after_epoch(self):
        torch.save(self.aps['esti'],f"esti-{self.cur_epoch}.pth")
        logger.info("turned on keeping pos labels")
        self.raw_model.estimator.push_extreme(self.push_extreme_mode,self.push_extreme_beta,0)
    
    def report_score(self, phase: str) -> float:
        aps=self.aps[phase].value()
        esti=self.aps['esti'].value()
        logger.info(f"aps:\n{aps}")
        logger.info(f"esti: {esti.mean().item()}\n{esti}")
        return aps.mean().item()


"""
## 1.0.0
- BasicFinetuneRecipe with logits from CLIP RN50x64.
- Loss: BCE with logits.

- 81.4 on VOC 2012

## 1.1.0 <- 1.0.0
- Test KL loss (temperature 1)

- No use. Bad than 1.0.0.

## 1.2.0 <- 1.0.0
- FinetuneRecipe with logits from CLIP RN50x64.
- No hard augmentations. Just the simplest one to find if the mixing is usefull or not.
- Use sum in mixing.

- Useful.

## 1.2.1 <- 1.2.0
- Use max in mixing.

- Useful.

## 1.3.0 <- 1.2.1
- Add weak & hard augmentations

- Seems bad. But that is contradicted with the former exp in cliptest. Maybe... hyperparameter?
- I'll just keep it. At least we can tune the propotion of the hard augs' loss.

## 1.4.0 <- 1.3.0
- Introduce the estimator, from ROLE.
- avg annotation loss and rev learning are NOT enabled, to test the code consistency.

- ? Strange improvement again. I don't know where it comes.
    - Okay I use wrong type int for pseudo target in the old code.

## 1.4.1 <- 1.4.0
- Enable estimator, avg annotation loss, rev learning.
- sum mix logit.

- max seems useful, but sum is shit.

## 1.4.3 <- 1.4.1
- Use max logit.

## ~~1.4.2 <- 1.4.1~~
- Try the loss in science space.

- No use.

## 1.4.4 <- 1.4.3
- Tune augmentation.
- old: top 300, overlay 1, scale 0.5, p 1
- new: top 300, overlay 3, scale 0.65, p 0.5

- Still 84.7, the same as 1.4.3

## 1.4.5 <- 1.4.4
- Add ROLE loss on model side too.

- Useful?

## 1.4.6 <- 1.4.5
- Is hard aug really useful?
- Set lambda_hard as 0

- Hard aug is important


## ~~1.5.0 <- 1.4.5~~
- Add partialbce on hard label exported from CLIP logits.

- Bad

## ~~1.5.1 <- 1.5.0~~
- Decrease coef of hard label loss

- WRONG SETTING...
- I made a mistake using the real labels...

## ~~1.5.2 <- 1.5.1~~
- Use the pseudo hard label

- No use 捏.

## ~~1.5.3 <- 1.5.2~~
- Remove partialbce on model side, only estimator side now.

- Bad.
- And the quality of estimator decreases, which means there is correction to clip's wrong labeling during training...

## 1.6.0 <- 1.4.5
- Introduce exp smoothing estimator on CLIP zero-shot.

- 操你妈的，有用但是巨 tm 没用

## 1.6.1 <- 1.6.0
- 把不同增强的 logit 在 estimator 上的叠加方法换成 max
- 进一步缩小  crop 到 0.15 0.5

- 有用但是完全超不了人家

## 1.6.2 <- 1.6.1
- 继续缩小 crop 的 scale，0.13 到 0.3

- 更小的图确实是会有更好的性能
- 顺便取 max 的时候 beta 似乎根本就没用，不如直接改成 0

## 1.6.3 <- 1.6.2
- 把 beta 切换到了百分比

- 87.5 on VOC

## ~~1.6.4 <- 1.6.3~~
- 发现 estimator 的数值分布极其抽象，所以决定锁定

- 反而变坏了，离谱


## ~~1.7.0 <- 1.6.3~~
- 增加 group prompt，你妈的

- 实测 VOC 上没啥用，我不会调你的 prompt，好了吧？

## 1.8.0 <- 1.6.3
- 数据清洗措施：突然发现 top1 错误几乎集中在后700个里，所以以每个 sample 的 max logit 排序，取前 5000。
- hard push

- 似乎那 700 张图像还是挺有用的……logit 有点下降
- 但是如果想用上这 700 张图像，可能得再单独分出来一个蒸馏 loss 和固定的 estimator。也说不准，毕竟 soft push 之后相对关系能保留住，说不定不需要。

## 1.8.1 <- 1.8.0
- 用 soft 的 push

- 更稳定了

## 1.8.2 <- 1.8.1
- 砍半模型 feat & latent & transformer: 512, 512, 1024

- ？似乎有用

## 1.8.3 <- 1.8.2
- 继续砍 feat & latent & transformer: 384, 256, 512

- 确实有用，到 88 了

## 1.8.4 <- 1.8.3
- 我看你还想继续砍: 256, 128, 256

- 感觉没必要到这种程度来着…算了还是按照这个来吧。

## 1.8.5 <- 1.8.4
- 把过滤掉的700张图片还回去，毕竟是 soft push 了……我服了。

- 反而，更好了。

## 1.9.0 <- 1.8.5
- 试一试用 decoder 结构和那个空间限制方法

- 我草草草草草草草草草草草！！！！！！！！！！！！！！！！！！！！！！90.9了！

## Rest
- 后续记录转移至 Obsidian

## TODO
- transofrmer 换成 silu

"""

def get_dataset(file:GlobalFileHelper,name:str,preprocess:Any,extra_preprocess:Any):
    if name=='voc2012':
        trainset=VOC2012Dataset(file.get_dataset_slot('voc2012'),'train',preprocess=preprocess,extra_preprocess=extra_preprocess)
        valset=VOC2012Dataset(file.get_dataset_slot('voc2012'),'val',preprocess=preprocess,extra_preprocess=extra_preprocess)
    elif name=='voc2007':
        trainset=VOC2007Dataset(file.get_dataset_slot('voc2007'),'train',preprocess=preprocess,extra_preprocess=extra_preprocess)
        valset=VOC2007Dataset(file.get_dataset_slot('voc2007'),'val',preprocess=preprocess,extra_preprocess=extra_preprocess)
    elif name=='coco2014':
        trainset=COCO2014Dataset(file.get_dataset_slot('coco2014'),'train',preprocess=preprocess,extra_preprocess=extra_preprocess)
        valset=COCO2014Dataset(file.get_dataset_slot('coco2014'),'val',preprocess=preprocess,extra_preprocess=extra_preprocess)
    elif name=='nus-wide':
        trainset=NusWideDataset(file.get_dataset_slot('NUS-WIDE'),'trainval',preprocess=preprocess,extra_preprocess=extra_preprocess)
        valset=NusWideDataset(file.get_dataset_slot('NUS-WIDE'),'test',preprocess=preprocess,extra_preprocess=extra_preprocess)
    else:
        raise NotImplementedError("no dataset")
    
    return trainset,valset

if __name__ == "__main__":
    dataset=chinopie.get_env('dataset')

    tb = TrainBootstrap(
        "deps",
        num_epoch=30,
        load_checkpoint=True,
        save_checkpoint=True,
        comment="interpolation",
        version="1.9.2",
        # version="vit",
        dataset=dataset,
        enable_prune=True,
        enable_snapshot=False,
        dev='cuda',
        world_size=1,
        seed=9,
    )

    tb.hp.reg_int("batch_size",16) # gs' result is 16. Fuck 2080's memory size.
    tb.hp.reg_float("lr", 1e-3) # 1e-3
    tb.hp.reg_float('lr_transformer',5e-5)
    tb.hp.reg_float('lr_estimator',1e-2) # gs' result is 1e-3
    tb.hp.reg_float('lr_backbone',0)

    tb.hp.reg_float('estimator_alpha',-1)
    tb.hp.reg_float('estimator_beta',0.01)

    tb.hp.reg_float('lambda_hard',0.7)
    tb.hp.reg_float('lambda_avg',0)
    tb.hp.reg_float('lambda_rev',0.5)
    tb.hp.reg_float('alpha',1.4)

    tb.hp.reg_int('topk',200)
    tb.hp.reg_int('overlay_num',1)
    tb.hp.reg_float('overlay_scale',0.65)
    tb.hp.reg_float('overlay_p',0.6)
    tb.hp.reg_category('image_backbone_stage_1','RN50x64')
    tb.hp.reg_category('image_backbone_stage_2','RN101')
    tb.hp.reg_category('push_extreme_mode','soft')
    tb.hp.reg_float('push_extreme_beta',1)

    trainset,valset=get_dataset(tb.file,dataset,preprocess=transform_zeroshot,extra_preprocess=None)
    tb.optimize(ZeroRecipe(dataset,trainset,valset),direction='maximize',inf_score=0,n_trials=1,num_epoch=1,stage=0)
    trainset,valset=get_dataset(tb.file,dataset,preprocess=transform_zeroshot_estimating,extra_preprocess=None)
    tb.optimize(ZeroEstimatorRecipe(dataset,trainset,valset),direction='maximize',inf_score=0,n_trials=1,stage=1)
    trainset,valset=get_dataset(tb.file,dataset,preprocess=transform_zeroshot,extra_preprocess=transform_train_hard)
    tb.optimize(FinetuneRecipe(trainset,valset),direction='maximize',inf_score=0,n_trials=1,num_epoch=1,stage=2)
    # tb.optimize(BasicFinetuneRecipe(trainset,valset),direction='maximize',inf_score=0,n_trials=1,num_epoch=10,stage=2)

    tb.release()
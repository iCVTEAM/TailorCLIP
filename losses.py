from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F

def loss_avg_annotation(output:Tensor,alpha:float):
    assert output.dim()==2
    num_labels=output.size(1)
    return ((output.sigmoid().sum(dim=1).mean(dim=0) - alpha) / num_labels) ** 2


def multilabel_categorical_crossentropy(y_pred:Tensor,y_true:Tensor):
    """
    多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素是0～1
           的数，表示当前类是目标类的概率；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 和
           https://kexue.fm/archives/9064。
    """
    EPS=1e-7
    
    y_mask = y_pred > -torch.inf / 10
    n_mask = (y_true < 1 - EPS) & y_mask
    p_mask = (y_true > EPS) & y_mask
    y_true = torch.clamp(y_true,EPS,1-EPS)
    infs = torch.zeros_like(y_pred)
    infs.fill_(torch.inf)
    y_neg = torch.where(n_mask,y_pred,-torch.inf)+torch.log(1-y_true)
    y_pos = torch.where(p_mask,-y_pred,-torch.inf)+torch.log(y_true)
    zeros = torch.zeros_like(y_pred[:,:1])
    y_neg = torch.cat([y_neg,zeros],dim=-1)
    y_pos = torch.cat([y_pos,zeros],dim=-1)
    neg_loss = torch.logsumexp(y_neg,dim=-1)
    pos_loss = torch.logsumexp(y_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


class PartialAsymmetricBCE(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True) -> None:
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
    
    def forward(self,outputs:Tensor,targets:Tensor,weights:Optional[Tensor]=None):
        masks=targets.clone()
        # generate mask. only 1 and -1 are valid labels.
        masks[masks==-1]=1
        # set -1 as 0 to fit standard BCE loss
        targets2=targets.clone()
        targets2[targets==-1]=0

        # Asymmetric Clipping. I don't know if this is necessary for ASL
        # if self.clip is not None and self.clip > 0:
        #     xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic BCE
        loss = F.binary_cross_entropy_with_logits(outputs,targets2,weight=weights,reduction='none')

        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            x_sigmoid = torch.sigmoid(outputs)
            xs_pos = x_sigmoid
            xs_neg = 1 - x_sigmoid
            # details about this line: https://github.com/Alibaba-MIIL/ASL/issues/31
            with torch.no_grad():
                pt0 = xs_pos * targets2
                pt1 = xs_neg * (1 - targets2)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * targets2 + self.gamma_neg * (1 - targets2)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w
        loss = loss * masks.float() # partial label

        return loss.sum(dim=1).mean(dim=0)
import torch
from torch import nn,Tensor
from chinopie import logger

def inverse_sigmoid(x:Tensor):
    EPS = 1e-6
    # clip
    x=x.clamp(EPS,1-EPS)
    return torch.log(x/(1-x))

class Estimator(nn.Module):
    def __init__(self,predefined_annotations:Tensor) -> None:
        super().__init__()

        if ((predefined_annotations<=1)&(predefined_annotations>=0)).all()==True:
            logger.info("the predefined annotation is sigmoid or softmax")
            predefined_annotations=inverse_sigmoid(predefined_annotations)
        self.annotations=nn.Parameter(predefined_annotations)

    def forward(self,ids:Tensor):
        return self.annotations[ids]

    def _resolve_topk(self, value:int|float, num_classes:int) -> int:
        if isinstance(value, float):
            if value <= 0:
                return 0
            if value <= 1:
                k = int(torch.ceil(torch.tensor(num_classes * value)).item())
            else:
                k = int(value)
        else:
            k = int(value)
        return max(0, min(num_classes, k))

    @torch.no_grad()
    def push_extreme(self, method:str, pos:int|float, neg:int|float):
        if method=='hard':
            if pos:
                ids=self.annotations.argmax(dim=-1)
                self.annotations[range(self.annotations.size(0)),ids]=inverse_sigmoid(torch.tensor(0.95))
            if neg:
                ids=self.annotations.argmin(dim=-1)
                self.annotations[range(self.annotations.size(0)),ids]=inverse_sigmoid(torch.tensor(0.05))
        elif method=='soft':
            pos_k=self._resolve_topk(pos,self.annotations.size(-1))
            neg_k=self._resolve_topk(neg,self.annotations.size(-1))
            self.select_and_push_extreme_sqrt(self.annotations,pos_k,neg_k)
        elif method=='linear':
            pos_k=self._resolve_topk(pos,self.annotations.size(-1))
            neg_k=self._resolve_topk(neg,self.annotations.size(-1))
            self.select_and_push_extreme_linear(self.annotations,pos_k,neg_k,alpha=0.1)
        elif method=='identity':
            pos_k=self._resolve_topk(pos,self.annotations.size(-1))
            neg_k=self._resolve_topk(neg,self.annotations.size(-1))
            self.select_and_push_extreme_identity(self.annotations,pos_k,neg_k)
        elif method=='threshold':
            assert type(pos)==float and type(neg)==float
            self.annotations[self.annotations>=inverse_sigmoid(torch.tensor(pos))]=inverse_sigmoid(torch.tensor(0.95))
            self.annotations[self.annotations<=inverse_sigmoid(torch.tensor(neg))]=inverse_sigmoid(torch.tensor(0.05))
        else:
            raise NotImplementedError()
    
    def select_and_push_extreme(self,logits:Tensor|nn.Parameter,pos:bool,neg:bool):
        if pos>0:
            ids=logits.argmax(dim=-1)
            former_pred=torch.sigmoid(logits[range(logits.size(0)),ids])
            logits[range(logits.size(0)),ids]=inverse_sigmoid((former_pred*100).sqrt()*10/100)
        if neg>0:
            ids=logits.argmin(dim=-1)
            former_pred=torch.sigmoid(logits[range(logits.size(0)),ids])
            logits[range(logits.size(0)),ids]=inverse_sigmoid((former_pred*100/10).pow(2)/100)
            
        return logits
    
    def select_and_push_extreme_sqrt(self,logits:Tensor|nn.Parameter,pos:int,neg:int):
        if pos:
            v,ids=torch.topk(logits,k=pos,dim=-1)
            oneth=torch.arange(0,logits.size(0)).unsqueeze(-1).repeat(1,pos).flatten()
            former_pred=torch.sigmoid(logits[oneth,ids.flatten()])
            logits[oneth,ids.flatten()]=inverse_sigmoid((former_pred*100).sqrt()*10/100)
        if neg:
            v,ids=torch.topk(logits,k=neg,dim=-1,largest=False)
            oneth=torch.arange(0,logits.size(0)).unsqueeze(-1).repeat(1,neg).flatten()
            former_pred=torch.sigmoid(logits[oneth,ids.flatten()])
            logits[oneth,ids.flatten()]=inverse_sigmoid((former_pred*100/10).pow(2)/100)
        return logits
    
    def select_and_push_extreme_linear(self,logits:Tensor|nn.Parameter,pos:int,neg:int,alpha:float=0.1):
        """
        Linear version of extreme pushing.
        alpha controls strength (default 0.1 ≈ sqrt version strength)
        """
        B = logits.size(0)
        if pos:
            _, ids = torch.topk(logits, k=pos, dim=-1)
            oneth = torch.arange(B, device=logits.device).unsqueeze(-1).repeat(1, pos).flatten()
            p = torch.sigmoid(logits[oneth, ids.flatten()])
            # p' = alpha + (1-alpha)*p   (linear compression of large probs)
            p_new = alpha + (1 - alpha) * p
            logits[oneth, ids.flatten()] = inverse_sigmoid(p_new)
        if neg:
            _, ids = torch.topk(logits, k=neg, dim=-1, largest=False)
            oneth = torch.arange(B, device=logits.device).unsqueeze(-1).repeat(1, neg).flatten()
            p = torch.sigmoid(logits[oneth, ids.flatten()])
            # p' = alpha * p   (linear shrink of small probs)
            p_new = alpha * p
            logits[oneth, ids.flatten()] = inverse_sigmoid(p_new)
        return logits
    
    def select_and_push_extreme_identity(self,logits:Tensor|nn.Parameter,pos:int,neg:int):
        """
        Identity version: select extreme logits but do NOT rescale probabilities.
        This is a control variant for sqrt / linear pushing.
        """
        B = logits.size(0)
        if pos:
            _, ids = torch.topk(logits, k=pos, dim=-1)
            oneth = torch.arange(B, device=logits.device).unsqueeze(-1).repeat(1, pos).flatten()
            p = torch.sigmoid(logits[oneth, ids.flatten()])
            logits[oneth, ids.flatten()] = inverse_sigmoid(p)

        if neg:
            _, ids = torch.topk(logits, k=neg, dim=-1, largest=False)
            oneth = torch.arange(B, device=logits.device).unsqueeze(-1).repeat(1, neg).flatten()
            p = torch.sigmoid(logits[oneth, ids.flatten()])
            logits[oneth, ids.flatten()] = inverse_sigmoid(p)
        return logits


    
    @torch.no_grad()
    def update_smoothing(self,ids:Tensor,targets:Tensor,alpha:float,beta:float):
        targets=inverse_sigmoid(targets)
        allow_slots=(self.annotations[ids]-targets).abs()<=beta
        self.annotations[ids]=(targets*allow_slots.int()+self.annotations[ids]*(~allow_slots).int())*alpha+self.annotations[ids]*(1-alpha)


    @torch.no_grad()
    def update_max(self,ids:Tensor,targets:Tensor,beta:float):
        # 大于某个值的浮动应该被视为发现了物体的出现
        allow_slots=(self.annotations[ids].sigmoid()-targets).abs()>=beta
        self.annotations[ids]=torch.max(self.annotations[ids],inverse_sigmoid(targets))*allow_slots.int()+self.annotations[ids]*(~allow_slots).int()
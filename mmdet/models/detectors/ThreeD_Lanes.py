from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector

@MODELS.register_module()
class ThreeD_Lanes(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.head = MODELS.build(head) if head is not None else None


    @property          
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    def extract_feat(self, batch_inputs):
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def forward(self, input, mode):
        if mode == 'loss':
            return self.forward_train(input)
        elif mode == 'predict':
            return self.predict(input)
        
    def forward_train(self, input):
        outs = self.extract_feat(input)
        if self.with_head:
            losses = self.head(input)
            return losses
        return outs

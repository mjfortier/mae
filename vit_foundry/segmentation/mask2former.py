
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from config import Mask2FormerConfig, Mask2FormerOutput
from components.transformer_module import Mask2FormerTransformerModule
from components.loss import Mask2FormerLoss
from timm.models.layers import trunc_normal_


class Mask2FormerModel(nn.Module):
    '''
    Custom Mask2Former model, based heavily on the HuggingFace implementation.
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/mask2former/modeling_mask2former.py

    Backbone here refers to both an encoder, and a feature pyramid constructor.
    Examples: (Swin Transformer + FPN), (ViTMAE + simple pyramid)
    '''
    def __init__(self, config: Mask2FormerConfig, backbone: nn.Module):
        super().__init__()
        self.config = config

        self.backbone = backbone

        self.transformer_module = Mask2FormerTransformerModule(
            hidden_size=config.transformer_hidden_size,
            fpn_hidden_size=backbone.config.fpn_hidden_size,
            mask_feature_size=backbone.config.mask_feature_size,
            num_queries=config.num_queries,
            num_feature_levels=backbone.pyramid_depth,
            decoder_layers=config.num_layers,
            activation_fn=config.activation_function
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.transformer_hidden_size, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        self._init_weights()


    def _init_weights(self):
        # Backbone and transformer_module weights are assumed to be initialized
        trunc_normal_(self.class_predictor.weight, std=0.02)
        nn.init.constant_(self.class_predictor.bias, 0)


    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Mask2FormerOutput:
        backbone_output = self.backbone(pixel_values)

        transformer_module_output = self.transformer_module(
            mask_features=backbone_output.mask_features,
            feature_pyramid=backbone_output.feature_pyramid,
        )

        mask_predictions = transformer_module_output.mask_predictions # note, this is from every layer of the transformer
        class_queries_logits = [self.class_predictor(hs) for hs in transformer_module_output.hidden_states]

        loss = None
        if mask_labels is not None and class_labels is not None:
            auxiliary_predictions = [(masks, logits) for masks, logits in zip(mask_predictions[:-1], class_queries_logits[:-1])]
            loss_dict = self.criterion(
                intermediate_mask_predictions=mask_predictions[-1], # final mask set prediction
                class_queries_logits=class_queries_logits[-1], # final class set prediction
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_predictions if config.use_auxiliary_loss else None,
            )
            # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
            # TODO: make sure this works w.r.t. iterator mutability
            for key, weight in self.weight_dict.items():
                for loss_key, loss in loss_dict.items():
                    if key in loss_key:
                        loss *= weight

            loss = sum(loss_dict.values())

        output = {
            'class_queries_logits': class_queries_logits[-1],
            'mask_predictions': mask_predictions[-1],
            'loss': loss,
        }
        if output_hidden_states:
            output['intermediate_class_queries_logits'] = class_queries_logits[:-1]
            output['intermediate_mask_predictions'] = mask_predictions[:-1]
            output['backbone_output'] = backbone_output
            output['transformer_module_output'] = transformer_module_output

        return Mask2FormerOutput(**output)




# from backbones.swin_fpn import SwinFPNBackbone, SwinFPNBackboneConfig
# bb_config = SwinFPNBackboneConfig()
# backbone = SwinFPNBackbone(bb_config)
# config = Mask2FormerConfig()
# model = Mask2FormerPanopticSegmentation(config, backbone)
# input = torch.randn(5,3,224,224)
# op = model(input, output_hidden_states=True)

# print([f.shape for f in op['backbone_output'].feature_pyramid])

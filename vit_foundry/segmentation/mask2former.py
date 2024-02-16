
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from config import Mask2FormerConfig
from components.transformer_module import Mask2FormerTransformerModule
from components.loss import Mask2FormerLoss



class Mask2FormerModel(nn.Module):
    def __init__(
            self,
            config: Mask2FormerConfig,
            backbone: nn.Module,
    ):
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

        self.initialize_weights()
    
    def initialize_weights(self): # Just a really naive weight initialization for now
        for module in self.modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


    def forward(
        self,
        pixel_values: Tensor,
    ):
        backbone_output = self.backbone(pixel_values)

        transformer_module_output = self.transformer_module(
            mask_features=backbone_output.mask_features,
            feature_pyramid=backbone_output.feature_pyramid,
        )

        return (backbone_output, transformer_module_output)


class Mask2FormerPanopticSegmentation(nn.Module):
    def __init__(self, config: Mask2FormerConfig, backbone: nn.Module):
        super().__init__()
        self.config = config
        self.model = Mask2FormerModel(
            config,
            backbone,
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.transformer_hidden_size, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        #self.post_init()

    def get_loss_dict(
        self,
        intermediate_mask_predictions,
        class_queries_logits,
        mask_labels,
        class_labels,
        auxiliary_predictions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            intermediate_mask_predictions=intermediate_mask_predictions,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append({"intermediate_mask_predictions": aux_binary_masks, "class_queries_logits": aux_classes})

        return auxiliary_logits

    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        backbone_output, transformer_module_output = self.model(pixel_values)
        mask_predictions = transformer_module_output.mask_predictions # note, this is from every layer of the transformer
        
        class_queries_logits = []
        for decoder_output in transformer_module_output.hidden_states:
            class_prediction = self.class_predictor(decoder_output)
            class_queries_logits.append(class_prediction)

        loss = None
        if mask_labels is not None and class_labels is not None:
            auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, mask_predictions)
            loss_dict = self.get_loss_dict(
                intermediate_mask_predictions=mask_predictions[-1], # final mask set prediction
                class_queries_logits=class_queries_logits[-1], # final class set prediction
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        output = {
            'loss': loss,
            'class_queries_logits': class_queries_logits[-1],
            'mask_predictions': mask_predictions[-1],
        }
        if output_hidden_states:
            output['intermediate_class_queries_logits'] = class_queries_logits[:-1]
            output['intermediate_mask_predictions'] = mask_predictions[:-1]
            output['backbone_output'] = backbone_output
            output['transformer_module_output'] = transformer_module_output

        return output

from backbones.swin_fpn import SwinFPNBackbone, SwinFPNBackboneConfig
bb_config = SwinFPNBackboneConfig()
backbone = SwinFPNBackbone(bb_config)
config = Mask2FormerConfig()
model = Mask2FormerPanopticSegmentation(config, backbone)
input = torch.randn(5,3,224,224)
op = model(input, output_hidden_states=True)

print([f.shape for f in op['backbone_output'].feature_pyramid])

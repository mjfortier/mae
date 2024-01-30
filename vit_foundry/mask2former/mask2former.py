
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from vit_foundry.mask2former.config import Mask2FormerConfig
from vit_foundry.mask2former.swin import SwinTransformerV2
from vit_foundry.mask2former.pixel_decoder import MaskFormerPixelDecoder
from vit_foundry.mask2former.transformer_module import Mask2FormerTransformerModule
from vit_foundry.mask2former.loss import Mask2FormerLoss



class Mask2FormerModel(nn.Module):
    def __init__(
            self,
            fpn_hidden_size: int = 256,
            transformer_hidden_size: int = 256,
            num_queries: int = 32,
            ):
        super().__init__()
        self.backbone_encoder = SwinTransformerV2()
        self.pixel_decoder = MaskFormerPixelDecoder(
            lateral_widths=self.backbone_encoder.num_features,
            feature_size=fpn_hidden_size,
            mask_feature_size=fpn_hidden_size,
        )
        self.transformer_module = Mask2FormerTransformerModule(
            fpn_hidden_size=fpn_hidden_size,
            hidden_size=transformer_hidden_size,
            num_queries=num_queries,
            )


    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        encoder_features = self.backbone_encoder(pixel_values)
        decoder_output = self.pixel_decoder(encoder_features)

        pixel_level_module_output = {
            'mask_features': decoder_output[0], # FPN final pixel embeddings 
            'multi_scale_features': decoder_output[1], # FPN feature maps
            'encoder_features': encoder_features,  # Swin transformer feature maps
        }

        transformer_module_output = self.transformer_module(
            mask_features=pixel_level_module_output['mask_features'],
            multi_scale_features=pixel_level_module_output['multi_scale_features'],
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return {
            'mask_features': pixel_level_module_output['mask_features'],
            'multi_scale_features': pixel_level_module_output['multi_scale_features'],
            'decoder_output': transformer_module_output,
            'encoder_features': pixel_level_module_output['encoder_features']
        }



class Mask2FormerForUniversalSegmentation(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.model = Mask2FormerModel(
            fpn_hidden_size=config.fpn_hidden_size,
            transformer_hidden_size=config.transformer_hidden_size,
            num_queries=config.num_queries,
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
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs['decoder_output']['intermediate_hidden_states']:
            class_prediction = self.class_predictor(decoder_output)
            class_queries_logits += (class_prediction,)

        intermediate_mask_predictions = outputs['decoder_output']['intermediate_mask_predictions']

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, intermediate_mask_predictions)

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                intermediate_mask_predictions=intermediate_mask_predictions[-1], # final mask set prediction
                class_queries_logits=class_queries_logits[-1], # final class set prediction
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs['encoder_features']
            pixel_decoder_hidden_states = outputs['multi_scale_features']
            transformer_decoder_hidden_states = outputs['decoder_output']['all_hidden_states']

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        return {
            'loss': loss,
            'class_queries_logits': class_queries_logits[-1],
            'intermediate_mask_predictions': intermediate_mask_predictions[-1],
            'auxiliary_logits': auxiliary_logits,
            'encoder_last_hidden_state': outputs['encoder_features'][-1],
            'pixel_decoder_last_hidden_state': outputs['mask_features'],
            'transformer_decoder_last_hidden_state': outputs['decoder_output']['hidden_states'],
            'encoder_hidden_states': encoder_hidden_states,
            'pixel_decoder_hidden_states': pixel_decoder_hidden_states,
            'transformer_decoder_hidden_states': transformer_decoder_hidden_states,
            'attentions': outputs['decoder_output']['attentions'],
        }



""" 
config = Mask2FormerConfig()
model = Mask2FormerForUniversalSegmentation(config)


ip = torch.randn((1,3,224,224))
mask_labels = [torch.randn(2,224,224)]
class_labels = [torch.Tensor([1, 3]).long()]

op = model(ip, mask_labels=mask_labels, class_labels=class_labels, output_hidden_states=True)
 """
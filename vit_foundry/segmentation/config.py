
from typing import List
from dataclasses import dataclass
from torch import Tensor


@dataclass
class Mask2FormerConfig():
    # Transformer module parameters
    transformer_hidden_size: int = 256
    num_queries: int = 32
    num_layers: int = 6 # Each layer contains one sub-layer per feature pyramid map (3L in the paper)
    num_attention_heads: int = 8
    dim_feedforward: int = 2048
    activation_function: str = "relu"

    # Mask2Former inference / loss parameters
    num_labels: int = 10
    dropout: float = 0.0
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    train_num_points: int = 12544
    use_auxiliary_loss: bool = True

    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    output_auxiliary_logits: bool = None


@dataclass
class Mask2FormerBackboneOutput():
    '''
    Backbone includes the feature pyramid component. Examples:
      - Swin Transformer with FPN
      - ViTMAE with simple feature pyramid
    
    S == Feature pyramid hidden size
    '''
    mask_features: Tensor # The final, transformed output from the feature pyramid. (B,S,H/4,W/4)
    feature_pyramid: List[Tensor] # Should be a list of [(B,S,H/16,W/16), (B,S,H/8,W/8), (B,S,H/4,W/4)]
    encoder_features: List[Tensor] # Will vary, hidden states of encoder


@dataclass
class Mask2FormerTransformerOutput():
    '''
    Q == Number of queries
    S == Transformer hidden size
    '''
    hidden_states: Tensor # Final output of the transformer module, (B,Q,S)
    mask_predictions: List[Tensor] # [(B,Q,H/4,W/4), ...]
    attentions : List[Tensor]


@dataclass
class Mask2FormerOutput():
    class_queries_logits: Tensor
    mask_predictions: Tensor
    loss: Tensor
    intermediate_class_queries_logits: List[Tensor] = None
    intermediate_mask_predictions: List[Tensor] = None
    backbone_output: Mask2FormerBackboneOutput = None
    transformer_module_output: Mask2FormerTransformerOutput = None

from sate.mixed_linearizer import linearize_sample
from sate.structural_linearized_stream import StructuralTransformerConfig
from sate.structural_linearized_stream import StructuralTransformer
from sate.structural_linearized_stream import build_mixed_bias
from sate.sate_feature_encoder import SATEFeatureEncoder, SATEConfig
from sate.fusion_gating import SATEWithFusion, FusionConfig
from sate.context_fusion_head import ContextFusionHead
from transformers import T5EncoderModel, T5TokenizerFast
import torch

sample_data = {
    "topic": 8546,
    "ast_fol": [
        {
            "topic": 8546,
            "expression": "( FORALL x ( vehicle ( x ) AND travel_land ( x ) AND travel_water ( x ) IMPLIES amphibious_vehicle ( x ) ) )",
            "tokens": [
                ["(", 0],
                ["FORALL", 1],
                ["x", 2],
                ["(", 3],
                ["vehicle", 4],
                ["(", 5],
                ["x", 6],
                [")", 7],
                ["AND", 8],
                ["travel_land", 9],
                ["(", 10],
                ["x", 11],
                [")", 12],
                ["AND", 13],
                ["travel_water", 14],
                ["(", 15],
                ["x", 16],
                [")", 17],
                ["IMPLIES", 18],
                ["amphibious_vehicle", 19],
                ["(", 20],
                ["x", 21],
                [")", 22],
                [")", 23],
                [")", 24],
            ],
            "type_paths": [
                {
                    "current_id": 1,
                    "current_node": "FORALL",
                    "paths": ["FORALL"],
                    "path_ids": [1],
                },
                {
                    "current_id": 2,
                    "current_node": "x",
                    "paths": ["FORALL", "Variable"],
                    "path_ids": [1, 2],
                },
                {
                    "current_id": 4,
                    "current_node": "vehicle",
                    "paths": ["FORALL", "IMPLIES", "AND", "AND", "Predicate"],
                    "path_ids": [1, 18, 13, 8, 4],
                },
                {
                    "current_id": 6,
                    "current_node": "x",
                    "paths": [
                        "FORALL",
                        "IMPLIES",
                        "AND",
                        "AND",
                        "Predicate",
                        "Variable",
                    ],
                    "path_ids": [1, 18, 13, 8, 4, 6],
                },
                {
                    "current_id": 8,
                    "current_node": "AND",
                    "paths": ["FORALL", "IMPLIES", "AND", "AND"],
                    "path_ids": [1, 18, 13, 8],
                },
                {
                    "current_id": 9,
                    "current_node": "travel_land",
                    "paths": ["FORALL", "IMPLIES", "AND", "AND", "Predicate"],
                    "path_ids": [1, 18, 13, 8, 9],
                },
                {
                    "current_id": 11,
                    "current_node": "x",
                    "paths": [
                        "FORALL",
                        "IMPLIES",
                        "AND",
                        "AND",
                        "Predicate",
                        "Variable",
                    ],
                    "path_ids": [1, 18, 13, 8, 9, 11],
                },
                {
                    "current_id": 13,
                    "current_node": "AND",
                    "paths": ["FORALL", "IMPLIES", "AND"],
                    "path_ids": [1, 18, 13],
                },
                {
                    "current_id": 14,
                    "current_node": "travel_water",
                    "paths": ["FORALL", "IMPLIES", "AND", "Predicate"],
                    "path_ids": [1, 18, 13, 14],
                },
                {
                    "current_id": 16,
                    "current_node": "x",
                    "paths": ["FORALL", "IMPLIES", "AND", "Predicate", "Variable"],
                    "path_ids": [1, 18, 13, 14, 16],
                },
                {
                    "current_id": 18,
                    "current_node": "IMPLIES",
                    "paths": ["FORALL", "IMPLIES"],
                    "path_ids": [1, 18],
                },
                {
                    "current_id": 19,
                    "current_node": "amphibious_vehicle",
                    "paths": ["FORALL", "IMPLIES", "Predicate"],
                    "path_ids": [1, 18, 19],
                },
                {
                    "current_id": 21,
                    "current_node": "x",
                    "paths": ["FORALL", "IMPLIES", "Predicate", "Variable"],
                    "path_ids": [1, 18, 19, 21],
                },
            ],
            "value_paths": [
                {
                    "current_id": 1,
                    "current_node": "FORALL",
                    "paths": ["FORALL"],
                    "path_ids": [1],
                },
                {
                    "current_id": 2,
                    "current_node": "x",
                    "paths": ["FORALL", "x"],
                    "path_ids": [1, 2],
                },
                {
                    "current_id": 4,
                    "current_node": "vehicle",
                    "paths": ["FORALL", "IMPLIES", "AND", "AND", "vehicle"],
                    "path_ids": [1, 18, 13, 8, 4],
                },
                {
                    "current_id": 6,
                    "current_node": "x",
                    "paths": ["FORALL", "IMPLIES", "AND", "AND", "vehicle", "x"],
                    "path_ids": [1, 18, 13, 8, 4, 6],
                },
                {
                    "current_id": 8,
                    "current_node": "AND",
                    "paths": ["FORALL", "IMPLIES", "AND", "AND"],
                    "path_ids": [1, 18, 13, 8],
                },
                {
                    "current_id": 9,
                    "current_node": "travel_land",
                    "paths": ["FORALL", "IMPLIES", "AND", "AND", "travel_land"],
                    "path_ids": [1, 18, 13, 8, 9],
                },
                {
                    "current_id": 11,
                    "current_node": "x",
                    "paths": ["FORALL", "IMPLIES", "AND", "AND", "travel_land", "x"],
                    "path_ids": [1, 18, 13, 8, 9, 11],
                },
                {
                    "current_id": 13,
                    "current_node": "AND",
                    "paths": ["FORALL", "IMPLIES", "AND"],
                    "path_ids": [1, 18, 13],
                },
                {
                    "current_id": 14,
                    "current_node": "travel_water",
                    "paths": ["FORALL", "IMPLIES", "AND", "travel_water"],
                    "path_ids": [1, 18, 13, 14],
                },
                {
                    "current_id": 16,
                    "current_node": "x",
                    "paths": ["FORALL", "IMPLIES", "AND", "travel_water", "x"],
                    "path_ids": [1, 18, 13, 14, 16],
                },
                {
                    "current_id": 18,
                    "current_node": "IMPLIES",
                    "paths": ["FORALL", "IMPLIES"],
                    "path_ids": [1, 18],
                },
                {
                    "current_id": 19,
                    "current_node": "amphibious_vehicle",
                    "paths": ["FORALL", "IMPLIES", "amphibious_vehicle"],
                    "path_ids": [1, 18, 19],
                },
                {
                    "current_id": 21,
                    "current_node": "x",
                    "paths": ["FORALL", "IMPLIES", "amphibious_vehicle", "x"],
                    "path_ids": [1, 18, 19, 21],
                },
            ],
        }
    ],
    "ast_nl": [
        {
            "topic": 8546,
            "expression": "An amphibious vehicle is define as a vehicle capable of travel on both land and water",
            "tokens": [
                ["define", 0],
                ["vehicle", 1],
                ["amphibious", 2],
                ["vehicle", 3],
                ["capable", 4],
                ["travel", 5],
                ["and", 6],
                ["land", 7],
                ["water", 8],
            ],
            "value_paths": [
                {
                    "current_id": 0,
                    "current_node": "define",
                    "paths": ["define"],
                    "path_ids": [0],
                },
                {
                    "current_id": 1,
                    "current_node": "vehicle",
                    "paths": ["define", "vehicle"],
                    "path_ids": [0, 1],
                },
                {
                    "current_id": 2,
                    "current_node": "amphibious",
                    "paths": ["define", "vehicle", "amphibious"],
                    "path_ids": [0, 1, 2],
                },
                {
                    "current_id": 3,
                    "current_node": "vehicle",
                    "paths": ["define", "vehicle"],
                    "path_ids": [0, 1],
                },
                {
                    "current_id": 4,
                    "current_node": "capable",
                    "paths": ["define", "vehicle", "capable"],
                    "path_ids": [0, 1, 4],
                },
                {
                    "current_id": 5,
                    "current_node": "travel",
                    "paths": ["define", "vehicle", "travel"],
                    "path_ids": [0, 1, 5],
                },
                {
                    "current_id": 6,
                    "current_node": "and",
                    "paths": ["define", "vehicle", "travel", "and"],
                    "path_ids": [0, 1, 5, 6],
                },
                {
                    "current_id": 7,
                    "current_node": "land",
                    "paths": ["define", "vehicle", "travel", "and", "land"],
                    "path_ids": [0, 1, 5, 6, 7],
                },
                {
                    "current_id": 8,
                    "current_node": "water",
                    "paths": ["define", "vehicle", "travel", "and", "water"],
                    "path_ids": [0, 1, 5, 6, 8],
                },
            ],
        },
        {
            "topic": 8546,
            "expression": "Therefore , if a vehicle can travel on land and water , it is an amphibious vehicle",
            "tokens": [
                ["infer", 0],
                ["vehicle", 1],
                ["possible", 2],
                ["travel", 3],
                ["vehicle", 4],
                ["and", 5],
                ["land", 6],
                ["water", 7],
                ["amphibious", 8],
            ],
            "value_paths": [
                {
                    "current_id": 0,
                    "current_node": "infer",
                    "paths": ["infer"],
                    "path_ids": [0],
                },
                {
                    "current_id": 1,
                    "current_node": "vehicle",
                    "paths": ["infer", "vehicle"],
                    "path_ids": [0, 1],
                },
                {
                    "current_id": 2,
                    "current_node": "possible",
                    "paths": ["infer", "vehicle", "possible"],
                    "path_ids": [0, 1, 2],
                },
                {
                    "current_id": 3,
                    "current_node": "travel",
                    "paths": ["infer", "vehicle", "possible", "travel"],
                    "path_ids": [0, 1, 2, 3],
                },
                {
                    "current_id": 4,
                    "current_node": "vehicle",
                    "paths": ["infer", "vehicle"],
                    "path_ids": [0, 1],
                },
                {
                    "current_id": 5,
                    "current_node": "and",
                    "paths": ["infer", "vehicle", "possible", "travel", "and"],
                    "path_ids": [0, 1, 2, 3, 5],
                },
                {
                    "current_id": 6,
                    "current_node": "land",
                    "paths": ["infer", "vehicle", "possible", "travel", "and", "land"],
                    "path_ids": [0, 1, 2, 3, 5, 6],
                },
                {
                    "current_id": 7,
                    "current_node": "water",
                    "paths": ["infer", "vehicle", "possible", "travel", "and", "water"],
                    "path_ids": [0, 1, 2, 3, 5, 7],
                },
                {
                    "current_id": 8,
                    "current_node": "amphibious",
                    "paths": ["infer", "vehicle", "amphibious"],
                    "path_ids": [0, 1, 8],
                },
            ],
        },
    ],
}


# 1) Linearize
item_mix = linearize_sample(sample_data)

# 2) Models
t5_name = "t5-base"

sate = SATEFeatureEncoder(SATEConfig(t5_name=t5_name, fine_tune_t5=False))
struct = StructuralTransformer(
    StructuralTransformerConfig(
        d_model=sate.cfg.d_ast,
        nhead=8,
        num_layers=2,
        dim_ff=2048,
        dropout=0.1,
        bias_scale_init=1.0,
        bias_channels=4,
    )
)
model = SATEWithFusion(
    sate, struct, FusionConfig(per_dim_alpha=True), bias_builder=build_mixed_bias
)

# 3) Forward SATE+Fusion
out_fusion = model(item_mix)  # h_fused[L,D], semantic_mask[L], ...

# 4) Context encoder trên NL⟂...⟂FOL
tokz = T5TokenizerFast.from_pretrained(t5_name)
enc = T5EncoderModel.from_pretrained(t5_name).to(next(model.parameters()).device)
batch = tokz(item_mix["expressions_cat"], return_tensors="pt")
batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
with torch.no_grad():
    ctx = enc(**batch).last_hidden_state.unsqueeze(0)  # [1,S,D]

# 5) Context fusion
head = ContextFusionHead(
    d_model=out_fusion["h_fused"].shape[-1],
    use_proj=True,
    dropout=0.1,
    sep_token=" <extra_id_0> ",
    token_joiner=" ",
)

res = head(
    h_fused=out_fusion["h_fused"],  # [L,D]
    tokens=item_mix["tokens"],  # [(tok, gid)] theo NL→FOL
    seg_meta=item_mix["seg_meta"],  # [(seg_id, modality)]
    tokenizer=tokz,
    semantic_mask=out_fusion["semantic_mask"],  # [L]
    t5_encoder=enc,  # tuỳ chọn: nếu truyền, head trả luôn context_plus
)

context_plus = res["context_plus"]  # [1,S,D] (nếu có t5_encoder)
bias_expanded = res["bias_expanded"]  # [1,S,D]
input_ids = res["input_ids"]  # để tái dùng/ghi cache
attn_mask = res["attention_mask"]
context_text = res["context_text"]  # "tok tok ... <extra_id_0> tok tok ..."

import argparse
import json

import torch
import torch.nn.functional as F

from ldm.modules.diffusionmodules.scene_graph_grounding_net import PositionNet


def parameter_grad_norm(module):
    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().pow(2).sum().item()
    return squared_norm**0.5


def build_inputs(batch_size, num_objects, num_relations, device):
    boxes = torch.rand(batch_size, num_objects, 4, device=device)
    lower = torch.minimum(boxes[..., :2], boxes[..., 2:])
    upper = torch.maximum(boxes[..., :2], boxes[..., 2:])
    boxes = torch.cat([lower, upper], dim=-1)
    masks = torch.ones(batch_size, num_objects, device=device)
    positive_embeddings = torch.randn(batch_size, num_objects, 768, device=device)

    relation_edges = torch.zeros(
        batch_size, num_relations, 2, dtype=torch.long, device=device
    )
    relation_edges[..., 0] = torch.randint(
        0, num_objects, (batch_size, num_relations), device=device
    )
    relation_edges[..., 1] = torch.randint(
        0, num_objects, (batch_size, num_relations), device=device
    )
    relation_masks = torch.ones(batch_size, num_relations, device=device)
    relation_embeddings = torch.randn(
        batch_size, num_relations, 768, device=device
    )
    relation_geo_features = torch.randn(
        batch_size, num_relations, 12, device=device
    )
    return {
        "boxes": boxes,
        "masks": masks,
        "positive_embeddings": positive_embeddings,
        "relation_edges": relation_edges,
        "relation_masks": relation_masks,
        "relation_embeddings": relation_embeddings,
        "relation_geo_features": relation_geo_features,
    }


def audit(graph_gate_init, device, prediction_source):
    torch.manual_seed(123)
    model = PositionNet(
        in_dim=768,
        out_dim=768,
        hidden_dim=768,
        base_hidden_dim=512,
        gat_layers=1,
        relation_dim=768,
        relation_geo_dim=12,
        graph_gate_init=graph_gate_init,
    ).to(device)
    inputs = build_inputs(
        batch_size=2,
        num_objects=6,
        num_relations=8,
        device=device,
    )

    if prediction_source == "masked_graph_delta":
        prediction = model.predict_relation_geo_from_masked_graph(**inputs)
    else:
        object_tokens = model(**inputs)
        prediction = model.predict_relation_geo(
            object_tokens,
            inputs["relation_edges"],
            relation_embeddings=inputs["relation_embeddings"],
        )
    loss = F.smooth_l1_loss(prediction, inputs["relation_geo_features"])
    loss.backward()

    gate_gradient = model.graph_gate.grad
    return {
        "graph_gate_logit": float(model.graph_gate.detach()),
        "graph_gate_sigmoid": float(torch.sigmoid(model.graph_gate.detach())),
        "loss": float(loss.detach()),
        "base_linears_grad_norm": parameter_grad_norm(model.linears),
        "gat_grad_norm": parameter_grad_norm(model.gat_layers),
        "predictor_grad_norm": parameter_grad_norm(model.relation_geo_predictor),
        "graph_gate_grad_abs": (
            float(gate_gradient.detach().abs())
            if gate_gradient is not None
            else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    results = {}
    for prediction_source in ("final_tokens", "masked_graph_delta"):
        source_results = {
            "gate_init_minus4": audit(-4.0, args.device, prediction_source),
            "gate_init_zero": audit(0.0, args.device, prediction_source),
        }
        low_gate = source_results["gate_init_minus4"]["gat_grad_norm"]
        open_gate = source_results["gate_init_zero"]["gat_grad_norm"]
        source_results["gat_gradient_ratio_minus4_vs_zero"] = low_gate / max(
            open_gate,
            1e-12,
        )
        results[prediction_source] = source_results
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

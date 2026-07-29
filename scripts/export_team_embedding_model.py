"""One-time export: a small pretrained CNN backbone -> ONNX.

Produces the embedding model used by
``agon.team.embedding_team_assigner.EmbeddingTeamClassifier`` to
cluster player crops into teams (see that module's docstring for why this
is more robust than raw-pixel KMeans).

Needs the ``[train]`` extra (torch + torchvision) to run -- same pattern as
exporting a YOLO checkpoint to ONNX: torch is only needed for this one-time
conversion, not for the resulting .onnx file's runtime inference.

Usage:
    python scripts/export_team_embedding_model.py [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/team_embedding.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone.eval()

    class EmbeddingModel(nn.Module):
        """Backbone conv features -> global average pool -> flatten.

        Deliberately stops before MobileNetV3's classification head: we want
        a general-purpose visual embedding (576-dim for the "small"
        variant), not ImageNet class logits.
        """

        def __init__(self, backbone: nn.Module):
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)

    model = EmbeddingModel(backbone)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print(f"Exported embedding model to {output_path}")


if __name__ == "__main__":
    main()

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required for GAT behavior tests")
class SpatialRelationGATLayerTest(unittest.TestCase):
    def test_nodes_without_valid_edges_are_unchanged(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            SpatialRelationGATLayer,
        )

        layer = SpatialRelationGATLayer(
            token_dim=4,
            relation_dim=2,
            relation_geo_dim=1,
            hidden_dim=4,
        )
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.zero_()
            layer.update[4].bias.fill_(1.0)

        tokens = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
        )
        relation_edges = torch.zeros(1, 1, 2)
        relation_masks = torch.zeros(1, 1)
        relation_embeddings = torch.zeros(1, 1, 2)
        relation_geo_features = torch.zeros(1, 1, 1)

        output = layer(
            tokens,
            relation_edges,
            relation_masks,
            relation_embeddings,
            relation_geo_features,
        )

        torch.testing.assert_close(output, tokens)


@unittest.skipUnless(torch is not None, "PyTorch is required for triplet-fuser tests")
class RelationTripletFuserTest(unittest.TestCase):
    def test_zero_initialized_fuser_is_identity_delta(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            RelationTripletFuser,
        )

        fuser = RelationTripletFuser(
            token_dim=4,
            relation_dim=2,
            relation_geo_dim=1,
            hidden_dim=8,
        )
        tokens = torch.randn(1, 3, 4)
        delta = fuser(
            tokens,
            relation_edges=torch.tensor([[[0, 1]]]),
            relation_masks=torch.tensor([[1.0]]),
            relation_embeddings=torch.randn(1, 1, 2),
            relation_geo_features=torch.randn(1, 1, 1),
            object_masks=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        torch.testing.assert_close(delta, torch.zeros_like(tokens))

    def test_fuser_only_updates_declared_endpoints(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            RelationTripletFuser,
        )

        fuser = RelationTripletFuser(
            token_dim=2,
            relation_dim=1,
            relation_geo_dim=1,
            hidden_dim=4,
        )
        with torch.no_grad():
            fuser.net[-1].bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        tokens = torch.zeros(1, 3, 2)
        delta = fuser(
            tokens,
            relation_edges=torch.tensor([[[0, 1]]]),
            relation_masks=torch.tensor([[1.0]]),
            relation_embeddings=torch.zeros(1, 1, 1),
            relation_geo_features=torch.zeros(1, 1, 1),
            object_masks=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        torch.testing.assert_close(delta[0, 0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(delta[0, 1], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(delta[0, 2], torch.zeros(2))


@unittest.skipUnless(torch is not None, "PyTorch is required for GAT behavior tests")
class GraphResidualLimitTest(unittest.TestCase):
    def test_caps_graph_delta_relative_to_base_token_norm(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            limit_graph_delta_norm,
        )

        base_tokens = torch.tensor([[[3.0, 4.0]]])
        graph_delta = torch.tensor([[[60.0, 80.0]]])

        limited = limit_graph_delta_norm(
            graph_delta,
            base_tokens,
            max_ratio=2.0,
        )

        self.assertAlmostEqual(limited.norm().item(), 10.0, places=5)

    def test_leaves_small_graph_delta_unchanged(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            limit_graph_delta_norm,
        )

        base_tokens = torch.tensor([[[3.0, 4.0]]])
        graph_delta = torch.tensor([[[0.6, 0.8]]])

        limited = limit_graph_delta_norm(
            graph_delta,
            base_tokens,
            max_ratio=2.0,
        )

        torch.testing.assert_close(limited, graph_delta)


@unittest.skipUnless(torch is not None, "PyTorch is required for GAT behavior tests")
class GraphResidualNormMatchingTest(unittest.TestCase):
    def test_matches_valid_delta_to_target_base_norm_ratio(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            match_graph_delta_norm,
        )

        base_tokens = torch.tensor([[[3.0, 4.0], [0.0, 10.0]]])
        graph_delta = torch.tensor([[[6.0, 8.0], [3.0, 4.0]]])

        matched = match_graph_delta_norm(
            graph_delta,
            base_tokens,
            target_ratio=0.08,
        )

        expected_norms = base_tokens.norm(dim=-1) * 0.08
        torch.testing.assert_close(matched.norm(dim=-1), expected_norms)

    def test_keeps_zero_delta_zero(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            match_graph_delta_norm,
        )

        base_tokens = torch.tensor([[[3.0, 4.0]]])
        graph_delta = torch.zeros_like(base_tokens)

        matched = match_graph_delta_norm(
            graph_delta,
            base_tokens,
            target_ratio=0.08,
        )

        torch.testing.assert_close(matched, graph_delta)

    def test_masks_padding_tokens(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            match_graph_delta_norm,
        )

        base_tokens = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
        graph_delta = torch.ones_like(base_tokens)
        masks = torch.tensor([[1.0, 0.0]])

        matched = match_graph_delta_norm(
            graph_delta,
            base_tokens,
            target_ratio=0.08,
            masks=masks,
        )

        self.assertGreater(matched[0, 0].norm().item(), 0.0)
        torch.testing.assert_close(matched[0, 1], torch.zeros(2))


@unittest.skipUnless(torch is not None, "PyTorch is required for GAT behavior tests")
class RelationGeometryAuxiliaryPathTest(unittest.TestCase):
    def _build_model(self, graph_gate_init):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import PositionNet

        return PositionNet(
            in_dim=4,
            out_dim=4,
            hidden_dim=8,
            base_hidden_dim=8,
            fourier_freqs=1,
            gat_layers=1,
            relation_dim=3,
            relation_geo_dim=2,
            graph_gate_init=graph_gate_init,
        )

    def _inputs(self):
        return {
            "boxes": torch.tensor(
                [[[0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]]]
            ),
            "masks": torch.ones(1, 2),
            "positive_embeddings": torch.randn(1, 2, 4),
            "relation_edges": torch.tensor([[[0, 1]]]),
            "relation_masks": torch.ones(1, 1),
            "relation_embeddings": torch.randn(1, 1, 3),
            "relation_geo_features": torch.tensor([[[0.25, -0.5]]]),
        }

    def test_masks_valid_relation_geometry_before_auxiliary_encoding(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import (
            mask_relation_geo_features_for_prediction,
        )

        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        masks = torch.tensor([[1.0, 0.0]])

        masked = mask_relation_geo_features_for_prediction(features, masks)

        torch.testing.assert_close(masked, torch.zeros_like(features))

    def test_auxiliary_graph_gradient_is_independent_of_global_gate(self):
        torch.manual_seed(7)
        low_gate_model = self._build_model(-4.0)
        open_gate_model = self._build_model(0.0)
        open_gate_model.load_state_dict(low_gate_model.state_dict())
        with torch.no_grad():
            low_gate_model.graph_gate.fill_(-4.0)
            open_gate_model.graph_gate.fill_(0.0)

        low_inputs = self._inputs()
        open_inputs = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in low_inputs.items()
        }

        low_prediction = low_gate_model.predict_relation_geo_from_masked_graph(
            **low_inputs
        )
        open_prediction = open_gate_model.predict_relation_geo_from_masked_graph(
            **open_inputs
        )
        low_prediction.sum().backward()
        open_prediction.sum().backward()

        low_grad = low_gate_model.gat_layers[0].src_msg[0].weight.grad
        open_grad = open_gate_model.gat_layers[0].src_msg[0].weight.grad
        torch.testing.assert_close(low_prediction, open_prediction)
        torch.testing.assert_close(low_grad, open_grad)
        self.assertGreater(low_grad.norm().item(), 0.0)

    def test_masked_geometry_auxiliary_reaches_triplet_fuser(self):
        from ldm.modules.diffusionmodules.scene_graph_grounding_net import PositionNet

        model = PositionNet(
            in_dim=4,
            out_dim=4,
            hidden_dim=8,
            base_hidden_dim=8,
            fourier_freqs=1,
            gat_layers=0,
            relation_dim=3,
            relation_geo_dim=2,
            use_triplet_fuser=True,
            triplet_fuser_hidden_dim=8,
        )
        prediction = model.predict_relation_geo_from_masked_graph(**self._inputs())
        prediction.sum().backward()

        final_projection_grad = model.triplet_fuser.net[-1].weight.grad
        self.assertIsNotNone(final_projection_grad)
        self.assertGreater(final_projection_grad.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

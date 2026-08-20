import unittest

import torch

from zoology.diagnostics import (
    model_gradients_sha256,
    model_parameters_sha256,
    named_tensors_sha256,
    rng_state_sha256,
    tensor_sha256,
)


class DiagnosticsTest(unittest.TestCase):
    def test_tensor_hash_is_stable_and_value_sensitive(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(tensor_sha256(tensor), tensor_sha256(tensor.clone()))

        changed = tensor.clone()
        changed[0, 0] += 1
        self.assertNotEqual(tensor_sha256(tensor), tensor_sha256(changed))

    def test_named_tensor_hash_includes_names_and_none(self):
        tensor = torch.arange(4)
        first = named_tensors_sha256([("a", tensor), ("b", None)])
        reordered = named_tensors_sha256([("b", None), ("a", tensor)])
        renamed = named_tensors_sha256([("c", tensor), ("b", None)])
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, renamed)

    def test_model_and_gradient_hashes(self):
        torch.manual_seed(123)
        model_a = torch.nn.Linear(4, 3)
        torch.manual_seed(123)
        model_b = torch.nn.Linear(4, 3)
        self.assertEqual(
            model_parameters_sha256(model_a),
            model_parameters_sha256(model_b),
        )
        self.assertEqual(
            model_gradients_sha256(model_a),
            model_gradients_sha256(model_b),
        )

        model_a(torch.ones(2, 4)).sum().backward()
        self.assertNotEqual(
            model_gradients_sha256(model_a),
            model_gradients_sha256(model_b),
        )

    def test_rng_hash_repeats_after_reseeding(self):
        torch.manual_seed(123)
        first = rng_state_sha256()
        torch.rand(8)
        self.assertNotEqual(first, rng_state_sha256())
        torch.manual_seed(123)
        self.assertEqual(first, rng_state_sha256())


if __name__ == "__main__":
    unittest.main()

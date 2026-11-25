# Copyright © 2026 MLX Contributors

import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.dspark import Sampling, verify
from mlx_lm.sample_utils import make_sampler


class TestDSparkSampling(unittest.TestCase):
    def test_distribution_matches_ordinary_sampler(self):
        cases = (
            ([0.6, 0.3, 0.1], 0.5, 0.75, 0, [0.8, 0.2, 0.0]),
            ([0.6, 0.3, 0.1], 1.0, 0.65, 2, [2 / 3, 1 / 3, 0.0]),
            ([0.5, 0.5, 0.0], 1e-8, 1.0, 0, [0.5, 0.5, 0.0]),
        )
        for probabilities, temp, top_p, top_k, expected in cases:
            with self.subTest(temp=temp, top_p=top_p, top_k=top_k):
                logits = mx.log(mx.array(probabilities))
                logprobs = logits - mx.logsumexp(logits)
                ordinary = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
                with patch(
                    "mlx_lm.sample_utils.categorical_sampling",
                    side_effect=lambda x, t: mx.softmax(x * (1.0 / t), axis=-1),
                ):
                    ordinary_probs = ordinary(logprobs)
                sampling = Sampling(temperature=temp, top_p=top_p, top_k=top_k)
                dspark_probs = mx.exp(sampling.log_distribution(logits))
                expected_probs = mx.array(expected)
                self.assertTrue(
                    mx.allclose(ordinary_probs, expected_probs, atol=1e-6).item()
                )
                self.assertTrue(
                    mx.allclose(dspark_probs, expected_probs, atol=1e-6).item()
                )
                self.assertEqual(
                    (dspark_probs == 0).tolist(), (expected_probs == 0).tolist()
                )

    def test_verify_uses_filtered_target_distribution(self):
        sampling = Sampling(temperature=0.5, top_p=0.75)
        logits = mx.log(mx.array([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]]))
        proposals = mx.array([0])
        draft_logprobs = [mx.array([0.0, -mx.inf, -mx.inf])]

        with patch.object(mx.random, "uniform", return_value=mx.array([0.9])):
            self.assertEqual(
                verify(logits, proposals, draft_logprobs, sampling), (0, 1)
            )

        with (
            patch.object(mx.random, "uniform", return_value=mx.array([0.1])),
            patch.object(mx.random, "categorical", return_value=mx.array(0)) as sample,
        ):
            self.assertEqual(
                verify(logits, proposals, draft_logprobs, sampling), (1, 0)
            )
        bonus_probs = mx.exp(sample.call_args.args[0])
        self.assertTrue(
            mx.allclose(bonus_probs, mx.array([0.8, 0.2, 0.0]), atol=1e-6).item()
        )

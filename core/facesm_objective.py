"""Reusable FaceSM objective, adapted from paper/core/facesm_attack_core.py.

Models are frozen surrogates; gradients flow through both views to the input.
A separate wrapper per pair avoids changing model state or global loss functions.
"""
import math
import tensorflow as tf


def compute_embedding(model, x, multi_view=False):
    out = model(x, training=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    emb = tf.nn.l2_normalize(out, axis=1)
    if not multi_view:
        return emb
    out_flip = model(tf.image.flip_left_right(x), training=False)
    if isinstance(out_flip, (tuple, list)):
        out_flip = out_flip[0]
    emb_flip = tf.nn.l2_normalize(out_flip, axis=1)
    return tf.nn.l2_normalize(0.5 * (emb + emb_flip), axis=1)


def attack_loss_sm(cos_t, cos_s, attack_type, source_lambda):
    if attack_type == 'impersonation_attack':
        score = cos_t - source_lambda * cos_s
    elif attack_type == 'dodging_attack':
        score = (1 - cos_t) + source_lambda * (1 - cos_s)
    else:
        raise ValueError(f'Unsupported attack type: {attack_type}')
    return tf.reduce_mean(score)


class FaceSMModel:
    """Mirror-fused surrogate with a fixed original-source reference."""

    def __init__(self, model, source, source_lambda=0.20):
        if not math.isfinite(source_lambda) or source_lambda < 0:
            raise ValueError('source_lambda must be finite and nonnegative')
        self.model = model
        self.source_lambda = float(source_lambda)
        self.source_embedding = tf.stop_gradient(self(source))

    def __call__(self, x, training=False):
        return compute_embedding(self.model, x, multi_view=True)

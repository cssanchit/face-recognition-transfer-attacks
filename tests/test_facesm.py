import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
from core import transfer_attack_core as core
from core.facesm_objective import FaceSMModel, compute_embedding, attack_loss_sm


class FaceSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(42)
        cls.model = tf.keras.Sequential([
            tf.keras.layers.Input((16, 16, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(12, activation='tanh'),
        ])
        cls.source = tf.random.uniform((1, 16, 16, 3), -0.7, 0.7)
        cls.target = tf.random.uniform((1, 16, 16, 3), -0.7, 0.7)

    def test_loss_equations(self):
        ct, cs = tf.constant([0.2, 0.6]), tf.constant([0.8, 0.3])
        self.assertAlmostEqual(float(attack_loss_sm(ct, cs, 'impersonation_attack', .2)), .29, places=6)
        self.assertAlmostEqual(float(attack_loss_sm(ct, cs, 'dodging_attack', .2)), .69, places=6)

    def test_fusion_and_gradient_match_explicit_formula(self):
        wrapped = FaceSMModel(self.model, self.source)
        target = compute_embedding(wrapped, self.target)
        x = (self.source + self.target) / 2
        for kind in ['impersonation_attack', 'dodging_attack']:
            with tf.GradientTape() as tape:
                tape.watch(x)
                actual = core.verification_loss(wrapped, compute_embedding(wrapped, x), target, kind)
            grad = tape.gradient(actual, x)
            with tf.GradientTape() as tape:
                tape.watch(x)
                e = compute_embedding(self.model, x, True)
                t = compute_embedding(self.model, self.target, True)
                s = compute_embedding(self.model, self.source, True)
                ct, cs = tf.reduce_sum(e*t, 1), tf.reduce_sum(e*s, 1)
                expected = tf.reduce_mean(ct-.2*cs if kind == 'impersonation_attack' else 1-ct+.2*(1-cs))
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            np.testing.assert_allclose(grad, tape.gradient(expected, x), atol=1e-6)
            self.assertTrue(np.isfinite(grad).all())
            self.assertGreater(float(tf.norm(grad)), 0)
        np.testing.assert_allclose(wrapped(x), wrapped(tf.image.flip_left_right(x)), atol=1e-6)

    def test_mirror_only_and_augmented_batch(self):
        wrapped = FaceSMModel(self.model, self.source, 0)
        target = compute_embedding(wrapped, self.target)
        e = compute_embedding(wrapped, tf.repeat(self.source, 3, axis=0))
        for kind in ['impersonation_attack', 'dodging_attack']:
            actual = core.verification_loss(wrapped, e, tf.repeat(target, 3, axis=0), kind)
            expected = core.attack_loss(tf.reduce_sum(e*target, 1), kind)
            np.testing.assert_allclose(actual, expected)

    def test_attack_bounds_and_vanilla_isolation(self):
        for attack in ['PGD', 'MI_FGSM', 'TI_FGSM', 'SI_NI_FGSM', 'BPA_CNN']:
            for kind in ['impersonation_attack', 'dodging_attack']:
                tf.random.set_seed(12)
                before = core.run_attack(attack, self.model, self.source, self.target, kind, (16,16))
                adv = core.run_attack(attack, self.model, self.source, self.target, kind, (16,16), objective='facesm')
                self.assertTrue(np.isfinite(adv).all())
                self.assertLessEqual(float(tf.reduce_max(tf.abs(adv-self.source))), core.EPSILON+1e-6)
                self.assertLessEqual(float(tf.reduce_max(tf.abs(adv))), 1)
                tf.random.set_seed(12)
                after = core.run_attack(attack, self.model, self.source, self.target, kind, (16,16))
                np.testing.assert_allclose(before, after)

    def test_all_facesm_backbones(self):
        old_steps = core.NUM_ITER
        core.NUM_ITER = 1
        model = tf.keras.Sequential([
            tf.keras.layers.Input((32, 32, 3)), tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(8, activation='tanh'),
        ])
        source = tf.random.uniform((1, 32, 32, 3), -.5, .5)
        target = tf.random.uniform((1, 32, 32, 3), -.5, .5)
        try:
            for attack in core.FACESM_ATTACKS:
                for kind in ['impersonation_attack', 'dodging_attack']:
                    with self.subTest(attack=attack, kind=kind):
                        adv = core.run_attack(attack, model, source, target, kind,
                                              (32, 32), objective='facesm')
                        self.assertTrue(np.isfinite(adv).all())
                        self.assertLessEqual(float(tf.reduce_max(tf.abs(adv-source))), core.EPSILON+1e-6)
                        self.assertLessEqual(float(tf.reduce_max(tf.abs(adv))), 1)
        finally:
            core.NUM_ITER = old_steps

    def test_cli_both_writes_separate_outputs(self):
        from experiments import generate_adversarial_examples as cli
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.full((16,16,3), 100, dtype=np.uint8)).save(root/'source.png')
            Image.fromarray(np.full((16,16,3), 150, dtype=np.uint8)).save(root/'target.png')
            pd.DataFrame([dict(row_id=1, img1='source.png', img2='target.png',
                               dataset='synthetic', attack_type='impersonation_attack')]).to_csv(root/'pairs.csv', index=False)
            args = ['generate', '--input-csv', str(root/'pairs.csv'), '--dataset-root', str(root),
                    '--output-root', str(root/'outputs'), '--attacker-model', 'ArcFace',
                    '--attacks', 'MI_FGSM', '--objective', 'both']
            with patch.object(sys, 'argv', args), patch.object(cli, 'build_attacker', return_value=self.model), patch.dict(cli.ATTACKER_MODELS, ArcFace=(16,16)):
                cli.main()
            result = pd.read_csv(root/'outputs'/'ArcFace_both_adv_paths.csv')
            for column in ['mi_fgsm_path', 'mi_fgsm_sm_path']:
                self.assertTrue(Path(result.loc[0, column]).is_file())
            self.assertNotEqual(result.loc[0, 'mi_fgsm_path'], result.loc[0, 'mi_fgsm_sm_path'])
            self.assertEqual(result.loc[0, 'objective'], 'both')
            self.assertEqual(result.loc[0, 'source_lambda'], .2)

    def test_invalid_options(self):
        for weight in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): FaceSMModel(self.model, self.source, weight)
        for options in [dict(objective='bad'), dict(objective='facesm',source_lambda=-1)]:
            with self.assertRaises(ValueError):
                core.run_attack('MI_FGSM', self.model, self.source, self.target, 'impersonation_attack', (16,16), **options)


if __name__ == '__main__': unittest.main()

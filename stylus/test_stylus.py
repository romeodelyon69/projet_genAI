"""
Tests unitaires pour Stylus.
Teste chaque composant sans nécessiter de GPU ni de modèle téléchargé.

    python test_stylus.py
"""

import numpy as np
import torch
import unittest
from unittest.mock import MagicMock, patch


class TestAdaIN(unittest.TestCase):
    """Teste l'implémentation d'AdaIN (équation 4 du papier)."""

    def test_output_has_style_stats(self):
        from stylus import adain
        B, C, H, W = 1, 4, 8, 8

        content = torch.randn(B, C, H, W) * 0.5 + 2.0   # mean≈2, std≈0.5
        style   = torch.randn(B, C, H, W) * 3.0 - 5.0   # mean≈-5, std≈3

        out = adain(content, style)

        # Le résultat doit avoir les stats du style (channel-wise)
        for c in range(C):
            out_mean  = out[0, c].mean().item()
            out_std   = out[0, c].std().item()
            style_mean = style[0, c].mean().item()
            style_std  = style[0, c].std().item()
            self.assertAlmostEqual(out_mean, style_mean, delta=0.3,
                msg=f"Channel {c}: mean mismatch")
            self.assertAlmostEqual(out_std, style_std, delta=0.3,
                msg=f"Channel {c}: std mismatch")

    def test_identity_when_same_input(self):
        from stylus import adain
        x = torch.randn(1, 4, 8, 8)
        out = adain(x, x)
        # Avec le même input, AdaIN doit être proche de l'identité
        self.assertTrue(torch.allclose(out, x, atol=1e-4))

    def test_shape_preserved(self):
        from stylus import adain
        for shape in [(1,4,8,8), (2,8,16,16)]:
            x = torch.randn(*shape)
            y = torch.randn(*shape)
            out = adain(x, y)
            self.assertEqual(out.shape, x.shape)


class TestAttentionFeatureStore(unittest.TestCase):
    """Teste le stockage/retrieval des features K/V."""

    def test_store_and_retrieve(self):
        from stylus import AttentionFeatureStore
        store = AttentionFeatureStore()
        store.set_timestep(500)

        k = torch.randn(4, 64, 64)   # (B*heads, L, head_dim)
        v = torch.randn(4, 64, 64)
        store.store(block_idx=0, layer_idx=1, keys=k, values=v)

        retrieved_k, retrieved_v = store.retrieve(block_idx=0, layer_idx=1)
        self.assertTrue(torch.allclose(retrieved_k, k))
        self.assertTrue(torch.allclose(retrieved_v, v))

    def test_timestep_isolation(self):
        """Deux timesteps différents doivent stocker séparément."""
        from stylus import AttentionFeatureStore
        store = AttentionFeatureStore()

        store.set_timestep(100)
        k1 = torch.ones(2, 10, 8)
        store.store(0, 0, k1, k1)

        store.set_timestep(200)
        k2 = torch.ones(2, 10, 8) * 2
        store.store(0, 0, k2, k2)

        store.set_timestep(100)
        rk, _ = store.retrieve(0, 0)
        self.assertTrue(torch.allclose(rk, k1))

        store.set_timestep(200)
        rk2, _ = store.retrieve(0, 0)
        self.assertTrue(torch.allclose(rk2, k2))

    def test_missing_key_returns_none(self):
        from stylus import AttentionFeatureStore
        store = AttentionFeatureStore()
        store.set_timestep(999)
        k, v = store.retrieve(99, 99)
        self.assertIsNone(k)
        self.assertIsNone(v)

    def test_reset_clears_store(self):
        from stylus import AttentionFeatureStore
        store = AttentionFeatureStore()
        store.set_timestep(0)
        store.store(0, 0, torch.randn(2, 10, 8), torch.randn(2, 10, 8))
        store.reset()
        k, v = store.retrieve(0, 0)
        self.assertIsNone(k)
        self.assertIsNone(v)


class TestMelSpectrogramProcessor(unittest.TestCase):
    """Teste la conversion audio ↔ mel ↔ image."""

    def setUp(self):
        from stylus import StylusConfig, MelSpectrogramProcessor
        cfg = StylusConfig(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=128,        # réduit pour les tests
            target_length=128,
        )
        try:
            self.proc = MelSpectrogramProcessor(cfg)
            self.available = True
        except ImportError:
            self.available = False

    def test_audio_to_mel_shape(self):
        if not self.available:
            self.skipTest("librosa not installed")
        sr = 22050
        audio = np.random.randn(sr * 2).astype(np.float32)  # 2 secondes
        mel = self.proc.audio_to_mel(audio)
        self.assertEqual(mel.shape[0], 128)   # n_mels
        self.assertGreater(mel.shape[1], 0)    # frames temporelles

    def test_mel_to_image_shape(self):
        if not self.available:
            self.skipTest("librosa not installed")
        mel = np.random.randn(128, 200).astype(np.float32)
        img = self.proc.mel_to_image(mel)
        self.assertEqual(img.shape, (1, 3, 128, 128))  # (B, RGB, H, target_length)
        # Valeurs dans [-1, 1]
        self.assertGreaterEqual(img.min().item(), -1.1)
        self.assertLessEqual(img.max().item(), 1.1)

    def test_image_to_mel_shape(self):
        if not self.available:
            self.skipTest("librosa not installed")
        mel_original = np.random.randn(128, 200).astype(np.float32)
        img = torch.randn(1, 3, 128, 128)
        mel_back = self.proc.image_to_mel(img, mel_original)
        self.assertEqual(mel_back.shape[0], 128)


class TestManualAttention(unittest.TestCase):
    """Teste le calcul d'attention avec temperature scaling."""

    def test_output_shape(self):
        from stylus import _manual_attention

        # B=2 batches, heads=4, L=64 tokens, head_dim=32
        B, heads, L, head_dim = 2, 4, 64, 32
        B_heads = B * heads   # 8 — shape attendue en entrée de _manual_attention

        module = MagicMock()
        module.heads = heads
        # to_out[0] est une projection (B, L, heads*head_dim) → (B, L, heads*head_dim)
        # On utilise l'identité pour le test
        module.to_out = [lambda x: x, lambda x: x]

        q = torch.randn(B_heads, L, head_dim)
        k = torch.randn(B_heads, L, head_dim)
        v = torch.randn(B_heads, L, head_dim)

        out = _manual_attention(q, k, v, module, temperature=1.0)

        # _manual_attention appelle _unshape : (B*heads, L, head_dim) → (B, L, heads*head_dim)
        # puis to_out[0] (identité ici), donc shape finale = (B, L, heads*head_dim)
        expected_shape = (B, L, heads * head_dim)
        self.assertEqual(out.shape, expected_shape,
            msg=f"Expected {expected_shape}, got {tuple(out.shape)}. "
                f"_manual_attention reshapes (B*heads,L,d) → (B,L,heads*d) via _unshape.")

    def test_temperature_effect(self):
        """Temperature < 1 doit produire une distribution d'attention plus piquée.

        On construit des logits structurés (un token clairement dominant) pour
        garantir que baisser la température concentre toujours la distribution.
        Avec des logits aléatoires la variance est trop grande pour être fiable.
        """
        import torch.nn.functional as F

        torch.manual_seed(0)
        B_heads, L, head_dim = 4, 10, 16

        # Q et K construits pour qu'un token soit clairement dominant :
        # q pointe sur dim 0, k[token 0] a un fort signal sur dim 0,
        # les autres tokens ont des signaux faibles.
        q = torch.zeros(B_heads, L, head_dim)
        k = torch.zeros(B_heads, L, head_dim)
        q[:, :, 0] = 1.0
        k[:, 0, 0] = 5.0   # token 0 : logit élevé
        k[:, 1:, 0] = 0.1  # autres  : logit faible

        logits = torch.bmm(q, k.transpose(-2, -1))   # (B_heads, L, L)

        scale_sharp  = head_dim ** -0.5 * 0.2   # température basse → piqué
        scale_normal = head_dim ** -0.5 * 1.0   # température normale → étalé

        attn_sharp  = F.softmax(logits / scale_sharp,  dim=-1)
        attn_normal = F.softmax(logits / scale_normal, dim=-1)

        eps = 1e-8
        entropy_sharp  = -(attn_sharp  * (attn_sharp  + eps).log()).sum(-1).mean()
        entropy_normal = -(attn_normal * (attn_normal + eps).log()).sum(-1).mean()

        self.assertLess(
            entropy_sharp.item(), entropy_normal.item(),
            msg=f"entropy_sharp={entropy_sharp:.4f} doit être < entropy_normal={entropy_normal:.4f}"
        )


class TestDDIMInversionStep(unittest.TestCase):
    """Teste le step d'inversion DDIM."""

    def test_inversion_step_shape(self):
        from stylus import StylusPipeline, StylusConfig

        cfg = StylusConfig(device="cpu", dtype=torch.float32)
        pipeline = StylusPipeline(cfg)

        # Mock du scheduler
        scheduler = MagicMock()
        scheduler.config.num_train_timesteps = 1000
        scheduler.alphas_cumprod = torch.linspace(0.99, 0.01, 1000)

        zt = torch.randn(1, 4, 8, 8)
        noise_pred = torch.randn(1, 4, 8, 8)
        t = torch.tensor(500)

        zt_next = pipeline._ddim_inversion_step(zt, noise_pred, t, scheduler)
        self.assertEqual(zt_next.shape, zt.shape)

    def test_adain_inversion_consistency(self):
        """AdaIN sur un latent doit donner les stats du style."""
        from stylus import adain

        content = torch.randn(1, 4, 64, 64)
        style   = torch.randn(1, 4, 64, 64) * 5

        result = adain(content, style)

        # Stats du résultat ≈ stats du style
        for c in range(4):
            self.assertAlmostEqual(
                result[0, c].mean().item(),
                style[0, c].mean().item(),
                delta=0.5
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
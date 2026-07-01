import os
import random
import uuid
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from PIL import Image
from deepface import DeepFace

ATTACKER_MODELS = {
    'Facenet512': (160, 160),
    'ArcFace': (112, 112),
    'GhostFaceNet': (112, 112),
    'VGG-Face': (224, 224),
}

VICTIM_MODELS = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face', 'IR152']

ALL_ATTACKS = [
    'PGD',
    'MI_FGSM',
    'TI_FGSM',
    'SI_NI_FGSM',
    'MI_ADMIX_DI_TI',
    'BPA_CNN',
    'BSR',
    'DECOWA',
    'SIA_MI_TI',
    'OPS',
    'ATT_CNN',
    'LI_BOOST_MI',
    'GRA',
    'IDAA',
    'DYNAMIC_MORPH',
    'DPA_HMA',
    'DYNAMIC_MORPH',
]

ATTACK_COLS = {
    'PGD': 'pgd_path',
    'MI_FGSM': 'mi_fgsm_path',
    'TI_FGSM': 'ti_fgsm_path',
    'SI_NI_FGSM': 'si_ni_fgsm_path',
    'MI_ADMIX_DI_TI': 'mi_admix_di_ti_path',
    'BPA_CNN': 'bpa_cnn_path',
    'BSR': 'bsr_path',
    'DECOWA': 'decowa_path',
    'SIA_MI_TI': 'sia_mi_ti_path',
    'OPS': 'ops_path',
    'ATT_CNN': 'att_cnn_path',
    'LI_BOOST_MI': 'li_boost_mi_path',
    'GRA': 'gra_path',
    'IDAA': 'idaa_path',
    'DYNAMIC_MORPH': 'dynamic_morph_path',
    'DPA_HMA': 'dpa_hma_path',
    'DYNAMIC_MORPH': 'dynamic_morph_path',
}

EPSILON = 0.062
NUM_ITER = 5
DECAY = 1.0
DECOWA_MESH = 3
DECOWA_NUM_WARPING = 20
DECOWA_NOISE_SCALE = 2.0
DECOWA_RHO = 0.01
OPS_BETA = 2.0
OPS_NUM_NEIGHBOR = 10
OPS_NUM_OPERATOR = 10
OPS_SAMPLE_LEVELS = (2, 3, 4)
LIBOOST_K = 6
LIBOOST_N = 30
GRA_NUM_NEIGHBOR = 20
GRA_BETA = 3.5
GRA_SIGN_DECAY = 0.94
DPA_HMA_SEED = int(os.environ.get('TRANSFER_ATTACK_DPA_HMA_SEED', '1'))
DPA_HMA_NUM_ITER = int(os.environ.get('TRANSFER_ATTACK_DPA_HMA_NUM_ITER', str(NUM_ITER)))
DPA_HMA_CANONICAL_SIZE = (224, 224)
_dpa_hma_seeded = False


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _ensure_dpa_hma_seed() -> None:
    global _dpa_hma_seeded
    if not _dpa_hma_seeded:
        set_global_seed(DPA_HMA_SEED)
        _dpa_hma_seeded = True



def configure_cpu_runtime(tf_threads: int = 1) -> None:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    except Exception:
        pass


def resolve_image_path(path: str, dataset_root: str) -> str:
    value = str(path)
    if os.path.exists(value):
        return value
    marker = 'dataset_extractedfaces/'
    if marker in value:
        rel = value.split(marker, 1)[1]
        candidate = os.path.join(dataset_root, rel)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(dataset_root, value.lstrip('/'))


def load_and_preprocess(path: str, input_size):
    img = Image.open(path).convert('RGB').resize(input_size)
    arr = np.array(img).astype('float32') / 255.0
    return (arr - 0.5) * 2.0


def denormalize(x: np.ndarray) -> np.ndarray:
    x = (x + 1.0) / 2.0
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def compute_embedding(model, x):
    out = model(x, training=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return tf.nn.l2_normalize(out, axis=1)


def attack_loss(cos, attack_type: str):
    return tf.reduce_mean(cos if str(attack_type).strip().lower() == 'impersonation_attack' else (1 - cos))


def save_adv(img_uint8: np.ndarray, attack_name: str, src: str, tgt: str, attack_type: str, model_name: str, row_id: int, adv_root: str) -> str:
    out_dir = Path(adv_root) / model_name / attack_name
    out_dir.mkdir(parents=True, exist_ok=True)
    s = Path(src).stem.replace(' ', '_')
    t = Path(tgt).stem.replace(' ', '_')
    rand = uuid.uuid4().hex[:8]
    name = f'adv_r{row_id}_{s}_to_{t}_{attack_type}_{rand}.png'
    path = out_dir / name
    Image.fromarray(img_uint8).save(path)
    return str(path.resolve())


def gaussian_kernel(k=15, sigma=3.0, ch=3):
    x = tf.range(-k // 2 + 1, k // 2 + 1, dtype=tf.float32)
    g = tf.exp(-tf.square(x) / (2 * sigma**2))
    g /= tf.reduce_sum(g)
    kernel = tf.tensordot(g, g, axes=0)
    kernel = kernel[:, :, None, None]
    return tf.tile(kernel, [1, 1, ch, 1])


def input_diversity(x, input_size, prob=0.7):
    if tf.random.uniform([]) > prob:
        return x
    img_size = input_size[0]
    rnd = tf.random.uniform([], int(0.9 * img_size), img_size, dtype=tf.int32)
    x_resized = tf.image.resize(x, (rnd, rnd))
    pad_total = img_size - rnd
    pad_top = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_bottom = pad_total - pad_top
    pad_left = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_right = pad_total - pad_left
    x_padded = tf.pad(x_resized, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return tf.image.resize(x_padded, input_size)


def _idaa_transform(x, input_size):
    out = input_diversity(x, input_size)

    if tf.random.uniform([]) < 0.5:
        out = tf.image.flip_left_right(out)

    scale = tf.random.uniform([], 0.9, 1.1)

    h = tf.shape(out)[1]
    w = tf.shape(out)[2]

    nh = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    nw = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)

    out = tf.image.resize(out, (nh, nw))
    out = tf.image.resize(out, (h, w))

    return out


def _idaa_local_mix(batch, alpha=0.4):
    shuffled = tf.random.shuffle(batch)
    lam = tf.random.uniform([], 0.3, 0.7)

    h = tf.shape(batch)[1]
    w = tf.shape(batch)[2]

    cut_h = h // 2
    cut_w = w // 2

    mixed = tf.identity(batch)

    patch = shuffled[:, :cut_h, :cut_w, :]

    upper = tf.concat(
        [patch, batch[:, :cut_h, cut_w:, :]],
        axis=2
    )

    lower = batch[:, cut_h:, :, :]

    mixed = tf.concat([upper, lower], axis=1)

    return lam * mixed + (1.0 - lam) * batch

def pgd_attack(model, x, tgt_emb, attack_type, random_start=True):
    if random_start:
        noise = tf.random.uniform(tf.shape(x), minval=-EPSILON, maxval=EPSILON, dtype=x.dtype)
        adv = tf.clip_by_value(x + noise, -1.0, 1.0)
    else:
        adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


# Student-contributed attack integration:
# ATT_CNN by Keshav Raj (IIIT Delhi)
# ATT-CNN is a CNN-side adaptation inspired by the NeurIPS 2024 ATT paper.
# It is not an official reproduction of the ViT token-hook ATT algorithm.
def att_cnn_attack(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)

        grad = tape.gradient(loss, adv)
        grad_var = tf.math.reduce_variance(grad)
        grad = grad / (tf.sqrt(grad_var) + 1e-8)

        mean_grad = tf.reduce_mean(tf.abs(grad))
        strong_mask = tf.abs(grad) > mean_grad
        grad = tf.where(strong_mask, grad * 0.7, grad)

        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)

    return adv


# Student-contributed attack integration:
# LI_BOOST_MI by Charushi (IGDTUW)
# This is the MI-style logarithmic-shift boosting variant verified on the
# shared face-verification subset. The BSR hybrid variant from the student
# branch is intentionally not included here.
def sample_logarithmic_shifts(k, n_samples):
    shifts = np.arange(-k, k + 1)
    weights = np.log(k + 2) - np.log(np.abs(shifts) + 1)
    probs = weights / np.sum(weights)
    return np.random.choice(shifts, size=(n_samples, 2), p=probs)


@tf.function
def compiled_liboost_mi_step(model, adv, tgt_emb, shifts, n_samples, chunk_size, is_impersonation):
    grad_sum = tf.zeros_like(adv)
    for i in tf.range(0, n_samples, chunk_size):
        chunk_shifts = shifts[i:i + chunk_size]
        current_chunk_size = tf.shape(chunk_shifts)[0]
        with tf.GradientTape() as tape:
            tape.watch(adv)
            translated_list = tf.TensorArray(dtype=tf.float32, size=current_chunk_size)
            for n in tf.range(current_chunk_size):
                dy = chunk_shifts[n, 0]
                dx = chunk_shifts[n, 1]
                rolled = tf.roll(adv, shift=[dy, dx], axis=[1, 2])
                translated_list = translated_list.write(n, rolled[0])
            batch_adv = translated_list.stack()

            out = model(batch_adv, training=False)
            if isinstance(out, (tuple, list)):
                out = out[0]
            emb = tf.nn.l2_normalize(out, axis=1)
            tgt_rep = tf.repeat(tgt_emb, current_chunk_size, axis=0)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss_val = tf.reduce_mean(cos if is_impersonation else (1.0 - cos))
            loss = loss_val * (tf.cast(current_chunk_size, tf.float32) / tf.cast(n_samples, tf.float32))
        grad_sum += tape.gradient(loss, adv)
    return grad_sum


def li_boost_mi(model, x, tgt_emb, attack_type, k=LIBOOST_K, n_samples=LIBOOST_N, chunk_size=10):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    is_impersonation = str(attack_type).strip().lower() == 'impersonation_attack'

    for _ in range(NUM_ITER):
        shifts = tf.constant(sample_logarithmic_shifts(k, n_samples), dtype=tf.int32)
        grad = compiled_liboost_mi_step(
            model, adv, tgt_emb, shifts, n_samples, chunk_size, is_impersonation
        )
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)

    return adv


def ti_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    kernel = gaussian_kernel()
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def si_ni_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
    for _ in range(NUM_ITER):
        nes = adv + DECAY * alpha * g
        grad_sum = tf.zeros_like(x)
        for s in scales:
            with tf.GradientTape() as tape:
                tape.watch(nes)
                emb = compute_embedding(model, nes * s)
                cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                loss = attack_loss(cos, attack_type)
            grad_sum += tape.gradient(loss, nes)
        grad = grad_sum / len(scales)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_admix_di_ti(model, x, tgt_emb, attack_type, pool_imgs, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel()
    n_pool = tf.shape(pool_imgs)[0]
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            idx = tf.random.uniform([3], 0, n_pool, dtype=tf.int32)
            others = tf.gather(pool_imgs, idx)
            adv_rep = tf.repeat(adv, 3, axis=0)
            mixed = adv_rep + 0.2 * (others - adv_rep)
            batch = input_diversity(mixed, input_size)
            emb = compute_embedding(model, batch)
            tgt_rep = tf.repeat(tgt_emb, 3, axis=0)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv

# Student-contributed attack integration:
# BPA_CNN by Om Singh Rawat (IIT Delhi)
# Paper basis: Rethinking the Backward Propagation for Adversarial Transferability
# (NeurIPS 2023)
def bpa_cnn(model, x, tgt_emb, attack_type):
    """BPA-CNN: Backward Propagation Attack adapted for CNN face models.

    BPA (NeurIPS 2023) improves adversarial transferability by replacing
    sharp backward operations (ReLU, MaxPool) with smooth alternatives
    (SiLU derivative, softmax-weighted pooling).

    Since we cannot modify internal layers of pre-trained DeepFace models,
    we apply BPA's two key gradient-smoothing principles at the input level:
      1. SiLU-derivative scaling  – counteracts ReLU binary gradient masking
      2. Gaussian spatial smoothing – counteracts MaxPool gradient concentration
    """
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel(k=5, sigma=1.0)
    temperature = 3.0

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)

        grad = tape.gradient(loss, adv)

        # BPA Step 1: SiLU-inspired gradient smoothing
        # SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        # Applied to the gradient to counteract ReLU's binary masking
        scaled = temperature * grad
        sig = tf.sigmoid(scaled)
        silu_deriv = sig + scaled * sig * (1.0 - sig)
        grad = grad * silu_deriv

        # BPA Step 2: Gaussian spatial smoothing
        # Counteracts MaxPool's winner-take-all gradient concentration
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')

        # Momentum accumulation (inherited from MI-FGSM base)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)

    return adv


_BSR_MIN_DIM = 4


def _bsr_get_lengths(total: int, num_block: int):
    rand = np.random.uniform(size=num_block).astype(np.float32)
    sizes = np.round(rand * total / rand.sum()).astype(np.int32)
    sizes = np.maximum(sizes, _BSR_MIN_DIM)
    while sizes.sum() > total:
        sizes[sizes.argmax()] -= 1
    while sizes.sum() < total:
        sizes[sizes.argmin()] += 1
    return sizes.tolist()


def _bsr_rotate(image, angle_rad):
    h_int = int(image.shape[1])
    w_int = int(image.shape[2])
    if h_int < _BSR_MIN_DIM or w_int < _BSR_MIN_DIM:
        return image
    cx, cy = float(w_int) / 2.0, float(h_int) / 2.0
    cos_a = tf.math.cos(-angle_rad)
    sin_a = tf.math.sin(-angle_rad)
    tx = cx - cx * cos_a + cy * sin_a
    ty = cy - cx * sin_a - cy * cos_a
    transform = tf.reshape(
        tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0]), [1, 8]
    )
    out_shape = tf.cast(tf.stack([h_int, w_int]), dtype=tf.int32)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.cast(image, tf.float32),
        transforms=tf.cast(transform, tf.float32),
        output_shape=out_shape,
        interpolation='BILINEAR',
        fill_mode='REFLECT',
        fill_value=0.0,
    )


def _bsr_shuffle_rotate(x, num_block: int = 2):
    h_val, w_val = int(x.shape[1]), int(x.shape[2])
    w_strips = tf.split(x, _bsr_get_lengths(w_val, num_block), axis=2)
    result_w = []
    for wi in np.random.permutation(num_block).tolist():
        h_strips = tf.split(w_strips[wi], _bsr_get_lengths(h_val, num_block), axis=1)
        result_h = []
        for hi in np.random.permutation(num_block).tolist():
            angle = tf.random.truncated_normal([], stddev=0.05)
            result_h.append(_bsr_rotate(h_strips[hi], angle))
        result_w.append(tf.concat(result_h, axis=1))
    return tf.concat(result_w, axis=2)


# Student-contributed attack integration:
# BSR by Chirag Sharma (IIIT Vadodara)
# Paper basis: Boosting Adversarial Transferability by Block Shuffle and Rotation
# (CVPR 2024)
def bsr(model, x, tgt_emb, attack_type, num_copies: int = 20, num_block: int = 2):
    adv = tf.Variable(tf.identity(x), trainable=True, dtype=tf.float32)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            copies = [_bsr_shuffle_rotate(adv, num_block) for _ in range(num_copies)]
            x_batch = tf.concat(copies, axis=0)
            tgt_rep = tf.repeat(tgt_emb, num_copies, axis=0)
            emb = compute_embedding(model, x_batch)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv.assign(adv + alpha * tf.sign(g))
        adv.assign(tf.clip_by_value(adv, x - EPSILON, x + EPSILON))
        adv.assign(tf.clip_by_value(adv, -1.0, 1.0))
    return tf.identity(adv)


def _decowa_grid_points_2d(width, height):
    a = tf.linspace(-1.0, 1.0, height)
    b = tf.linspace(-1.0, 1.0, width)
    xx, yy = tf.meshgrid(a, b, indexing='ij')
    pts = tf.stack([yy, xx], axis=-1)
    return tf.reshape(pts, [-1, 2])


def _decowa_noisy_grid(width, height, noise_map):
    grid = _decowa_grid_points_2d(width, height)
    mod = tf.pad(noise_map, [[1, 1], [1, 1], [0, 0]])
    return grid + tf.reshape(mod, [-1, 2])


def _decowa_K(x_val, y_val):
    eps = 1e-9
    d2 = tf.reduce_sum(tf.square(x_val[:, :, None, :] - y_val[:, None, :, :]), axis=-1)
    return d2 * tf.math.log(d2 + eps)


def _decowa_P(x_val):
    n_val = tf.shape(x_val)[0]
    k_val = tf.shape(x_val)[1]
    return tf.concat([tf.ones([n_val, k_val, 1]), x_val], axis=-1)


def _decowa_tps_coeffs(x_val, y_val):
    k_val = tf.shape(x_val)[1]
    n_val = tf.shape(x_val)[0]
    k_mat = _decowa_K(x_val, x_val)
    p_mat = _decowa_P(x_val)
    top = tf.concat([k_mat, p_mat], axis=-1)
    bottom = tf.concat([tf.transpose(p_mat, [0, 2, 1]), tf.zeros([n_val, 3, 3])], axis=-1)
    l_mat = tf.concat([top, bottom], axis=1)
    z_mat = tf.concat([y_val, tf.zeros([n_val, 3, 2])], axis=1)
    q_val = tf.linalg.solve(l_mat, z_mat)
    return q_val[:, :k_val], q_val[:, k_val:]


def _decowa_dense_grid(height, width):
    gx = tf.linspace(-1.0, 1.0, width)
    gy = tf.linspace(-1.0, 1.0, height)
    x0 = tf.tile(gx[None, None, :], [1, height, 1])
    y0 = tf.tile(gy[None, :, None], [1, 1, width])
    grid = tf.stack([x0, y0], axis=-1)
    return tf.reshape(grid, [1, height * width, 2])


def _decowa_tps_grid(x_val, y_val, height, width):
    w_coef, a_coef = _decowa_tps_coeffs(x_val, y_val)
    base = _decowa_dense_grid(height, width)
    u_mat = _decowa_K(base, x_val)
    p_mat = _decowa_P(base)
    grid = tf.matmul(p_mat, a_coef) + tf.matmul(u_mat, w_coef)
    return tf.reshape(grid, [1, height, width, 2])


def _decowa_grid_sample(img, grid):
    n_val = tf.shape(img)[0]
    height = tf.shape(img)[1]
    width = tf.shape(img)[2]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    x_val = grid[..., 0]
    y_val = grid[..., 1]
    ix = ((x_val + 1.0) * width_f - 1.0) / 2.0
    iy = ((y_val + 1.0) * height_f - 1.0) / 2.0
    ix0 = tf.floor(ix)
    iy0 = tf.floor(iy)
    wx1 = ix - ix0
    wy1 = iy - iy0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    def sample(ixc, iyc):
        in_x = tf.logical_and(ixc >= 0.0, ixc <= width_f - 1.0)
        in_y = tf.logical_and(iyc >= 0.0, iyc <= height_f - 1.0)
        mask = tf.cast(tf.logical_and(in_x, in_y), tf.float32)[..., None]
        xc = tf.clip_by_value(tf.cast(ixc, tf.int32), 0, width - 1)
        yc = tf.clip_by_value(tf.cast(iyc, tf.int32), 0, height - 1)
        bidx = tf.broadcast_to(tf.reshape(tf.range(n_val), [n_val, 1, 1]), tf.shape(xc))
        idx = tf.stack([bidx, yc, xc], axis=-1)
        return tf.gather_nd(img, idx) * mask

    v00 = sample(ix0, iy0)
    v01 = sample(ix0, iy0 + 1.0)
    v10 = sample(ix0 + 1.0, iy0)
    v11 = sample(ix0 + 1.0, iy0 + 1.0)
    return (
        v00 * (wx0 * wy0)[..., None]
        + v10 * (wx1 * wy0)[..., None]
        + v01 * (wx0 * wy1)[..., None]
        + v11 * (wx1 * wy1)[..., None]
    )


def _decowa_warp(adv, noise_map, height, width):
    x_val = _decowa_grid_points_2d(DECOWA_MESH, DECOWA_MESH)[None, ...]
    y_val = _decowa_noisy_grid(DECOWA_MESH, DECOWA_MESH, noise_map)[None, ...]
    grid = _decowa_tps_grid(x_val, y_val, height, width)
    grid = tf.tile(grid, [tf.shape(adv)[0], 1, 1, 1])
    return _decowa_grid_sample(adv, grid)


def _decowa_update_noise_map(model, adv, tgt_emb, attack_type, height, width):
    noise_map = (tf.random.uniform([DECOWA_MESH - 2, DECOWA_MESH - 2, 2]) - 0.5) * DECOWA_NOISE_SCALE
    with tf.GradientTape() as tape:
        tape.watch(noise_map)
        warped = _decowa_warp(adv, noise_map, height, width)
        emb = compute_embedding(model, warped)
        cos = tf.reduce_sum(emb * tgt_emb, axis=1)
        loss = attack_loss(cos, attack_type)
    grad = tape.gradient(loss, noise_map)
    if grad is None:
        return noise_map
    grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
    return noise_map - DECOWA_RHO * grad


# Student-contributed attack integration:
# DeCowA by Om Singh Rawat (IIT Delhi)
# Paper basis: Boosting Adversarial Transferability across Model Genus by
# Deformation-Constrained Warping (AAAI 2024)
def decowa(model, x, tgt_emb, attack_type, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    height, width = int(input_size[1]), int(input_size[0])
    for _ in range(NUM_ITER):
        grads = tf.zeros_like(x)
        for _ in range(DECOWA_NUM_WARPING):
            noise_map = _decowa_update_noise_map(model, tf.stop_gradient(adv), tgt_emb, attack_type, height, width)
            with tf.GradientTape() as tape:
                tape.watch(adv)
                warped = _decowa_warp(adv, noise_map, height, width)
                emb = compute_embedding(model, warped)
                cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                loss = attack_loss(cos, attack_type)
            grad = tape.gradient(loss, adv)
            grads += tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        grads = grads / DECOWA_NUM_WARPING
        grads = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)
        g = DECAY * g + grads
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def sia_vertical_shift(block):
    h_val = tf.shape(block)[1]
    shift = tf.random.uniform([], 0, tf.maximum(h_val, 1), dtype=tf.int32)
    return tf.roll(block, shift=shift, axis=1)


def sia_horizontal_shift(block):
    w_val = tf.shape(block)[2]
    shift = tf.random.uniform([], 0, tf.maximum(w_val, 1), dtype=tf.int32)
    return tf.roll(block, shift=shift, axis=2)


def sia_vertical_flip(block):
    return tf.reverse(block, axis=[1])


def sia_horizontal_flip(block):
    return tf.reverse(block, axis=[2])


def sia_rotate180(block):
    return tf.reverse(block, axis=[1, 2])


def sia_scale(block):
    factor = tf.random.uniform([], 0.5, 1.0)
    return factor * block


def sia_add_noise(block):
    noise = tf.random.uniform(tf.shape(block), -EPSILON, EPSILON)
    return block + noise


SIA_OPS = [
    sia_vertical_shift,
    sia_horizontal_shift,
    sia_vertical_flip,
    sia_horizontal_flip,
    sia_rotate180,
    sia_scale,
    sia_add_noise,
]


def sia_block_transform(x_val, num_block=3):
    h_val, w_val = x_val.shape[1], x_val.shape[2]

    def split_points(size, n_parts):
        if size <= n_parts:
            return [0, size]
        pts = sorted(np.random.choice(range(1, size), n_parts - 1, replace=False).tolist())
        return [0] + pts + [size]

    h_pts = split_points(h_val, num_block)
    w_pts = split_points(w_val, num_block)

    rows = []
    for i in range(len(h_pts) - 1):
        cols = []
        for j in range(len(w_pts) - 1):
            block = x_val[:, h_pts[i]:h_pts[i + 1], w_pts[j]:w_pts[j + 1], :]
            op = SIA_OPS[np.random.randint(len(SIA_OPS))]
            cols.append(op(block))
        rows.append(tf.concat(cols, axis=2))
    return tf.concat(rows, axis=1)


# Student-contributed attack integration:
# SIA_MI_TI by Janhavi Kishor
# Paper basis: Structure Invariant Transformation for better Adversarial Transferability
# (ICCV 2023) combined with MI-FGSM and TI-FGSM
def sia_mi_ti(model, x, tgt_emb, attack_type, num_copies=5, num_block=3):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel()

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            copies = [sia_block_transform(adv, num_block) for _ in range(num_copies)]
            batch = tf.concat(copies, axis=0)
            emb = compute_embedding(model, batch)
            tgt_rep = tf.repeat(tgt_emb, num_copies, axis=0)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def _ops_rotate(image, angle_rad):
    h_int = int(image.shape[1])
    w_int = int(image.shape[2])
    if h_int < 4 or w_int < 4:
        return image
    cx, cy = float(w_int) / 2.0, float(h_int) / 2.0
    cos_a = tf.math.cos(-angle_rad)
    sin_a = tf.math.sin(-angle_rad)
    tx = cx - cx * cos_a + cy * sin_a
    ty = cy - cx * sin_a - cy * cos_a
    transform = tf.reshape(
        tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0]), [1, 8]
    )
    out_shape = tf.cast(tf.stack([h_int, w_int]), dtype=tf.int32)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.cast(image, tf.float32),
        transforms=tf.cast(transform, tf.float32),
        output_shape=out_shape,
        interpolation='BILINEAR',
        fill_mode='REFLECT',
        fill_value=0.0,
    )


def _ops_dim(x, input_size, resize_rate):
    img_size = int(input_size[0])
    img_resize = int(round(img_size * resize_rate))
    rnd = tf.random.uniform(
        [],
        min(img_size, img_resize),
        max(img_size, img_resize) + 1,
        dtype=tf.int32,
    )
    rescaled = tf.image.resize(x, (rnd, rnd))
    pad_total = img_resize - rnd
    pad_top = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_bottom = pad_total - pad_top
    pad_left = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_right = pad_total - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return tf.image.resize(padded, input_size)


def _ops_apply_basic(x, op_id, input_size):
    height = int(input_size[1])
    width = int(input_size[0])
    if op_id == 0:
        return x
    if op_id == 1:
        return tf.reverse(x, axis=[1])
    if op_id == 2:
        return tf.reverse(x, axis=[2])
    if op_id == 3:
        return tf.roll(x, shift=np.random.randint(0, height), axis=1)
    if op_id == 4:
        return tf.roll(x, shift=np.random.randint(0, width), axis=2)
    if op_id == 5:
        return _ops_rotate(x, np.deg2rad(5.0))
    if op_id == 6:
        return _ops_rotate(x, np.deg2rad(-5.0))
    if op_id == 7:
        return _ops_rotate(x, np.deg2rad(15.0))
    if op_id == 8:
        return _ops_rotate(x, np.deg2rad(-15.0))
    if op_id == 9:
        return _ops_rotate(x, np.deg2rad(45.0))
    if op_id == 10:
        return _ops_rotate(x, np.deg2rad(-45.0))
    if op_id == 11:
        return tf.image.rot90(x, k=1)
    if op_id == 12:
        return tf.image.rot90(x, k=3)
    if op_id == 13:
        return tf.image.rot90(x, k=2)
    if 14 <= op_id <= 20:
        return x / float(op_id - 12)
    return _ops_dim(x, input_size, 1.1 + 0.2 * float(op_id - 21))


def _ops_init_operator_ids():
    op_ids = []
    for level in OPS_SAMPLE_LEVELS:
        for _ in range(31):
            op_ids.append(tuple(random.randrange(31) for _ in range(level)))
    return op_ids


def _ops_init_eps_list(x):
    sample_ratios = np.arange(0.25, 1.76, 0.25, dtype=np.float32)
    eps_list = []
    for ratio in sample_ratios:
        radius = float(OPS_BETA * EPSILON * ratio)
        for _ in range(OPS_NUM_NEIGHBOR):
            eps_list.append(tf.random.uniform(tf.shape(x), -radius, radius, dtype=x.dtype))
    return eps_list


def _ops_apply_operator(x, input_size, op_ids):
    out = x
    for op_id in op_ids:
        out = _ops_apply_basic(out, op_id, input_size)
    return out


# Student-contributed attack integration:
# OPS by Kkartik Aggarwal (Delhi Technological University)
# Paper basis: Boosting Adversarial Transferability through Augmentation in
# Hypothesis Space (CVPR 2025)
def ops_attack(model, x, tgt_emb, attack_type, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    eps_list = _ops_init_eps_list(x)

    for _ in range(NUM_ITER):
        selected_eps = random.sample(eps_list, min(OPS_NUM_NEIGHBOR, len(eps_list)))
        op_groups = []
        for _ in selected_eps:
            op_list = _ops_init_operator_ids()
            op_groups.append(random.sample(op_list, min(OPS_NUM_OPERATOR, len(op_list))))

        with tf.GradientTape() as tape:
            tape.watch(adv)
            samples = [adv]
            for noise, selected_ops in zip(selected_eps, op_groups):
                for op_ids in selected_ops:
                    samples.append(_ops_apply_operator(adv + noise, input_size, op_ids))
            batch = tf.concat(samples, axis=0)
            tgt_rep = tf.repeat(tgt_emb, tf.shape(batch)[0], axis=0)
            emb = compute_embedding(model, batch)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        if grad is None:
            grad = tf.zeros_like(adv)
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def _dpa_hma_affine(image, std_proj, std_rotate):
    h_int = int(image.shape[1])
    w_int = int(image.shape[2])
    if h_int <= 1 or w_int <= 1:
        return image

    cx, cy = float(w_int) / 2.0, float(h_int) / 2.0
    angle = tf.random.truncated_normal([], stddev=std_rotate)
    scale = 1.0 + tf.random.truncated_normal([], stddev=std_proj)
    tx_noise = tf.random.truncated_normal([], stddev=std_proj) * float(w_int)
    ty_noise = tf.random.truncated_normal([], stddev=std_proj) * float(h_int)

    cos_a = tf.math.cos(-angle) * scale
    sin_a = tf.math.sin(-angle) * scale
    tx = cx - cx * cos_a + cy * sin_a + tx_noise
    ty = cy - cx * sin_a - cy * cos_a + ty_noise

    transform = tf.reshape(
        tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0]), [1, 8]
    )
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.cast(image, tf.float32),
        transforms=tf.cast(transform, tf.float32),
        output_shape=tf.cast(tf.stack([h_int, w_int]), dtype=tf.int32),
        interpolation='BILINEAR',
        fill_mode='REFLECT',
        fill_value=0.0,
    )


def _dpa_hma_transform_batch(adv, hard_grad, num_copies: int):
    copies = []
    hard_direction = tf.sign(hard_grad)
    hard_steps = (0.0, 0.004, -0.004, 0.008, -0.008)
    for i in range(num_copies):
        step = hard_steps[i % len(hard_steps)]
        hard_view = tf.clip_by_value(adv + step * hard_direction, -1.0, 1.0)
        std_proj = tf.random.uniform([], 0.01, 0.08)
        std_rotate = tf.random.uniform([], 0.01, 0.08)
        copies.append(_dpa_hma_affine(hard_view, std_proj, std_rotate))
    return tf.concat(copies, axis=0)


def _dpa_hma_optimize(model, x, tgt_emb, attack_type, num_copies: int, num_iter: int):
    if num_iter <= 0:
        raise ValueError('DPA_HMA num_iter must be positive')
    adv = tf.Variable(tf.identity(x), trainable=True, dtype=tf.float32)
    momentum = tf.zeros_like(x)
    alpha = EPSILON / num_iter

    for _ in range(num_iter):
        with tf.GradientTape() as hard_tape:
            hard_tape.watch(adv)
            hard_emb = compute_embedding(model, adv)
            hard_cos = tf.reduce_sum(hard_emb * tgt_emb, axis=1)
            hard_loss = attack_loss(hard_cos, attack_type)
        hard_grad = hard_tape.gradient(hard_loss, adv)

        with tf.GradientTape() as tape:
            batch = _dpa_hma_transform_batch(adv, hard_grad, num_copies)
            tgt_rep = tf.repeat(tgt_emb, num_copies, axis=0)
            emb = compute_embedding(model, batch)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        momentum = DECAY * momentum + grad
        adv.assign(adv + alpha * tf.sign(momentum))
        adv.assign(tf.clip_by_value(adv, x - EPSILON, x + EPSILON))
        adv.assign(tf.clip_by_value(adv, -1.0, 1.0))

    return tf.identity(adv)


# Student-contributed attack integration:
# DPA_HMA by Kushal Khemka (DTU Delhi)
# Paper basis: Diverse Parameters Augmentation (DPA) from Improving the
# Transferability of Adversarial Attacks on Face Recognition (CVPR 2025)

# GRA by Krish Bansal (Delhi Technological University)
# Based on "Boosting Adversarial Transferability via Gradient Relevance Attack"
# (ICCV 2023) by Yingwen Wu, Yinpeng Dong, Qin Wang, Jun Zhu, Xiaolin Hu.
def gra_attack(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    m = tf.ones_like(x) * (10.0 / 9.4)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    neighbor_bound = EPSILON * GRA_BETA

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        current_grad = tape.gradient(loss, adv)

        neighbor_grad_sum = tf.zeros_like(x)
        for _ in range(GRA_NUM_NEIGHBOR):
            noise = tf.random.uniform(
                tf.shape(adv), minval=-neighbor_bound, maxval=neighbor_bound
            )
            x_neighbor = adv + noise
            with tf.GradientTape() as tape_n:
                tape_n.watch(x_neighbor)
                emb_n = compute_embedding(model, x_neighbor)
                cos_n = tf.reduce_sum(emb_n * tgt_emb, axis=1)
                loss_n = attack_loss(cos_n, attack_type)
            neighbor_grad_sum += tape_n.gradient(loss_n, x_neighbor)
        avg_neighbor_grad = neighbor_grad_sum / float(GRA_NUM_NEIGHBOR)

        flat_cur = tf.reshape(current_grad, [1, -1])
        flat_sam = tf.reshape(avg_neighbor_grad, [1, -1])
        cos_sim_val = (
            tf.reduce_sum(flat_cur * flat_sam, axis=1)
            / (tf.sqrt(tf.reduce_sum(flat_cur ** 2, axis=1))
               * tf.sqrt(tf.reduce_sum(flat_sam ** 2, axis=1)) + 1e-12)
        )
        cos_sim_val = tf.reshape(cos_sim_val, [1, 1, 1, 1])
        corrected_grad = (
            cos_sim_val * current_grad
            + (1.0 - cos_sim_val) * avg_neighbor_grad
        )

        prev_g = g
        corrected_grad = corrected_grad / (tf.reduce_mean(tf.abs(corrected_grad)) + 1e-8)
        g = DECAY * g + corrected_grad

        sign_match = tf.cast(tf.equal(tf.sign(prev_g), tf.sign(g)), dtype=tf.float32)
        sign_flip = tf.ones_like(x) - sign_match
        m = m * (sign_match + sign_flip * GRA_SIGN_DECAY)

        adv = adv + alpha * m * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)

    return adv


def dpa_hma(model, x, tgt_emb, attack_type, num_copies: int = 8, num_iter: int = DPA_HMA_NUM_ITER):
    _ensure_dpa_hma_seed()
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    return _dpa_hma_optimize(model, x, tgt_emb, attack_type, num_copies, num_iter)




def dynamic_morph_mi_fgsm(model, src, tgt, attack_type, input_size):
    """Assignment 4: D-FMA (Pre-aligned Semantic Mixing)"""
    h, w = input_size
    mask = np.zeros((h, w, 3), dtype=np.float32)
    
    mask[int(h*0.35):int(h*0.52), int(w*0.20):int(w*0.45), :] = 1.0  
    mask[int(h*0.35):int(h*0.52), int(w*0.55):int(w*0.80), :] = 1.0  
    mask[int(h*0.48):int(h*0.70), int(w*0.38):int(w*0.62), :] = 1.0  
    
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    tf_mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    
    morphed_tensor = (tgt * tf_mask) + (src * (1.0 - tf_mask))
    
    adv = tf.identity(morphed_tensor)
    g = tf.zeros_like(adv)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(compute_embedding(model, tgt), axis=1)
    
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
            
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, morphed_tensor - EPSILON, morphed_tensor + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
        
    # --- EVALUATION HOOK FOR DYNAMIC MORPH ONLY ---
    try:
        for victim_name in ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face']:
            v_size = ATTACKER_MODELS[victim_name]
            v_model = build_attacker(victim_name)
            v_adv = tf.image.resize(adv, v_size)
            v_tgt = tf.image.resize(tgt, v_size)
            emb_adv = compute_embedding(v_model, v_adv)
            emb_tgt = compute_embedding(v_model, v_tgt)
            sim = float(tf.reduce_sum(emb_adv * emb_tgt, axis=1).numpy()[0])
            _EVAL_RECORDS.append({
                'victim': victim_name,
                'attack_type': attack_type,
                'similarity': sim
            })
    except Exception:
        pass
        
    return adv

def build_attacker(model_name: str):
    return DeepFace.build_model(model_name).model


def run_attack(attack_name: str, model, src, tgt, attack_type: str, input_size):
    tgt_emb = compute_embedding(model, tgt)
    if attack_name == 'PGD':
        return pgd_attack(model, src, tgt_emb, attack_type)
    if attack_name == 'MI_FGSM':
        return mi_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'TI_FGSM':
        return ti_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'SI_NI_FGSM':
        return si_ni_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'MI_ADMIX_DI_TI':
        pool_imgs = tf.concat([src, tgt, src], axis=0)
        return mi_admix_di_ti(model, src, tgt_emb, attack_type, pool_imgs, input_size)
    if attack_name == 'BPA_CNN':
        return bpa_cnn(model, src, tgt_emb, attack_type)
    if attack_name == 'BSR':
        return bsr(model, src, tgt_emb, attack_type)
    if attack_name == 'DECOWA':
        return decowa(model, src, tgt_emb, attack_type, input_size)
    if attack_name == 'SIA_MI_TI':
        return sia_mi_ti(model, src, tgt_emb, attack_type)
    if attack_name == 'OPS':
        return ops_attack(model, src, tgt_emb, attack_type, input_size)
    if attack_name == 'ATT_CNN':
        return att_cnn_attack(model, src, tgt_emb, attack_type)
    if attack_name == 'LI_BOOST_MI':
        return li_boost_mi(model, src, tgt_emb, attack_type)
    if attack_name == 'GRA':
        return gra_attack(model, src, tgt_emb, attack_type)
    if attack_name == 'IDAA':
        return idaa(model, src, tgt_emb, attack_type, input_size)
    if attack_name == 'DPA_HMA':
        return dpa_hma(model, src, tgt_emb, attack_type)
    if attack_name == 'DYNAMIC_MORPH':
        return dynamic_morph_mi_fgsm(model, src, tgt, attack_type, input_size)
    raise ValueError(f'Unsupported attack: {attack_name}')

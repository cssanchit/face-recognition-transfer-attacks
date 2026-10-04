from RobFR.attack.base import ConstrainedMethod
import torch
from RobFR.attack.face_landmark import getlist_landmark
import json

__all__ = ['LGC']
class LGC(ConstrainedMethod):
    def __init__(self, model, goal, distance_metric, eps, dataset='lfw', 
        iters=20, mu=1.0, num_samples=4, sigma=1, **kwargs):
        super(LGC, self).__init__(model, goal, distance_metric, eps, **kwargs)
        self.iters = iters
        self.mu = mu
        self.num_samples = num_samples
        self.sigma = sigma
        with open('./data/{}_aligned_landmarks.json'.format(dataset), 'r') as f:
            self.landmark_values = json.load(f)
    def batch_attack(self, xs, ys_feat, pairs, **kwargs):
        target_feat, source_feat = self.prepare_reference_features(xs, kwargs['ys'], ys_feat)
        xs_adv = xs.clone().detach().requires_grad_(True)
        names = []
        for pair in pairs:
            src_path = pair[0]
            tokens = src_path.split('/')
            name = '_'.join(tokens[3:])
            names.append(name)
        g = torch.zeros_like(xs_adv)
        for _ in range(self.iters):
            img_shape = xs_adv.shape[2:]
            mask = getlist_landmark(names, self.landmark_values,
                self.num_samples, img_shape, sigma=self.sigma)
            mask = torch.tensor(
                mask.transpose((0, 3, 1, 2)),
                device=xs_adv.device,
                dtype=xs_adv.dtype,
            )
            features = self.encode(xs_adv * mask)
            loss = self.getLoss(features, target_feat, source_feat)
            loss.backward()
            grad = xs_adv.grad
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            g = g * self.mu + grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, g, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return xs_adv

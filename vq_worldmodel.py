from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Distribution, Independent


class ExponentialMovingAverages(nn.Module):
    """
    Implementation adopted from [1].

    Uses bias correction similar to the Adam optimizer, if bias_correction is set to True, otherwise it uses the EMA
    implementation from the original VQ-VAE paper.

    References:
        [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/moving_averages.py
    """

    def __init__(
            self,
            shape_or_initial_value: Union[Tuple[int, ...], Tensor],
            decay: float,
            bias_correction: bool) -> None:
        super().__init__()
        self.register_buffer('_decay', torch.tensor(decay, dtype=torch.float64))
        self._bias_correction = bias_correction

        if isinstance(shape_or_initial_value, Tensor):
            shape = shape_or_initial_value.shape
            initial_value = shape_or_initial_value
        else:
            shape = shape_or_initial_value
            initial_value = None

        if bias_correction:
            if initial_value is None:
                initial_value = torch.zeros(shape)

            self.register_buffer('_average', torch.zeros(shape))
            self.register_buffer('_values', initial_value)
            self.register_buffer('_counter', torch.zeros(1, dtype=torch.long))
        else:
            if initial_value is None:
                initial_value = torch.randn(shape)  # TODO

            self.register_buffer('_average', initial_value)

    @property
    def decay(self) -> float:
        """Returns the value of the decay parameter."""
        return self._decay.item()

    @decay.setter
    def decay(self, value: float) -> None:
        """Sets the value of the decay parameter."""
        self._decay.copy_(torch.tensor(value))

    @property
    def average(self) -> Tensor:
        """Returns the current exponential moving average."""
        return self._average

    def update(self, new_values: Tensor) -> Tensor:
        """Updates the exponential moving average, and returns the new average.

        Arguments:
            new_values (Tensor): Tensor containing the new values.

        Returns:
            Tensor containing the new exponential moving average.
        """
        if self._bias_correction:
            self._counter += 1
            self._values -= (self._values - new_values) * (1 - self._decay)
            self._average.copy_(self._values / (1 - torch.pow(self._decay, self._counter[0]).float()))
        else:
            self._average.copy_(self._decay * self._average + (1 - self._decay) * new_values)
        return self._average

    def copy_(self, new_values: Tensor) -> None:
        self._average.copy_(new_values)


class VectorQuantizer(nn.Module):
    """Quantizes vectors using a list of embedding vectors, as used for the vector quantized-variational autoencoder
    (VQ-VAE) [1]. Implementation adopted from TensorFlow version by DeepMind [2].

    Arguments:
        num_embeddings (int): The number of embedding vectors.
        embedding_size (int): The size of each embedding vector.
        commitment_cost (float, optional): The commitment cost used in the loss. Defaults to 0.25.
        exponential_moving_averages (bool, optional): Whether or not to use exponential moving averages to update the
            embedding vectors. For more details, see Appendix A.1 in [1]. Defaults to ``False``.
        ema_decay (float, optional): The decay parameter used for the exponential moving averages, if used.
            Defaults to 0.99.
        ema_epsilon (float, optional): The epsilon parameter used for the exponential moving averages, if used.
            Defaults to 1e-5.

    Inputs: z_e
        **z_e** of shape `(d1, ..., dn, embedding_size)`: Tensor containing the vectors which will be quantized.

    Outputs: z, z_q
        **z** of shape `(d1, ..., dn)`: Tensor containing the indices of the nearest embedding vectors.
        **z_q** of shape `(d1, ..., dn, embedding_size)`: Tensor containing the quantized vectors.

    References:
        [1] Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu (2018). Neural Discrete Representation Learning.
            arXiv preprint: https://arxiv.org/abs/1711.00937
        [2] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_size: int,
            commitment_cost: float = 0.25,
            exponential_moving_averages: bool = False,
            ema_decay: Optional[float] = 0.99,
            ema_epsilon: Optional[float] = 1e-5) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(torch.zeros(num_embeddings, embedding_size))
        limit = np.sqrt(3. / num_embeddings)  # LeCun's uniform initialization
        nn.init.uniform_(self.embeddings, -limit, limit)

        self.register_buffer('_commitment_cost', torch.tensor(commitment_cost, dtype=torch.float))

        self._exponential_moving_averages = exponential_moving_averages
        if exponential_moving_averages:
            assert ema_decay is not None and ema_epsilon is not None
            self._ema_epsilon = ema_epsilon

            self._ema_dw = ExponentialMovingAverages(
                torch.zeros(num_embeddings, embedding_size),
                decay=ema_decay, bias_correction=True)

            self._ema_cluster_sizes = ExponentialMovingAverages(
                torch.zeros(num_embeddings),
                decay=ema_decay, bias_correction=False)

            self.embeddings.requires_grad_(False)

            # nn.functional.one_hot seems to be slow (at least PyTorch 1.6.0),
            # so we cache the results and index the variable
            self.register_buffer(
                '_one_hot_cache', nn.functional.one_hot(torch.arange(num_embeddings), num_embeddings).float(),
                persistent=False)

    @property
    def num_embeddings(self) -> int:
        """Returns the number of embedding vectors."""
        return self.embeddings.shape[0]

    @property
    def embedding_size(self) -> int:
        """Returns the size of each embedding vector."""
        return self.embeddings.shape[1]

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.nearest_indices(z_e)
        z_q = self.lookup(z)
        return z, z_q

    def nearest_indices(self, z_e: Tensor) -> Tensor:
        """TODO docstring"""
        assert z_e.shape[-1] == self.embedding_size
        w = self.embeddings
        z_e_flat = z_e.reshape(-1, self.embedding_size)
        distances_flat = z_e_flat.square().sum(1).unsqueeze(1) - 2 * (z_e_flat @ w.T) + w.square().sum(1).unsqueeze(0)
        z = torch.argmin(distances_flat, dim=-1).reshape(z_e.shape[:-1])
        return z

    def lookup(self, z: Tensor) -> Tensor:
        """TODO docstring"""
        z_q = nn.functional.embedding(z, self.embeddings)
        return z_q

    def compute_loss(
            self,
            z_e: Tensor,
            z: Tensor,
            z_q: Tensor,
            update_ema: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
        """TODO docstring"""
        z_e_loss = nn.functional.mse_loss(z_q.detach(), z_e)
        if self._exponential_moving_averages:
            loss = self._commitment_cost * z_e_loss
            stats = {
                'vq_loss': loss.detach().clone(),
                'z_e_loss': z_e_loss.detach().clone()
            }
            if update_ema:
                self.update_ema(z_e, z)
        else:
            z_q_loss = nn.functional.mse_loss(z_q, z_e.detach())
            loss = z_q_loss + self._commitment_cost * z_e_loss
            stats = {
                'vq_loss': loss.detach().clone(),
                'z_e_loss': z_q_loss.detach().clone(),
                'z_q_loss': z_q_loss.detach().clone()
            }
        return loss, stats

    def update_ema(self, z_e: Tensor, z: Tensor) -> None:
        """TODO docstring"""
        assert self._exponential_moving_averages
        with torch.no_grad():
            flat_z_e = z_e.reshape(-1, self.embedding_size)
            flat_z = z.reshape(-1)
            flat_one_hot_z = self._one_hot_cache[flat_z]
            # without cache:
            # flat_one_hot_z = nn.functional.one_hot(
            #     flat_z, num_classes=num_embeddings)

            cluster_sizes = flat_one_hot_z.sum(0)
            # sum of closest input vectors
            dw = flat_one_hot_z.T @ flat_z_e
            average_cluster_sizes = self._ema_cluster_sizes.update(cluster_sizes)
            average_dw = self._ema_dw.update(dw)

            n = average_cluster_sizes.sum()
            stable_average_cluster_sizes = \
                (average_cluster_sizes + self._ema_epsilon) / (n + self.num_embeddings * self._ema_epsilon) * n

            self.embeddings.data = average_dw / stable_average_cluster_sizes.unsqueeze(1)

class VQVAE(nn.Module, ABC):
    """Base class for VQ-VAEs [1]. Subclasses have to implement `encode()` and `decode()`.

    Arguments:
        num_embeddings (int): The number of embedding vectors.
        embedding_size (int): The size of each embedding vector.
        latent_height (int): The height of the latent variable returned by `encode()`.
        latent_width (int): The width of the latent variable returned by `encode()`.
        exponential_moving_averages (bool, optional): Whether or not to use exponential moving averages to update the
            embedding vectors. For more details, see Appendix A.1 in [1]. Defaults to ``False``.
        commitment_cost (float, optional): The commitment cost used in the loss. Defaults to 0.25.
        ema_decay (float, optional): The decay parameter used for the exponential moving averages, if used.
            Defaults to 0.99.
        ema_epsilon (float, optional): The epsilon parameter used for the exponential moving averages, if used.
            Defaults to 1e-5.

    Inputs: TODO docstring

    Outputs: TODO docstring

    References:
        [1] Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu (2018). Neural Discrete Representation Learning.
            arXiv preprint: https://arxiv.org/abs/1711.00937
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_size: int,
            latent_height: int,
            latent_width: int,
            commitment_cost: float = 0.25,
            exponential_moving_averages: bool = False,
            ema_decay: Optional[float] = 0.99,
            ema_epsilon: Optional[float] = 1e-5) -> None:
        super().__init__()
        self.vq = VectorQuantizer(
            num_embeddings, embedding_size, commitment_cost, exponential_moving_averages, ema_decay, ema_epsilon)
        self.latent_height = latent_height
        self.latent_width = latent_width

        prob = 1. / self.num_embeddings
        kl = -math.log(prob) * (latent_height * latent_width)
        self.register_buffer('_kl', torch.tensor(kl), persistent=False)

    @property
    def num_embeddings(self) -> int:
        """Returns the number of embedding vectors."""
        return self.vq.num_embeddings

    @property
    def embedding_size(self) -> int:
        """Returns the size of each embedding vector."""
        return self.vq.embedding_size

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """TODO docstring"""
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Distribution:
        """TODO docstring"""
        pass

    def prior(self, device: Optional[torch.device] = None) -> Distribution:
        """TODO docstring"""
        if device is None:
            device = next(self.parameters()).device

        h, w = self.latent_shape
        prob = 1 / self.num_embeddings
        probs = torch.full((1, h, w, self.num_embeddings,), prob, dtype=torch.float, device=device)

        return Independent(Categorical(probs=probs), reinterpreted_batch_ndims=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Distribution]:
        z_e = self.encode(x)
        z, z_q = self.quantize(z_e)
        z_q = self.straight_through_estimator(z_q, z_e)
        x_posterior = self.decode(z_q)
        return z_e, z, z_q, x_posterior

    def quantize(self, z_e: Tensor) -> Tuple[Tensor, Tensor]:
        """TODO docstring"""
        z, z_q = self.vq(z_e)
        return z, z_q

    def straight_through_estimator(self, z_q: Tensor, z_e: Tensor) -> Tensor:
        """TODO docstring"""
        return z_e + (z_q - z_e).detach()

    def encode_and_quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """TODO docstring"""
        z_e = self.encode(x)
        z, z_q = self.vq(z_e)
        return z, z_q

    def lookup(self, z: Tensor) -> Tensor:
        """TODO docstring"""
        z_q = self.vq.lookup(z)
        return z_q

    def compute_loss(
            self,
            x: Tensor,
            z_e: Tensor,
            z: Tensor,
            z_q: Tensor,
            x_posterior: Distribution) -> Tuple[Tensor, Dict[str, Tensor]]:
        """TODO docstring"""
        # ELBO = E[log p(x|z)] - KL(q(z)||p(z))
        log_likelihood = x_posterior.log_prob(x).mean(0)
        elbo = log_likelihood - self._kl

        stats = {'elbo': elbo.detach().clone(),
                 'log_likelihood': log_likelihood.detach().clone(),
                 'kl': self._kl.clone()}

        vq_loss, vq_stats = self.vq.compute_loss(z_e, z, z_q)
        loss = -elbo + vq_loss
        stats.update(vq_stats)

        return loss, stats

    def sample(
            self,
            sample_shape: torch.Size = torch.Size(),
            sample_decoder: bool = False,
            device: Optional[torch.device] = None) -> Tensor:
        """TODO docstring"""
        with torch.no_grad():
            if device is None:
                device = next(self.parameters()).device
            z_prior = self.prior(device=device)
            z = z_prior.sample(sample_shape)
            x_posterior = self.decode(z)
            x = x_posterior.sample() if sample_decoder else mode(x_posterior)
            return x

    def reconstruct(self, x: Tensor, sample: bool = False) -> Tensor:
        """TODO docstring"""
        with torch.no_grad():
            z_e, z, z_q, x_posterior = self.forward(x)
            x_recon = x_posterior.sample() if sample else mode(x_posterior)
            return x_recon


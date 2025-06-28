# By: Chance Brownfield
# Cerebrum
# "0.1.0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import tempfile
import random
import string
import math

# Try to import numpy, but don't fail if it's not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Using PyTorch tensors instead.")
    # Create a mock numpy module using torch
    class MockNumpy:
        def __init__(self):
            self.random = MockRandom()
        
        def array(self, data, **kwargs):
            if isinstance(data, torch.Tensor):
                return data
            return torch.tensor(data, **kwargs)
        
        def vstack(self, arrays):
            if isinstance(arrays[0], torch.Tensor):
                return torch.cat(arrays, dim=0)
            return torch.tensor(arrays)
        
        def random(self):
            return self.random
    
    class MockRandom:
        def __init__(self):
            self.seed = lambda x: torch.manual_seed(x)
            self.randn = lambda *args: torch.randn(*args)
    
    np = MockNumpy()

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union


# --- Configuration System ---

@dataclass
class CerebrumConfig:
    """Configuration class for different Cerebrum types."""
    
    # Model type
    model_type: str = "mmm"  # "gmm", "hmm", "mmm"
    
    # Architecture parameters
    input_dim: int = 64
    hidden_dim: int = 128
    z_dim: int = 32
    rnn_hidden: int = 64
    num_states: int = 8
    n_mix: int = 4
    trans_d_model: int = 128
    trans_nhead: int = 8
    trans_layers: int = 4
    output_dim: int = 64
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    kl_anneal_epochs: int = 50
    clip_norm: float = 5.0
    label_smoothing: float = 0.1
    entropy_coef: float = 0.01
    
    # Task-specific parameters
    vocab_size: Optional[int] = None
    max_sequence_length: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'z_dim': self.z_dim,
            'rnn_hidden': self.rnn_hidden,
            'num_states': self.num_states,
            'n_mix': self.n_mix,
            'trans_d_model': self.trans_d_model,
            'trans_nhead': self.trans_nhead,
            'trans_layers': self.trans_layers,
            'output_dim': self.output_dim
        }


# Predefined configurations for different Cerebrum types
class CerebrumConfigs:
    """Predefined configurations for different Cerebrum applications."""
    
    @staticmethod
    def eeg_cerebrum() -> CerebrumConfig:
        """Configuration for EEG seizure prediction/recognition."""
        return CerebrumConfig(
            model_type="mmm",
            input_dim=64,  # EEG channels
            hidden_dim=256,
            z_dim=128,
            rnn_hidden=128,
            num_states=16,  # More states for complex brain patterns
            n_mix=8,
            trans_d_model=256,
            trans_nhead=16,
            trans_layers=6,
            output_dim=64,
            learning_rate=5e-5,
            kl_anneal_epochs=100,
            max_sequence_length=1024  # Longer sequences for EEG
        )
    
    @staticmethod
    def tts_cerebrum() -> CerebrumConfig:
        """Configuration for Text-to-Speech voice cloning."""
        return CerebrumConfig(
            model_type="mmm",
            input_dim=80,  # Mel spectrogram features
            hidden_dim=512,
            z_dim=256,
            rnn_hidden=256,
            num_states=32,  # Many states for voice characteristics
            n_mix=16,
            trans_d_model=512,
            trans_nhead=16,
            trans_layers=8,
            output_dim=80,
            learning_rate=1e-4,
            kl_anneal_epochs=50,
            max_sequence_length=2048  # Long audio sequences
        )
    
    @staticmethod
    def speaker_recognition_cerebrum() -> CerebrumConfig:
        """Configuration for Speaker Recognition."""
        return CerebrumConfig(
            model_type="mmm",
            input_dim=40,  # MFCC features
            hidden_dim=256,
            z_dim=128,
            rnn_hidden=128,
            num_states=24,  # States for different speakers
            n_mix=12,
            trans_d_model=256,
            trans_nhead=8,
            trans_layers=4,
            output_dim=40,
            learning_rate=1e-4,
            kl_anneal_epochs=30,
            max_sequence_length=800  # Medium audio sequences
        )
    
    @staticmethod
    def conversational_cerebrum() -> CerebrumConfig:
        """Configuration for Conversational AI text generation."""
        return CerebrumConfig(
            model_type="mmm",
            input_dim=512,  # Token embeddings
            hidden_dim=1024,
            z_dim=512,
            rnn_hidden=512,
            num_states=64,  # Many states for language patterns
            n_mix=32,
            trans_d_model=1024,
            trans_nhead=16,
            trans_layers=12,
            output_dim=512,
            vocab_size=50000,  # Large vocabulary
            learning_rate=5e-5,
            kl_anneal_epochs=100,
            label_smoothing=0.1,
            max_sequence_length=2048  # Long text sequences
        )


# --- Building Blocks ---

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc_out(h))


class TinyDiffusionBlock(nn.Module):
    """
    A minimal denoiser that takes (B, D_block) and returns refined (B, D_block).
    In practice this would be a U‑Net or small transformer; here it's an MLP.
    """

    def __init__(self, block_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(block_dim, block_dim * 2),
            nn.ReLU(),
            nn.Linear(block_dim * 2, block_dim)
        )

    def denoise(self, x):
        # x: (B, block_dim)
        return x + 0.1 * self.net(x)  # residual step


class DraftScorer(nn.Module):
    """
    Scores a candidate token sequence (B, T) in latent space,
    by cross‑attending it to z_plan and measuring self‑reconstruction.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, draft_tokens, z_plan):
        # draft_tokens: (B, T), z_plan: (B, Z)
        B, T = draft_tokens.shape

        # 1) Embed the draft
        emb = self.model.token_embedding(draft_tokens)  # (B,T,Z)
        emb = emb.permute(1, 0, 2)  # (T,B,Z)

        # 2) Cross‑attend draft → plan under mixed precision
        device_type = 'cuda' if z_plan.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            # transformer(src, tgt): src=(1,B,Z), tgt=(T,B,Z)
            attended = self.model.transformer(
                z_plan.unsqueeze(0),  # (1,B,Z) as encoder input
                emb  # (T,B,Z) as decoder input
            )  # returns (T,B,Z)

        # 3) Bring back to (B,T,Z)
        attended = attended.permute(1, 0, 2)  # (B,T,Z)

        # 4) Decode with your auto_decoder & to_vocab
        #    (teacher‐force decode to itself for self‑reconstruction score)
        dec_in = attended.permute(1, 0, 2)  # (T,B,Z)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(z_plan.device)
        out_z = self.model.auto_decoder(dec_in, dec_in, tgt_mask=mask)  # (T,B,Z)
        logits = self.model.to_vocab(out_z.permute(1, 0, 2))  # (B,T,V)

        # 5) Negative log‑likelihood of draft under its own logits
        ll = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            draft_tokens.reshape(-1),
            reduction='mean'
        )
        return ll


class RecurrentNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_states):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.state_emissions = nn.Linear(hidden_dim, num_states)
        self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states))

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        emissions = F.log_softmax(self.state_emissions(rnn_out), dim=-1)
        transitions = F.log_softmax(self.transition_matrix, dim=-1)
        return emissions, transitions


class GaussianMixture(nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_features))

    def get_weights(self):
        return F.softmax(self.logits, dim=0)

    def get_means(self):
        return self.means

    def get_variances(self):
        return torch.exp(self.log_vars)

    def log_prob(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.means.dtype, device=self.means.device)
        else:
            X = X.to(self.means.device).type(self.means.dtype)
        N, D = X.shape
        diff = X.unsqueeze(1) - self.means.unsqueeze(0)
        inv_vars = torch.exp(-self.log_vars)
        exp_term = -0.5 * torch.sum(diff * diff * inv_vars.unsqueeze(0), dim=2)
        log_norm = -0.5 * (torch.sum(self.log_vars, dim=1) + D * math.log(2 * math.pi))
        comp_log_prob = exp_term + log_norm.unsqueeze(0)
        log_weights = F.log_softmax(self.logits, dim=0)
        weighted = comp_log_prob + log_weights.unsqueeze(0)
        return torch.logsumexp(weighted, dim=1)

    def get_log_likelihoods(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.means.dtype, device=self.means.device)
        else:
            X = X.to(self.means.device).type(self.means.dtype)
        with torch.no_grad():
            ll = self.log_prob(X)
        return ll.cpu().numpy()

    def score(self, X):
        ll = self.get_log_likelihoods(X)
        return float(ll.mean())


class HiddenMarkov(nn.Module):
    def __init__(self, n_states, n_mix, n_features):
        super().__init__()
        self.n_states = n_states
        self.n_mix = n_mix
        self.n_features = n_features
        self.pi_logits = nn.Parameter(torch.zeros(n_states))
        self.trans_logits = nn.Parameter(torch.zeros(n_states, n_states))
        self.weight_logits = nn.Parameter(torch.zeros(n_states, n_mix))
        self.means = nn.Parameter(torch.randn(n_states, n_mix, n_features))
        self.log_vars = nn.Parameter(torch.zeros(n_states, n_mix, n_features))

    def get_initial_prob(self):
        return F.softmax(self.pi_logits, dim=0)

    def get_transition_matrix(self):
        return F.softmax(self.trans_logits, dim=1)

    def get_weights(self):
        return F.softmax(self.weight_logits, dim=1)

    def get_means(self):
        return self.means

    def get_variances(self):
        return torch.exp(self.log_vars)

    def log_prob(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.means.dtype, device=self.means.device)
        else:
            X = X.to(self.means.device).type(self.means.dtype)
        T, D = X.shape
        diff = X.unsqueeze(1).unsqueeze(2) - self.means.unsqueeze(0)
        inv_vars = torch.exp(-self.log_vars)
        exp_term = -0.5 * torch.sum(diff * diff * inv_vars.unsqueeze(0), dim=3)
        log_norm = -0.5 * (torch.sum(self.log_vars, dim=2) + D * math.log(2 * math.pi))
        comp_log_prob = exp_term + log_norm.unsqueeze(0)
        log_mix_weights = F.log_softmax(self.weight_logits, dim=1)
        weighted = comp_log_prob + log_mix_weights.unsqueeze(0)
        emission_log_prob = torch.logsumexp(weighted, dim=2)
        log_pi = F.log_softmax(self.pi_logits, dim=0)
        log_A = F.log_softmax(self.trans_logits, dim=1)
        log_alpha = torch.zeros(T, self.n_states, dtype=X.dtype, device=X.device)
        log_alpha[0] = log_pi + emission_log_prob[0]
        for t in range(1, T):
            prev = log_alpha[t - 1].unsqueeze(1)
            log_alpha[t] = emission_log_prob[t] + torch.logsumexp(prev + log_A, dim=1)
        return torch.logsumexp(log_alpha[-1], dim=0)

    def get_log_likelihoods(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.means.dtype, device=self.means.device)
        else:
            X = X.to(self.means.device).type(self.means.dtype)
        with torch.no_grad():
            if X.dim() == 3:
                return [self.log_prob(seq).item() for seq in X]
            else:
                return [self.log_prob(X).item()]

    def score(self, X):
        lls = self.get_log_likelihoods(X)
        return float(sum(lls) / len(lls))


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src_emb = self.input_proj(src)
        tgt_emb = self.input_proj(tgt)
        out = self.transformer(src_emb, tgt_emb)
        return self.output_proj(out)


class MultiMixtureTransformer(nn.Module):
    """
    Variational Encoder + RNN-HMM + Hidden GMM + Transformer hybrid.
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 z_dim,
                 rnn_hidden,
                 num_states,
                 n_mix,
                 trans_d_model,
                 trans_nhead,
                 trans_layers,
                 output_dim,
                 token_embedding=None,
                 vocab_size=None):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)
        self.rn = RecurrentNetwork(z_dim, rnn_hidden, num_states)
        self.hm = HiddenMarkov(num_states, n_mix, z_dim)
        self.transformer = TimeSeriesTransformer(
            input_dim=z_dim,
            d_model=trans_d_model,
            nhead=trans_nhead,
            num_layers=trans_layers,
            output_dim=output_dim
        )
        self.pred_weights = nn.Parameter(torch.ones(z_dim))
        self.recog_weights = nn.Parameter(torch.ones(z_dim))
        self.gen_weights = nn.Parameter(torch.ones(z_dim))
        self.reg_weights = nn.Parameter(torch.ones(z_dim))
        self.rules = {}
        # optional regression components
        self.token_embedding = token_embedding
        if token_embedding is not None and vocab_size is not None:
            decoder_layer = nn.TransformerDecoderLayer(d_model=z_dim, nhead=trans_nhead)
            self.auto_decoder = nn.TransformerDecoder(decoder_layer, num_layers=trans_layers)
            self.to_vocab = nn.Linear(z_dim, vocab_size)
        else:
            self.auto_decoder = None
            self.to_vocab = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def add_rule(self, name, label, reward_target, reward):
        assert label in ('pred', 'recog', 'gen', 'reg')
        self.rules[name] = {
            'label': label,
            'target': torch.tensor(reward_target, device=self.pred_weights.device),
            'reward': reward
        }

    def remove_rule(self, name):
        self.rules.pop(name, None)

    def check_rule(self, name):
        return self.rules.get(name)

    def back_loss(self):
        loss = 0.0
        for rule in self.rules.values():
            w = getattr(self, f"{rule['label']}_weights")
            # you could also do something like (w - rule['attention']).abs().mean()
            current = w.mean()
            diff = (current - rule['target']).pow(2)
            loss = loss + rule['reward'] * diff
        return loss

    def get_kl_annealing_weight(self, epoch, total_epochs, anneal_type='linear'):
        """
        Compute KL annealing weight to prevent mode collapse.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            anneal_type: Type of annealing ('linear', 'cosine', 'step')
        
        Returns:
            float: Annealing weight between 0 and 1
        """
        if total_epochs == 0:
            return 1.0
            
        progress = epoch / total_epochs
        
        if anneal_type == 'linear':
            return min(1.0, progress)
        elif anneal_type == 'cosine':
            return 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif anneal_type == 'step':
            # Step annealing: 0 for first 20%, then 0.5 for 20-60%, then 1.0
            if progress < 0.2:
                return 0.0
            elif progress < 0.6:
                return 0.5
            else:
                return 1.0
        else:
            return min(1.0, progress)

    def training_step(self, x, optimizer, epoch=0, total_epochs=100, anneal_type='linear'):
        """
        Enhanced training step with KL annealing and better loss computation.
        """
        outputs = self.forward(x)
        base_loss = self.loss(x, outputs)
        
        # Compute KL annealing weight
        kl_weight = self.get_kl_annealing_weight(epoch, total_epochs, anneal_type)
        
        # Apply KL annealing to the loss
        if hasattr(outputs, 'kld'):
            total_loss = base_loss + kl_weight * outputs['kld']
        else:
            total_loss = base_loss
            
        # Add auxiliary rule losses
        rule_loss = self.back_loss()
        total_loss = total_loss + rule_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.detach(),
            'base_loss': base_loss.detach(),
            'kl_weight': kl_weight,
            'rule_loss': rule_loss.detach() if isinstance(rule_loss, torch.Tensor) else torch.tensor(rule_loss)
        }

    def forward(self, x, tgt=None):
        """
        Forward pass with comprehensive error handling and logging.
        """
        try:
            # Input validation
            if x is None:
                raise ValueError("Input tensor x cannot be None")
            
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Input x must be a torch.Tensor, got {type(x)}")
            
            if x.dim() not in [2, 3]:
                raise ValueError(f"Input x must be 2D or 3D tensor, got shape {x.shape}")
            
            # Store original shape for reconstruction
            original_shape = x.shape
            
            if x.dim() == 3:
                T, B, _ = x.size()
                zs, mus, logvars = [], [], []
                for t in range(T):
                    try:
                        mu_t, logvar_t = self.encoder(x[t])
                        z_t = self.reparameterize(mu_t, logvar_t)
                        zs.append(z_t)
                        mus.append(mu_t)
                        logvars.append(logvar_t)
                    except Exception as e:
                        raise RuntimeError(f"Encoding failed at timestep {t}: {e}")
                zs = torch.stack(zs)
                mus = torch.stack(mus)
                logvars = torch.stack(logvars)
            else:
                try:
                    mu, logvar = self.encoder(x)
                    zs = self.reparameterize(mu, logvar)
                    mus, logvars = mu, logvar
                except Exception as e:
                    raise RuntimeError(f"Single-step encoding failed: {e}")

            # Reconstruction
            try:
                recon = self.decoder(zs.view(-1, zs.size(-1))).view_as(x)
            except Exception as e:
                raise RuntimeError(f"Reconstruction failed: {e}")
            
            # RNN and HMM processing
            try:
                emissions, transitions = self.rn(zs.permute(1, 0, 2))
                flat_z = zs.view(-1, zs.size(-1))
                seq_ll = self.hm.log_prob(flat_z)
                hgmm_ll = seq_ll.view(1, 1, 1).expand_as(emissions)
            except Exception as e:
                raise RuntimeError(f"RNN/HMM processing failed: {e}")
            
            # Optional transformer pass
            trans_out = None
            if tgt is not None:
                try:
                    trans_out = self.transformer(zs, tgt)
                except Exception as e:
                    raise RuntimeError(f"Transformer forward pass failed: {e}")

            return {
                'reconstruction': recon,
                'mu': mus,
                'logvar': logvars,
                'emissions': emissions,
                'transitions': transitions,
                'hgmm_log_likelihood': hgmm_ll,
                'transformer_out': trans_out,
                'original_shape': original_shape
            }
            
        except Exception as e:
            # Log the error with context
            error_msg = f"Forward pass failed: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def loss(self, x, outputs):
        recon, mu, logvar = outputs['reconstruction'], outputs['mu'], outputs['logvar']
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        hgmm_nll = -torch.sum(outputs['hgmm_log_likelihood'])
        return recon_loss + kld + hgmm_nll

    def predict(self, x):
        """
        Given x, predict next‐step (or next‐sequence) by:
         1) encoding to z,
         2) reweighting latent dims by pred_weights,
         3) decoding back to input space.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        # elementwise weight on latent dims
        z_pred = z * torch.sigmoid(self.pred_weights)
        return self.decoder(z_pred)

    def predict_loss(self, x, target, reward):
        """
        MSE between predict(x) and target,
        weighted by a scalar reward (+/-).
        """
        pred = self.predict(x)
        loss = F.mse_loss(pred, target, reduction='mean')
        # reward >1 amplifies, <1 punishes
        return reward * loss

    def recognize(self, x, tgt_z=None):
        """
        Recognize: map x→z, then transform to tgt_z space via transformer,
        then decode to reconstruct in original space.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        # optionally refine z via transformer with some target embedding
        if tgt_z is not None:
            # expand dims: (S,B,Z) expected by transformer
            z_in = z.unsqueeze(0)
            tgt = tgt_z.unsqueeze(0)
            z_out = self.transformer(z_in, tgt).squeeze(0)
        else:
            z_out = z
        z_rec = z_out * torch.sigmoid(self.recog_weights)
        return self.decoder(z_rec)

    def recognition_loss(self, x, target, reward):
        """
        Recon loss between recognize(x) and target,
        weighted by reward.
        """
        rec = self.recognize(x)
        loss = F.mse_loss(rec, target, reduction='mean')
        return reward * loss

    def generate(self, num_steps, batch_size=1, z0=None):
        """
        Generate a sequence of length num_steps by:
         1) sampling initial z from prior (HMM's mixture),
         2) rolling it through the RNN-HMM to get a latent trajectory,
         3) reweight by gen_weights and decode each step.
        """
        # sample initial state from HMM pi
        pi = self.hm.get_initial_prob().detach()
        state = torch.multinomial(pi, num_samples=batch_size, replacement=True)
        z = []
        for t in range(num_steps):
            # for each chosen state, pick a mixture component
            w = self.hm.get_weights()[state]  # (B, n_mix)
            mix_idx = torch.multinomial(w, 1).squeeze(-1)
            mu_t = self.hm.get_means()[state, mix_idx]
            # reweight latent dims
            z_t = mu_t * torch.sigmoid(self.gen_weights)
            z.append(z_t)
            # advance state by sampling from transition
            A = self.hm.get_transition_matrix()[state]
            state = torch.multinomial(A, 1).squeeze(-1)
        Z = torch.stack(z, dim=0)  # (T, B, Z)
        # decode each
        recon = self.decoder(Z.view(-1, Z.size(-1))).view(num_steps, batch_size, -1)
        return recon

    def generation_loss(self, generated, target_seq, reward):
        """
        Sequence‐level loss between generated and target,
        weighted by reward (+/-).
        """
        loss = F.mse_loss(generated, target_seq, reduction='mean')
        return reward * loss

    def regression(self, context_sequence, max_len=50, start_token=0,
                   K=4,  # number of diffusion blocks
                   num_drafts=3,  # CoRe² drafts
                   top_p=0.9,  # nucleus sampling
                   entropy_coef=0.01  # encourage diversity
                   ):
        """
        Unified AR+diffusion+CoRe² regression.
        context_sequence: (L, B, z_dim)
        """
        L, B, Z = context_sequence.size()

        # — 1) Build latent plan z_plan with your existing VAE+RNN-HMM-GMM —
        zs = []
        for t in range(L):
            mu, logvar = self.encoder(context_sequence[t])
            zs.append(self.reparameterize(mu, logvar))
        z_plan = torch.stack(zs).mean(0) * torch.sigmoid(self.reg_weights)  # (B, Z)

        # — 2) Block‑diffusion: split into K sub‑blocks and denoise each —
        block_size = Z // K
        refined_blocks = []
        for k in range(K):
            blk = z_plan[:, k * block_size:(k + 1) * block_size]
            denoiser = TinyDiffusionBlock(block_size).to(z_plan.device)
            refined_blocks.append(denoiser.denoise(blk))
        z_refined = torch.cat(refined_blocks, dim=-1)  # (B, Z)

        # — 3) CoRe²: draft N candidates in parallel via top‑p AR —
        drafts = []
        for _ in range(num_drafts):
            seq = []
            input_tok = torch.full((B, 1), start_token, dtype=torch.long, device=z_plan.device)
            for _ in range(max_len):
                tok_emb = self.token_embedding(input_tok).squeeze(1)  # (B, Z)
                # cross‑attend to refined plan under mixed precision
                device_type = 'cuda' if z_refined.is_cuda else 'cpu'
                # enable only on GPU, but cpu also accepts the API call
                with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                    attn = self.transformer(
                        z_refined.unsqueeze(0),  # (1,B,Z)
                        tok_emb.unsqueeze(0)  # (1,B,Z)
                    ).squeeze(0)
                logits = self.to_vocab(attn)  # (B,V)
                # top‑p sampling
                probs = torch.softmax(logits, dim=-1)
                sorted_p, indices = torch.sort(probs, descending=True, dim=-1)
                cum_p = sorted_p.cumsum(dim=-1)
                mask = cum_p > top_p
                sorted_p = sorted_p.masked_fill(mask, 0.0)
                sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                next_tok = indices.gather(1, torch.multinomial(sorted_p, 1))
                seq.append(next_tok)
                input_tok = next_tok
            drafts.append(torch.cat(seq, dim=1))  # (B,max_len)

        # — 4) Reflect: score each draft and pick the best —
        scorer = DraftScorer(self).to(z_plan.device)
        scores = torch.stack([scorer(d, z_refined) for d in drafts], dim=0)  # (num_drafts,)
        best_idx = scores.argmax(dim=0)  # pick the draft with highest scalar score
        best_draft = drafts[best_idx]  # (B, max_len)

        # — 5) Refine: run one pass of the auto_decoder over the best draft —
        emb = self.token_embedding(best_draft)  # (B, T, Z)
        dec_in = (emb + z_refined.unsqueeze(1)).permute(1, 0, 2)  # (T,B,Z)
        mask = nn.Transformer.generate_square_subsequent_mask(dec_in.size(0)).to(z_plan.device)
        out = self.auto_decoder(dec_in, dec_in, tgt_mask=mask)  # (T,B,Z)

        # — 6) Compute final logits and optionally add entropy regularization —
        logits = self.to_vocab(out.permute(1, 0, 2))  # (B,T,V)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        return logits, -entropy_coef * entropy

    def regression_loss(self,
                        context_sequence,  # (L, B, z_dim)
                        target_tokens,  # (B, T)
                        max_len=None,
                        K=4,
                        entropy_coef=0.01,
                        label_smoothing=0.1):
        if self.auto_decoder is None:
            raise RuntimeError('Autoregressive components not initialized')
        L, B, Z = context_sequence.size()
        T = target_tokens.size(1)
        max_len = T - 1 if max_len is None else max_len

        # — 1) Build latent plan
        zs = [self.reparameterize(*self.encoder(context_sequence[t]))
              for t in range(L)]
        z_plan = torch.stack(zs).mean(0) * torch.sigmoid(self.reg_weights)  # (B, Z)

        # — 2) Block‑diffusion on plan
        block_size = Z // K
        refined = []
        for k in range(K):
            blk = z_plan[:, k * block_size:(k + 1) * block_size]
            denoiser = TinyDiffusionBlock(block_size).to(z_plan.device)
            refined.append(denoiser.denoise(blk))
        z_refined = torch.cat(refined, -1)  # (B, Z)

        # — 3) Teacher‑forced embedding + plan infusion
        #    Input tokens shifted right for teacher forcing
        inp = target_tokens[:, :-1]  # (B, T-1)
        lbl = target_tokens[:, 1:]  # (B, T-1)
        emb = self.token_embedding(inp)  # (B, T-1, Z)
        dec_in = (emb + z_refined.unsqueeze(1)).permute(1, 0, 2)  # (T-1,B,Z)

        # — 4) AR decode
        mask = nn.Transformer.generate_square_subsequent_mask(dec_in.size(0)).to(z_plan.device)
        out = self.auto_decoder(dec_in, dec_in, tgt_mask=mask)  # (T-1,B,Z)

        # — 5) Final logits and CE with label smoothing
        logits = self.to_vocab(out).permute(1, 0, 2)  # (B, T-1, V)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            lbl.reshape(-1),
            reduction='mean',
            label_smoothing=label_smoothing
        )

        # — 6) Entropy regularization—discourage collapse
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        return ce_loss - entropy_coef * entropy


class MM(nn.Module):
    """Multi-Mixture."""

    def __init__(self, n_components, n_features, model_type='gmm', n_mix=1):
        super().__init__()
        self.model_type = model_type.lower()
        self.n_features = n_features
        self.gmms = []
        self.hgmm_models = {}
        self.active_hmm = None
        if self.model_type == 'gmm':
            self.gmm = GaussianMixture(n_components, n_features)
        elif self.model_type == 'hgmm':
            self.hm = HiddenMarkov(n_components, n_mix, n_features)
        else:
            raise ValueError("model_type must be 'gmm' or 'hgmm'")

    def _prepare_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X.float()

    def fit(self, X, init_params=None, lr=1e-2, epochs=100, verbose=False, data_id=None):
        if init_params is not None:
            self.import_model(init_params)

        X_tensor = self._prepare_tensor(X).to(next(self.parameters()).device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            if self.model_type == 'gmm':
                loss = -torch.mean(self.gmm.log_prob(X_tensor))
            else:
                if X_tensor.dim() == 3:
                    loss = -sum(self.hm.log_prob(seq) for seq in X_tensor) / X_tensor.size(0)
                else:
                    loss = -self.hm.log_prob(X_tensor)
            loss.backward()
            optimizer.step()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if self.model_type == 'gmm':
            if data_id is None:
                data_id = len(self.gmms)
            while isinstance(data_id, int) and data_id < len(self.gmms) and self.gmms[data_id] is not None:
                data_id += 1
            if data_id == len(self.gmms):
                self.gmms.append(self.gmm)
            else:
                self.gmms[data_id] = self.gmm
        else:
            if data_id is None:
                while True:
                    data_id = ''.join(random.choices(string.ascii_lowercase, k=6))
                    if data_id not in self.hgmm_models:
                        break
            self.hgmm_models[data_id] = self.hm
            self.active_hmm = data_id

        return data_id

    def unfit(self, data_id):
        if isinstance(data_id, int):
            if 0 <= data_id < len(self.gmms):
                del self.gmms[data_id]
            else:
                raise ValueError(f"GMM with id {data_id} does not exist.")
        elif isinstance(data_id, str):
            if data_id in self.hgmm_models:
                del self.hgmm_models[data_id]
                if self.active_hmm == data_id:
                    self.active_hmm = None
            else:
                raise ValueError(f"HMM model with name '{data_id}' does not exist.")
        else:
            raise TypeError("data_id must be an int (GMM) or str (HMM)")

    def check_data(self):
        data = {i: 'gmm' for i in range(len(self.gmms))}
        data.update({name: 'hmm' for name in self.hgmm_models.keys()})
        return data

    def score(self, X):
        with torch.no_grad():
            X_tensor = self._prepare_tensor(X).to(next(self.parameters()).device)
            if self.model_type == 'gmm':
                return float(self.gmm.log_prob(X_tensor).mean().cpu().item())
            else:
                if X_tensor.dim() == 3:
                    return float(sum(self.hm.log_prob(seq).item() for seq in X_tensor) / X_tensor.size(0))
                else:
                    return float(self.hm.log_prob(X_tensor).cpu().item())

    def get_log_likelihoods(self, X):
        with torch.no_grad():
            X_tensor = self._prepare_tensor(X).to(next(self.parameters()).device)
            if self.model_type == 'gmm':
                return self.gmm.log_prob(X_tensor).cpu().numpy()
            else:
                if X_tensor.dim() == 3:
                    return [self.hm.log_prob(seq).item() for seq in X_tensor]
                else:
                    return [self.hm.log_prob(X_tensor).item()]

    def get_means(self):
        return (self.gmm if self.model_type == 'gmm' else self.hgmm).get_means().cpu().detach().numpy()

    def get_variances(self):
        return (self.gmm if self.model_type == 'gmm' else self.hgmm).get_variances().cpu().detach().numpy()

    def get_weights(self):
        return (self.gmm if self.model_type == 'gmm' else self.hgmm).get_weights().cpu().detach().numpy()

    def export_model(self, filepath=None):
        state = self.state_dict()
        if filepath:
            torch.save(state, filepath)
        return state

    def import_model(self, source):
        if isinstance(source, str):
            state = torch.load(source)
        elif isinstance(source, dict):
            state = source
        else:
            raise ValueError("Unsupported source for import_model")
        self.load_state_dict(state)


class MMMan(nn.Module):
    """Multi-Mixture Manager."""

    def __init__(self):
        super().__init__()
        self.gmms = []  # List of GaussianMixture models
        self.hgmm_models = {}  # Dict of HM models keyed by string IDs
        self.active_hmm = None  # Optional: active HGMM for scoring/fitting

    def _generate_unique_id(self):
        while True:
            candidate = ''.join(random.choices(string.ascii_lowercase, k=6))
            if candidate not in self.hgmm_models:
                return candidate

    def fit(self, data=None, model_type='gmm', n_components=1, n_features=1, n_mix=1,
            data_id=None, init_params=None, lr=1e-2, epochs=100):
        """
        Fit or absorb a model:
        - If `data` is a tensor/array, fit a new model.
        - If `data` is a pre-trained model, absorb it directly.
        - `data_id` determines storage; if None, generate a unique one.
        """
        if model_type == 'gmm':
            if data_id is None:
                data_id = len(self.gmms)
                while data_id < len(self.gmms) and self.gmms[data_id] is not None:
                    data_id += 1
            if isinstance(data, GaussianMixture):
                # Absorb pretrained model
                if data_id < len(self.gmms):
                    self.gmms[data_id] = data
                else:
                    while len(self.gmms) < data_id:
                        self.gmms.append(None)
                    self.gmms.append(data)
            else:
                # Train new model
                model = MM(n_components, n_features, model_type='gmm')
                model.fit(data, init_params=init_params, lr=lr, epochs=epochs)
                if data_id < len(self.gmms):
                    self.gmms[data_id] = model.gmm
                else:
                    while len(self.gmms) < data_id:
                        self.gmms.append(None)
                    self.gmms.append(model.gmm)
        elif model_type == 'hmm':
            if data_id is None:
                data_id = self._generate_unique_id()
            if isinstance(data, HiddenMarkov):
                self.hgmm_models[data_id] = data
            else:
                model = MM(n_components, n_features, model_type='hmm', n_mix=n_mix)
                model.fit(data, init_params=init_params, lr=lr, epochs=epochs)
                self.hgmm_models[data_id] = model.hm
        else:
            raise ValueError("model_type must be 'gmm' or 'hmm'")
        return data_id

    def export_model(self, data_id):
        """
        Export the model associated with the data_id.
        Returns a GaussianMixture or HiddenMarkov instance.
        """
        if isinstance(data_id, int):
            if 0 <= data_id < len(self.gmms):
                return self.gmms[data_id]
            else:
                raise ValueError(f"GMM with id {data_id} does not exist.")
        elif isinstance(data_id, str):
            if data_id in self.hgmm_models:
                return self.hgmm_models[data_id]
            else:
                raise ValueError(f"HMM model with name '{data_id}' does not exist.")
        else:
            raise TypeError("data_id must be an int (GMM) or str (HMM)")

    def unfit(self, data_id):
        """
        Remove a model from the internal storage (GMM or HMM).
        """
        if isinstance(data_id, int):
            if 0 <= data_id < len(self.gmms):
                del self.gmms[data_id]
            else:
                raise ValueError(f"GMM with id {data_id} does not exist.")
        elif isinstance(data_id, str):
            if data_id in self.hgmm_models:
                del self.hgmm_models[data_id]
                if self.active_hmm == data_id:
                    self.active_hmm = None
            else:
                raise ValueError(f"HMM model with name '{data_id}' does not exist.")
        else:
            raise TypeError("data_id must be an int (GMM) or str (HMM)")

    def check_data(self):
        """
        Returns a dict mapping each stored data's ID to its type.

        - Integer keys → 'gmm'
        - String keys   → 'hmm'
        """
        data = {i: 'gmm' for i in range(len(self.gmms)) if self.gmms[i] is not None}
        data.update({name: 'hmm' for name in self.hgmm_models.keys()})
        return data

    def _all_ids(self):
        return list(self.check_data().keys())

    def _normalize_ids(self, data_ids):
        if data_ids is None:
            return self._all_ids()
        if isinstance(data_ids, (int, str)):
            return [data_ids]
        return list(data_ids)

    def _get_submodel(self, data_id):
        if isinstance(data_id, int):
            return self.gmms[data_id]
        return self.hgmm_models[data_id]

    def get_means(self, data_ids=None):
        """
        If data_ids is None, returns a dict {id: means} for all components;
        if a single id, returns just that component's means (numpy array);
        if a list/tuple, returns a dict.
        """
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).get_means() for d in ids}
        # unwrap singletons
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

    def get_variances(self, data_ids=None):
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).get_variances() for d in ids}
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

    def get_weights(self, data_ids=None):
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).get_weights() for d in ids}
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

    def score(self, X, data_ids=None):
        """
        Average log-likelihood(s) of X under each specified component.
        """
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).score(X) for d in ids}
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out

    def get_log_likelihoods(self, X, data_ids=None):
        """
        Per-sample log-likelihood(s) of X under each specified component.
        """
        ids = self._normalize_ids(data_ids)
        out = {d: self._get_submodel(d).get_log_likelihoods(X) for d in ids}
        if isinstance(data_ids, (int, str)):
            return out[ids[0]]
        return out


class Cerebrum(nn.Module):
    """
    Manager for multiple models: GMM, HMM, and MMM.
    """

    def __init__(self):
        super().__init__()
        self.models = nn.ModuleDict()

    def _generate_unique_id(self, prefix='model'):
        while True:
            candidate = f"{prefix}_{''.join(random.choices(string.ascii_lowercase, k=6))}"
            if candidate not in self.models:
                return candidate

    def add_model(self, model: nn.Module, model_id: str = None):
        if model_id is None:
            model_id = self._generate_unique_id(model.__class__.__name__)
        if model_id in self.models:
            raise KeyError(f"Model with id '{model_id}' already exists.")
        self.models[model_id] = model
        return model_id

    def fit_and_add(self,
                    data,
                    config: Optional[CerebrumConfig] = None,
                    model_type: str = 'gmm',
                    model_id: str = None,
                    kl_anneal_epochs: int = 0,
                    clip_norm: float = 5.0,
                    weight_decay: float = 1e-5,
                    **kwargs):
        """
        Fit and add a model using configuration or legacy parameters.
        
        Args:
            data: Training data
            config: CerebrumConfig object (preferred)
            model_type: Legacy parameter for model type
            model_id: Optional model identifier
            kl_anneal_epochs: Legacy KL annealing parameter
            clip_norm: Gradient clipping norm
            weight_decay: Weight decay for optimizer
            **kwargs: Additional parameters
        """
        # Use config if provided, otherwise use legacy parameters
        if config is not None:
            model_type = config.model_type
            kl_anneal_epochs = config.kl_anneal_epochs
            weight_decay = config.weight_decay
            clip_norm = config.clip_norm
            # Update kwargs with config parameters
            config_dict = config.to_dict()
            kwargs.update(config_dict)
        
        model_type = model_type.lower()
        
        if model_type in ('gmm', 'hmm'):
            mm = MMMan()
            mm.fit(data, model_type=model_type, **kwargs)
            model = mm

        elif model_type == 'mmm':
            # build hybrid model using config or kwargs
            required_params = ['input_dim', 'hidden_dim', 'z_dim', 'rnn_hidden', 
                             'num_states', 'n_mix', 'trans_d_model', 'trans_nhead', 
                             'trans_layers', 'output_dim']
            
            # Check if all required parameters are available
            missing_params = [param for param in required_params if param not in kwargs]
            if missing_params:
                raise ValueError(f"Missing required parameters for MMM: {missing_params}")
            
            # Extract parameters
            input_dim = kwargs.pop('input_dim')
            hidden_dim = kwargs.pop('hidden_dim')
            z_dim = kwargs.pop('z_dim')
            rnn_hidden = kwargs.pop('rnn_hidden')
            num_states = kwargs.pop('num_states')
            n_mix = kwargs.pop('n_mix')
            trans_d_model = kwargs.pop('trans_d_model')
            trans_nhead = kwargs.pop('trans_nhead')
            trans_layers = kwargs.pop('trans_layers')
            output_dim = kwargs.pop('output_dim')
            
            # Optional parameters
            vocab_size = kwargs.pop('vocab_size', None)
            token_embedding = kwargs.pop('token_embedding', None)
            
            model = MultiMixtureTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                z_dim=z_dim,
                rnn_hidden=rnn_hidden,
                num_states=num_states,
                n_mix=n_mix,
                trans_d_model=trans_d_model,
                trans_nhead=trans_nhead,
                trans_layers=trans_layers,
                output_dim=output_dim,
                token_embedding=token_embedding,
                vocab_size=vocab_size
            )
            
            # Use config learning rate if available
            lr = config.learning_rate if config else kwargs.get('lr', 1e-4)
            epochs = kwargs.get('epochs', 100)
            
            optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            x = data.float().to(next(model.parameters()).device)

            for epoch in range(epochs):
                model.train()
                
                # Use enhanced training step with KL annealing
                if config:
                    step_result = model.training_step(
                        x, optim, epoch, epochs, 
                        anneal_type='linear'
                    )
                    total_loss = step_result['total_loss']
                    
                    # Log training progress
                    if epoch % max(1, epochs // 10) == 0:
                        print(f"Epoch {epoch}: loss={total_loss.item():.4f}, "
                              f"kl_weight={step_result['kl_weight']:.3f}")
                else:
                    # Legacy training
                    optim.zero_grad()
                    out = model(x, kwargs.get('tgt', None))

                    # Reconstruction loss: MSE
                    recon = out['reconstruction']
                    recon_loss = F.mse_loss(recon, x, reduction='sum')

                    # Clamp encoder logvars for KLD
                    mu, logvar = out['mu'], out['logvar']
                    logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
                    kld = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())

                    # HMM NLL with clamped likelihoods
                    hgmm_ll = out['hgmm_log_likelihood']
                    hgmm_ll = torch.clamp(hgmm_ll, min=-1e6, max=1e6)
                    hgmm_nll = -torch.sum(hgmm_ll)

                    # numeric safe
                    kld = torch.nan_to_num(kld, nan=0.0, posinf=1e8, neginf=-1e8)
                    hgmm_nll = torch.nan_to_num(hgmm_nll, nan=0.0, posinf=1e8, neginf=-1e8)

                    # Annealing weight
                    anneal_w = min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else 1.0
                    total_loss = recon_loss + anneal_w * (kld + hgmm_nll)

                    # Backprop & gradient clipping
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optim.step()

                    # Optional logging
                    if epoch % max(1, epochs // 5) == 0:
                        print(f"Epoch {epoch}: recon={recon_loss.item():.1f}, kld={kld.item():.1f}, "
                              f"hmll={hgmm_nll.item():.1f}, anneal_w={anneal_w:.2f}")
        else:
            raise ValueError("model_type must be 'gmm','hmm', or 'mmm'")

        assigned_id = self.add_model(model, model_id)
        return assigned_id

    def export_model(self, model_id: str, filepath: str = None):
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not found.")
        model = self.models[model_id]
        state = model.state_dict()
        if filepath:
            torch.save(state, filepath)
        return state

    def import_model(self, model_id: str, source):
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not found.")
        model = self.models[model_id]
        if isinstance(source, str):
            state = torch.load(source)
        elif isinstance(source, dict):
            state = source
        else:
            raise ValueError("source must be filepath or state dict")
        model.load_state_dict(state)

    def _select_data(self, mm, fn, data_ids=None, *args, **kwargs):
        all_keys = list(mm.check_data().keys())
        if data_ids is None:
            ids = all_keys
        elif isinstance(data_ids, (list, tuple)):
            ids = data_ids
        else:
            ids = [data_ids]
        out = {d: fn(mm, d, *args, **kwargs) for d in ids}
        if not isinstance(data_ids, (list, tuple)) and data_ids is not None:
            return out[data_ids]
        return out

    def get_means(self, model_id: str, data_ids=None):
        mm = self.get_mmm(model_id)
        return self._select_data(
            mm,
            lambda m, d: m._get_submodel(d).get_means(),
            data_ids
        )

    def get_variances(self, model_id: str, data_ids=None):
        mm = self.get_mmm(model_id)
        return self._select_data(
            mm,
            lambda m, d: m._get_submodel(d).get_variances(),
            data_ids
        )

    def get_weights(self, model_id: str, data_ids=None):
        mm = self.get_mmm(model_id)
        return self._select_data(
            mm,
            lambda m, d: m._get_submodel(d).get_weights(),
            data_ids
        )

    def get_log_likelihoods(self, model_id: str, X, data_ids=None):
        mm = self.get_mmm(model_id)

        def fn(m, d):
            sub = m._get_submodel(d)
            return sub.get_log_likelihoods(X)

        return self._select_data(mm, fn, data_ids)

    def score(self, model_id: str, X, data_ids=None):
        mm = self.get_mmm(model_id)

        def fn(m, d):
            sub = m._get_submodel(d)
            return sub.score(X)

        return self._select_data(mm, fn, data_ids)

    def get_mmm(self, model_id: str):
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not found.")
        return self.models[model_id]

    def assimilate(self, other_path: str):
        """
        Assimilate all models from another Cerebrum instance stored at `other_path`.
        Each submodel is exported to a temporary .pt file, imported into this instance,
        and the temporary file is removed.

        Returns:
            List[str]: The updated list of model IDs in this Cerebrum.
        """
        # Load the other Cerebrum
        other = Cerebrum.load(other_path)

        # Iterate through each model in the other Cerebrum
        for old_id, model in other.models.items():
            # Determine a new unique ID for this Cerebrum
            new_id = old_id if old_id not in self.models else self._generate_unique_id(old_id)

            # Deep-copy the architecture of the model into this Cerebrum
            model_copy = copy.deepcopy(model)
            self.models[new_id] = model_copy

            # Export state from the other model into a temporary file
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                tmp_path = tmp.name
            other.export_model(old_id, tmp_path)

            # Import the state into the newly added model
            self.import_model(new_id, tmp_path)

            # Clean up the temporary file
            os.remove(tmp_path)

        return list(self.models.keys())

    def save(self, path: str):
        torch.save(self, path)

    @classmethod
    def load(cls, path: str):
        return torch.load(path, weights_only=False)

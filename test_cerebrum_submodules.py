import torch
from modeling_cerebrum import GaussianMixture, HiddenMarkov, MM, MMMan

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Synthetic data parameters
N = 100  # samples
D = 4    # features
K = 3    # GMM components / HMM states
M = 2    # HMM mixtures

# Generate synthetic data for GMM (N, D)
gmm_data = torch.randn(N, D, device=device)

# Generate synthetic data for HMM (T, D)
T = 20  # sequence length
hmm_data = torch.randn(T, D, device=device)

# Generate batch of sequences for HMM (B, T, D)
B = 5
hmm_batch = torch.randn(B, T, D, device=device)

# --- Test GaussianMixture via MM wrapper ---
print('Testing GaussianMixture via MM...')
mm_gmm = MM(K, D, model_type='gmm').to(device)
mm_gmm.fit(gmm_data, lr=1e-2, epochs=10)
logp_gmm = mm_gmm(gmm_data)
print('GMM log-likelihoods:', logp_gmm[:5])
assert torch.isfinite(logp_gmm).all(), 'GMM log-likelihoods contain NaN/Inf'

# --- Test HiddenMarkov via MM wrapper ---
print('Testing HiddenMarkov via MM...')
mm_hmm = MM(K, D, model_type='hmm', n_mix=M).to(device)
mm_hmm.fit(hmm_data, lr=1e-2, epochs=10)
logp_hmm = mm_hmm(hmm_data)
print('HMM log-likelihood:', logp_hmm)
assert torch.isfinite(logp_hmm).all(), 'HMM log-likelihood contains NaN/Inf'

# Test batch scoring
logp_hmm_batch = mm_hmm(hmm_batch)
print('HMM batch log-likelihoods:', logp_hmm_batch)
assert torch.isfinite(logp_hmm_batch).all(), 'HMM batch log-likelihoods contain NaN/Inf'

# --- Test MM wrapper (redundant, but for completeness) ---
print('Testing MM (GMM mode)...')
mm_gmm2 = MM(K, D, model_type='gmm').to(device)
mm_gmm2.fit(gmm_data, lr=1e-2, epochs=10)
logp_mm_gmm = mm_gmm2(gmm_data)
print('MM-GMM log-likelihoods:', logp_mm_gmm[:5])
assert torch.isfinite(logp_mm_gmm).all(), 'MM-GMM log-likelihoods contain NaN/Inf'

print('Testing MM (HMM mode)...')
mm_hmm2 = MM(K, D, model_type='hmm', n_mix=M).to(device)
mm_hmm2.fit(hmm_data, lr=1e-2, epochs=10)
logp_mm_hmm = mm_hmm2(hmm_data)
print('MM-HMM log-likelihood:', logp_mm_hmm)
assert torch.isfinite(logp_mm_hmm).all(), 'MM-HMM log-likelihood contains NaN/Inf'

# --- Test MMMan manager ---
print('Testing MMMan...')
manager = MMMan()
gmm_id = manager.fit(gmm_data, model_type='gmm', n_components=K, n_features=D, epochs=10)
hmm_id = manager.fit(hmm_data, model_type='hmm', n_components=K, n_features=D, n_mix=M, epochs=10)

# Score using manager
logp_mgr_gmm = manager(gmm_data, data_id=gmm_id)
logp_mgr_hmm = manager(hmm_data, data_id=hmm_id)
print('MMMan GMM log-likelihoods:', logp_mgr_gmm[:5])
print('MMMan HMM log-likelihood:', logp_mgr_hmm)
assert torch.isfinite(logp_mgr_gmm).all(), 'MMMan GMM log-likelihoods contain NaN/Inf'
assert torch.isfinite(logp_mgr_hmm).all(), 'MMMan HMM log-likelihood contains NaN/Inf'

print('All Cerebrum submodule tests passed.') 
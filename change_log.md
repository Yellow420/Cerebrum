## Version: 0.1.3 – Submodule Stabilization
### Date: 2025-07-06
### Contributor: @kito (Architectural Specialist)

#### Summary:
- Stabilized and verified GMM, HMM, and MMMan submodules for robust PyTorch usage.

#### Changes:
- Clamped `log_vars` in GMM and HMM to prevent NaNs
- Added `.forward()` methods to all submodules for PyTorch compatibility
- Removed unsafe `.cpu().numpy()` calls (default to tensor output)
- Added `as_numpy=True` option to mean/var/weight accessors
- Improved naming and tensor safety in `fit()` and `score()`
- Verified stability using synthetic data
- Added `test_cerebrum_submodules.py` for reproducibility
- Ensured no NaN/Inf values during training and scoring
- Ready for evaluation and further architecture refinements
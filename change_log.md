**Changelog for Cerebrum — 2025-06-27**

![Version](https://img.shields.io/badge/version-0.1.0-red)

**Added**

* **KL Annealing System**

  * Integrated comprehensive KL annealing to prevent mode collapse in VAE components
  * Implemented three annealing schedules: linear, cosine, and step
  * Enhanced the training loop with detailed loss breakdown and progress metrics

* **Label Smoothing for Text Generation**

  * Extended `regression_loss` with a configurable label smoothing parameter (default: 0.1)
  * Improved output diversity and reduced repetition in Conversational-Cerebrum

* **Configuration System**

  * Introduced `CerebrumConfig` dataclass for unified model setup
  * Provided `CerebrumConfigs` presets for four specialized models:

    * **EEG-Cerebrum**: 64 channels, 16 states (seizure prediction)
    * **TTS-Cerebrum**: 80 mel features, 32 states (voice cloning)
    * **SpeakerRecognition-Cerebrum**: 40 MFCC features, 24 states (speaker identification)
    * **Conversational-Cerebrum**: 512 token dimensions, 64 states (text generation)

**Fixed**

* **Dependencies and Environment**

  * Corrected `requirements.txt` (removed invalid PyTorch version; aligned all packages for Python 3.13)
  * Added all necessary libraries for each Cerebrum type

* **Error Handling and Stability**

  * Added input validation across core functions
  * Implemented step-by-step error tracking to improve failure diagnostics

**Changed / Improved**

* **Initial Analysis and Requirements Review**

  * Audited the codebase to identify refactoring opportunities
  * Clarified requirements for specialized Cerebrum variants

* **Training Pipeline Enhancements**

  * Updated `fit_and_add` to utilize the new config system with automatic parameter validation
  * Enhanced logging to show fine-grained training progress

* **Code Cleanup and Documentation**

  * Removed test scaffolding and temporary files from the main repository
  * Added `USAGE_EXAMPLES.md` with detailed usage guides for all model types
  * Reorganized project structure to include only essential source files and documentation

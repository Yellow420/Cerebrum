**Changelog for Cerebrum — 2025-07-06**

![Version](https://img.shields.io/badge/version-0.1.2-red)


# Changelog

## Added

* **Custom `model_type` dispatch**
  At the top of `fit_and_add`, we now check:

  ```python
  if model_type in model_type_map:
      ModelClass = model_type_map[model_type]
      model = ModelClass(**kwargs)
  ```

  This allows arbitrary model classes to be registered and instantiated via a `model_type_map` (e.g. imported from a sub-module) before falling back to the legacy GMM/HMM/MMM logic.

* **`export_model` method**
  A new `export_model(self, model_id: str, filepath: str = None)` API to retrieve (and optionally save) a registered model’s `state_dict`.

* **`model_type_map` auto-augmentation with Transformers**
  In `sub_module/sub_mod.py`, after seeding `model_type_map` with any custom entries, we now attempt to import `transformers.CONFIG_MAPPING` and do:

  ```python
  for cfg_cls in CONFIG_MAPPING.values():
      model_type_map.setdefault(cfg_cls.model_type, cfg_cls)
  ```

  This automatically registers all supported Hugging Face model types (e.g. `"bert"`, `"gpt2"`, `"roberta"`) — without overwriting any custom entries.

* **Listing of registered types**
  After merging in Transformer entries, we print:

  ```python
  for key in sorted(model_type_map):
      print(f"{key!r} → {model_type_map[key].__name__}")
  ```

  This gives a quick console overview of which model types are available at runtime.

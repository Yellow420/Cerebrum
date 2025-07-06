# sub_module/sub_mod.py

# Initialize mapping with any custom models first (so transformer entries won't overwrite custom ones)
model_type_map = {}

# Example: Add your custom models here
# from my_submodule import MyCustomModel
# model_type_map['my_model_type'] = MyCustomModel

# Then, if available, import and merge Hugging Face Transformers model types
try:
    from transformers import CONFIG_MAPPING

    # Update mapping with transformer model types (won't overwrite custom entries)
    for config_class in CONFIG_MAPPING.values():
        model_type_map.setdefault(config_class.model_type, config_class)

    # print all registered model types
    for key in sorted(model_type_map):
        print(f"{key!r} → {model_type_map[key].__name__}")

except ImportError:
    # Transformers library not installed; skip adding transformer model types
    pass
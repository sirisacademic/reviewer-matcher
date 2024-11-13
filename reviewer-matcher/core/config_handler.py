import importlib

class ConfigManager:
    def __init__(self, config_module_name):
        config_module = importlib.import_module(config_module_name)
        self.config = self._load_configurations(config_module)

    def _load_configurations(self, module):
        return {attr: getattr(module, attr) for attr in dir(module) if attr.isupper()}

    def get(self, key, default=None):
        return self.config.get(key, default)


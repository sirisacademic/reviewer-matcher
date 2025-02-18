import importlib

class ConfigManager:
    def __init__(self, config_module_names):
        self.configs = {}
        for module_name in config_module_names:
            namespace = module_name.split('.')[-1]  # Use the last part of the module name as namespace
            self.configs[namespace] = importlib.import_module(module_name)
   
    def get(self, key, default=None, namespace=None):
        # Namespace-specific lookup
        if namespace:
            if namespace not in self.configs:
                raise ValueError(f"Namespace '{namespace}' does not exist.")
            return getattr(self.configs[namespace], key, default)
        # Global lookup across all namespaces
        found_value = None
        found_in_namespaces = []
        for namespace, config in self.configs.items():
            if hasattr(config, key):
                found_value = getattr(config, key)
                found_in_namespaces.append(namespace)
        # Check for ambiguous keys
        if len(found_in_namespaces) > 1:
            print(f"Warning: Key '{key}' found in multiple namespaces: {found_in_namespaces}. Using the first occurrence.")
        return found_value if found_value is not None else default

    def set(self, key, value, namespace=None):
        if namespace:
            if namespace not in self.configs:
                raise AttributeError(f"Namespace {namespace} does not exist.")
            setattr(self.configs[namespace], key, value)
        else:
            raise AttributeError("Namespace is required for setting configuration values.")

    def print_all_configs(self):
        for namespace, config in self.configs.items():
            print(f"\nNamespace: {namespace}")
            for key in dir(config):
                # Skip private or special attributes
                if not key.startswith("_"):
                    value = getattr(config, key)
                    print(f"  {key}: {value}")
                    
    def print_all_configs2(self):
        for namespace, config in self.configs.items():
            print(f"\nNamespace: {namespace}")
            for key in dir(config):
                # Skip private or special attributes
                if not key.startswith("_"):
                    value = getattr(config, key)
                    print(f"  {key}: {value}")
                    
                    

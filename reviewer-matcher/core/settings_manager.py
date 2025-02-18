import json

class SettingsManager:
    def __init__(self, settings_path):
        self.settings = self._load_settings(settings_path)

    def _load_settings(self, path):
        with open(path, 'r') as file:
            return json.load(file)

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = value


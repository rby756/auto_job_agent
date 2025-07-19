# auto_job_apply/config/config_loader.py
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from . import defaults

class ConfigLoader:
    """Configuration loader that supports YAML, JSON, and environment variables."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = {}
        self.config_file = config_file
        self._load_defaults()
        
    def _load_defaults(self):
        """Load default configurations."""
        # Load all uppercase variables from defaults
        for key in dir(defaults):
            if key.isupper():
                self.config[key] = getattr(defaults, key)
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from file if specified."""
        if not self.config_file:
            return self.config
            
        config_path = Path(self.config_file)
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}. Using defaults.")
            return self.config
            
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ('.yaml', '.yml'):
                file_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
        # Deep merge with defaults
        self._deep_merge(self.config, file_config)
        return self.config
        
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Recursively merge two dictionaries."""
        for key, value in update.items():
            if (key in base and isinstance(base[key], dict) 
                    and isinstance(value, dict)):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# Global configuration instance
_config_loader = ConfigLoader()
config = _config_loader.config

def get_config(key: str = None, default: Any = None) -> Any:
    """Get a configuration value or the entire config if no key is provided."""
    if key is None:
        return config
    keys = key.split('.')
    value = config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default
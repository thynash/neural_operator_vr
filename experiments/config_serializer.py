"""Configuration serializer for neural operator experiments."""

import json
import yaml
from pathlib import Path
from typing import Union

from experiments.config_schema import Config


def save_config(config: Config, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration object to save
        output_path: Path to output file (.yaml, .yml, or .json)
        
    Raises:
        ValueError: If output format is not supported
        IOError: If file cannot be written
    """
    output_path = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary
    config_dict = config.to_dict()
    
    # Write to file based on extension
    try:
        with open(output_path, 'w') as f:
            if output_path.suffix in ['.yaml', '.yml']:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                )
            elif output_path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(
                    f"Unsupported output format: {output_path.suffix}. "
                    f"Use .yaml, .yml, or .json"
                )
    except Exception as e:
        raise IOError(f"Failed to write configuration to {output_path}: {e}")

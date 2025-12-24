"""
Configuration loader for deepfake detection pipeline.

This module provides a centralized way to load and access configuration values
from config.yaml. All pipeline scripts should use this instead of hard-coding values.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


# Cache the loaded config
_config_cache = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary with configuration values
    """
    global _config_cache
    
    # Return cached config if already loaded
    if _config_cache is not None:
        return _config_cache
    
    # Find config file (check current dir and project root)
    config_file = Path(config_path)
    if not config_file.exists():
        # Try project root
        project_root = Path(__file__).parent.parent
        config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Cache and return
    _config_cache = config
    return config


def get_config() -> Dict[str, Any]:
    """
    Get the loaded configuration (loads if not already loaded).
    
    Returns:
        Configuration dictionary
    """
    return load_config()


def reload_config():
    """Reload configuration from disk (clears cache)."""
    global _config_cache
    _config_cache = None
    return load_config()


# Convenience accessors for common config values
class Config:
    """Configuration accessor with dot notation."""
    
    @staticmethod
    def _get_nested(keys: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        config = get_config()
        parts = keys.split('.')
        value = config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value
    
    # Model configuration
    @staticmethod
    def style_dim() -> int:
        """Get style dimension (always 100 for multi-stat)."""
        return Config._get_nested('model.style_dim', 100)
    
    @staticmethod
    def base_features() -> int:
        """Get number of base features (always 25)."""
        return Config._get_nested('model.base_features', 25)
    
    @staticmethod
    def normalize() -> bool:
        """Whether to normalize features (always True)."""
        return Config._get_nested('model.normalize', True)
    
    @staticmethod
    def patch_size() -> int:
        """Get patch size."""
        return Config._get_nested('model.patch_size', 512)
    
    @staticmethod
    def stride() -> int:
        """Get patch stride."""
        return Config._get_nested('model.stride', 512)
    
    @staticmethod
    def checkpoint() -> str:
        """Get default checkpoint path."""
        return Config._get_nested('model.checkpoint', 'checkpoints/pure_style_512.pt')
    
    # Training configuration
    @staticmethod
    def batch_size() -> int:
        """Get training batch size."""
        return Config._get_nested('training.batch_size', 128)
    
    @staticmethod
    def learning_rate() -> float:
        """Get learning rate."""
        return Config._get_nested('training.learning_rate', 0.001)
    
    @staticmethod
    def epochs() -> int:
        """Get training epochs."""
        return Config._get_nested('training.epochs', 30)
    
    @staticmethod
    def device() -> str:
        """Get device (auto, cuda, or cpu)."""
        return Config._get_nested('training.device', 'auto')
    
    # Paths
    @staticmethod
    def metadata_path() -> str:
        """Get metadata JSON path."""
        return Config._get_nested('paths.metadata', 'openfake-annotation/datasets/combined/metadata.json')
    
    @staticmethod
    def embeddings_cache() -> str:
        """Get embeddings cache path."""
        return Config._get_nested('paths.embeddings_cache', 'openfake-annotation/datasets/combined/cache/pure_style_embeddings.npz')
    
    @staticmethod
    def pair_frequencies() -> str:
        """Get pair frequencies JSON path."""
        return Config._get_nested('paths.pair_frequencies', 'feature_importance/pair_freq_norm.json')
    
    # Feature interaction
    @staticmethod
    def top_features() -> int:
        """Get number of top features for interaction analysis."""
        return Config._get_nested('feature_interaction.top_features', 10)
    
    @staticmethod
    def top_pairs() -> int:
        """Get number of top feature pairs to report."""
        return Config._get_nested('feature_interaction.top_pairs', 5)
    
    # LLM
    @staticmethod
    def llm_base_model() -> str:
        """Get LLM base model name."""
        return Config._get_nested('llm.base_model', 'Qwen/Qwen2.5-1.5B-Instruct')
    
    @staticmethod
    def lora_adapter() -> str:
        """Get LoRA adapter path."""
        return Config._get_nested('llm.lora_adapter', 'trained_qwen2.5_1.5b_feature_interpreter/model')
    
    # Limits
    @staticmethod
    def max_fake_samples() -> int:
        """Get max fake samples limit."""
        return Config._get_nested('limits.max_fake_samples', 1000)
    
    @staticmethod
    def max_real_samples() -> int:
        """Get max real samples limit."""
        return Config._get_nested('limits.max_real_samples', 1000)
    
    # Random seed
    @staticmethod
    def random_seed() -> int:
        """Get random seed for reproducibility."""
        return Config._get_nested('random_seed', 42)


# Module-level convenience functions
def get_device():
    """
    Get device for PyTorch operations.
    
    Returns:
        'cuda' if available and configured, else 'cpu'
    """
    import torch
    device_config = Config.device()
    
    if device_config == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_config == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            return 'cpu'
        return 'cuda'
    else:
        return 'cpu'


def ensure_multi_stat():
    """
    Verify that multi-stat pooling is configured.
    
    Raises:
        ValueError: If style_dim is not 100
    """
    style_dim = Config.style_dim()
    if style_dim != 100:
        raise ValueError(
            f"Pipeline requires style_dim=100 (multi-stat pooling), "
            f"but config has style_dim={style_dim}. "
            f"Please update config.yaml."
        )


def print_config_summary():
    """Print a summary of key configuration values."""
    print("CONFIGURATION SUMMARY")
    print(f"Model:")
    print(f"  Style Dimension: {Config.style_dim()}D (multi-stat pooling)")
    print(f"  Patch Size: {Config.patch_size()}x{Config.patch_size()}")
    print(f"  Stride: {Config.stride()}")
    print(f"  Normalize: {Config.normalize()}")
    print(f"  Checkpoint: {Config.checkpoint()}")
    print(f"\nTraining:")
    print(f"  Batch Size: {Config.batch_size()}")
    print(f"  Learning Rate: {Config.learning_rate()}")
    print(f"  Epochs: {Config.epochs()}")
    print(f"  Device: {Config.device()} (resolved: {get_device()})")
    print(f"\nPaths:")
    print(f"  Metadata: {Config.metadata_path()}")
    print(f"  Embeddings: {Config.embeddings_cache()}")
    print(f"\nLLM:")
    print(f"  Base Model: {Config.llm_base_model()}")
    print(f"  LoRA Adapter: {Config.lora_adapter()}")
    print(f"\nLimits:")
    print(f"  Max Fake Samples: {Config.max_fake_samples()}")
    print(f"  Max Real Samples: {Config.max_real_samples()}")
    print(f"\nRandom Seed: {Config.random_seed()}")


# Example usage
if __name__ == "__main__":
    # Load and display config
    print_config_summary()
    
    # Test access methods
    print("\nTesting config access:")
    print(f"  Config.style_dim() = {Config.style_dim()}")
    print(f"  Config.patch_size() = {Config.patch_size()}")
    print(f"  Config.checkpoint() = {Config.checkpoint()}")
    
    # Verify multi-stat
    try:
        ensure_multi_stat()
        print("\nMulti-stat pooling verified")
    except ValueError as e:
        print(f"\n Configuration error: {e}")


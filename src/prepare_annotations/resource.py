from pathlib import Path
from platformdirs import user_cache_dir
import os


def get_cache_dir() -> Path:
    """Get the root cache directory for all annotations."""
    env_cache = os.getenv("JUST_DNA_PIPELINES_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return Path(user_cache_dir(appname="just-dna-pipelines"))

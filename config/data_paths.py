"""
Simple helper to centralise the local data path for the project.
Code should import DATA_DIR and build file paths from it so users can override
the location by setting the DATA_DIR environment variable.

Example usage:
    from config.data_paths import DATA_DIR
    import os
    path = os.path.join(DATA_DIR, 'Client_1', 'local_train.csv')

This file is intentionally tiny and safe to keep in the repo.
"""
import os

DATA_DIR = os.getenv("DATA_DIR", "data")

def data_path(*segments):
    """Return an absolute path into the data directory for the given segments."""
    return os.path.join(DATA_DIR, *segments)

__all__ = ["DATA_DIR", "data_path"]

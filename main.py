#!/usr/bin/env python3
"""
Main entry point for Federated Learning with Rotation Aggregation
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from original_fl_rotation import main

if __name__ == "__main__":
    print("Starting Federated Learning with Rotation Aggregation...")
    main()
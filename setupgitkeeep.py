#!/usr/bin/env python3
"""
Create .gitkeep files in empty directories to preserve structure in git.

Usage:
    python setup_gitkeep.py
"""
from pathlib import Path

# Directories that should exist but will be empty in git
DIRECTORIES = [
    'data',
    'data/raw',
    'data/processed',
    'data/labeled',
    'models',
    'results',
    'logs',
    'artifacts',
    'artifacts/geocoding_cache',
    'artifacts/dspy_cache',
    'artifacts/metrics',
    'mlruns',
    'stage1-bert/data',
    'stage2/data',
    'stage3/data',
    'stage4/data',
    'annotations'
]

def create_gitkeep_files():
    """Create .gitkeep files in specified directories."""
    project_root = Path(__file__).parent
    
    created = []
    already_exists = []
    
    for dir_path in DIRECTORIES:
        full_path = project_root / dir_path
        gitkeep_path = full_path / '.gitkeep'
        
        # Create directory if it doesn't exist
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep if it doesn't exist
        if not gitkeep_path.exists():
            gitkeep_path.touch()
            created.append(dir_path)
        else:
            already_exists.append(dir_path)
    
    # Print results
    print("=" * 60)
    print("Git Directory Structure Setup")
    print("=" * 60)
    
    if created:
        print(f"\n✓ Created .gitkeep in {len(created)} directories:")
        for dir_path in created:
            print(f"  - {dir_path}/")
    
    if already_exists:
        print(f"\n✓ Already exists in {len(already_exists)} directories:")
        for dir_path in already_exists:
            print(f"  - {dir_path}/")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review the created directories")
    print("  2. git add */.gitkeep")
    print("  3. git commit -m 'Add project directory structure'")
    print("=" * 60)

if __name__ == '__main__':
    create_gitkeep_files()
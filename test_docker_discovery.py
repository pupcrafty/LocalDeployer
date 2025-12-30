#!/usr/bin/env python3
"""Test Docker discovery for diamondDrip"""
from pathlib import Path
from docker_deployment import get_docker_manager

workspace = Path("D:/Workspace")
dm = get_docker_manager(workspace)

print("Testing Docker discovery for diamondDrip...")
result = dm.discover_build_instructions("diamondDrip")

if result:
    print(f"\n[SUCCESS] Found instructions for project: {result.get('project')}")
    print(f"  AWS folder: {result.get('aws_folder')}")
    print(f"\n  Files found: {len(result.get('instructions', {}))}")
    for rel_path, file_data in result.get('instructions', {}).items():
        print(f"    - {rel_path} ({file_data.get('type')})")
        print(f"      Path: {file_data.get('path')}")
else:
    print("\n[FAILED] No instructions found")
    print("\nChecking if aws folder exists...")
    aws_folder = workspace / "diamondDrip" / "aws"
    print(f"  AWS folder exists: {aws_folder.exists()}")
    if aws_folder.exists():
        print(f"  Files in aws folder:")
        for f in aws_folder.iterdir():
            if f.is_file():
                print(f"    - {f.name}")


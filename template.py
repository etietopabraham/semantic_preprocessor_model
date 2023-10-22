"""
Template.py

Purpose:
    Automates the creation of project folders and files for this project.
    This script will generate a predefined project structure with folders and files.
    If a specified file already exists, it won't be overwritten.

Usage:
    Simply run this script in the desired location to scaffold the project structure.
    `python template.py`

Dependencies:
    - os, pathlib, logging
"""

# Required libraries
import os
from pathlib import Path
import logging

# Set up logging to display activities and any potential issues.
# It logs the creation of directories and files.
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the root project name. This will be used to name primary project directories.
project_name = "semantic_preprocessor_model"

# Define the project structure.
# Each entry in this list represents either a directory or a file.
# Directories will be created first, followed by files.
list_of_files = [
    # GitHub workflows directory
    ".github/workflows/.gitkeep",
    # Source directory with various sub-directories for modular organization
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    # Config files at the project root
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    # Primary application files
    "main.py",
    "app.py",
    # Docker and requirements for deployment and environment setup
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    # Research directory for exploratory work
    "research/trials.ipynb",
    # Templates directory for any HTML templates
    "templates/index.html",
]

# Iterate through the list and create the folders and files
for file_path_str in list_of_files:
    file_path = Path(file_path_str)
    
    # If there's a parent directory (i.e., it's not at the root), create it.
    if file_path.parent and not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {file_path.parent}")

    # Create the file if it doesn't exist or if it exists but is empty.
    if not file_path.exists() or file_path.stat().st_size == 0:
        file_path.touch()
        logging.info(f"Creating empty file: {file_path}")
    else:
        logging.info(f"{file_path.name} already exists")
"""Setup script for EDM Cue Analyzer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')

setup(
    name="edm-cue-analyzer",
    version="1.0.0",
    author="EDM Cue Analyzer Project",
    description="Automated cue point generation for DJ performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/edm-cue-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'edm-cue-analyzer=edm_cue_analyzer.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'edm_cue_analyzer': ['../default_config.yaml'],
    },
    zip_safe=False,
)

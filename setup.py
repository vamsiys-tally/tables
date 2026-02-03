"""Setup configuration for the tables package."""

from setuptools import setup, find_packages
from pathlib import Path


def read_requirements():
    """Read requirements from requirements.txt."""
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path) as f:
            # Filter out comments and empty lines
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="tables",
    version="0.1.0",
    description="Bank Statement PDF Processing System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "ml": [
            "sentence-transformers>=2.2.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tables=tables.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

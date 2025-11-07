"""
Setup configuration for moonboard_core package.

This package contains shared utilities for processing Moonboard climbing problems.
"""

from setuptools import setup, find_packages

setup(
    name="moonboard_core",
    version="0.1.0",
    description="Shared utilities for Moonboard problem processing",
    author="Moonboard Grader Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)


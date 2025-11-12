from setuptools import setup, find_packages

setup(
    name="moonboard-generator",
    version="0.1.0",
    description="Moonboard climbing problem generator using Conditional VAE",
    author="Moonboard Grader Team",
    python_requires=">=3.8",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "tensorboard>=2.14.0",
        ],
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


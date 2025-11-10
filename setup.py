from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mtgnn",
    version="0.1.0",
    author="MTGNN Authors",
    description="Multivariate Time Series Forecasting with Graph Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nnzhan/MTGNN",
    packages=["mtgnn"],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.2.0",
        "numpy>=1.17.0",
        "scipy>=1.4.0",
        "pandas>=0.25.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.1.0",
        "safetensors>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
)

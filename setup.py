"""
Setup script for Quantum Network Simulator
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-network-simulator",
    version="1.0.0",
    author="Quantum Network Research Team",
    description="Distributed RL for temporal-spectral resource allocation in quantum networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-network-simulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "torch>=2.2.2",
        "networkx>=3.1",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.3",
        "matplotlib>=3.7.3",
        "pandas>=2.0.3",
        "tqdm>=4.66.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "mypy>=1.7.1",
        ],
        "quantum": [
            "qutip>=4.7.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "qns-simulate=run_simulation:main",
            "qns-server=backend.main:main",
        ],
    },
)

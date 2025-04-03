from setuptools import setup, find_packages
setup(
    name="iscm",
    version="0.0.1",
    description="A data generating package implemented as part of 'Standardizing Structural Causal Models'",
    author="Weronika Ormaniec",
    author_email='wormaniec@ethz.ch',
    url="https://github.com/werkaaa/iscm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "setuptools>=61.0",
        "numpy>=1.24.4",
        "igraph>=0.11.3",
        "networkx>=2.8.2",
    ]
)
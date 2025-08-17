from setuptools import setup, find_packages

setup(
    name="IncunabulaML",
    version="0.1.0",
    description="Foundational machine learning algorithms implemented from scratch in Python.",
    author="Harshdeep Sharma",
    author_email="harsh7251909511@gmail.com", 
    url="https://github.com/har5hdeep5harma/IncunabulaML", 
    packages=find_packages(),
    install_requires=[
        "numpy", 
        "scipy",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Intended Audience :: Education",
        "Topic :: Artificial Intelligence",
    ],
)

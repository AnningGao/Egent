from setuptools import setup, find_packages

setup(
    name="egent",
    version="0.1.0",
    description="A Python package",
    author="Yuan-Sen Ting",
    author_email="ting.74@osu.edu",
    url="https://github.com/tingyuansen/Egent",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Add your dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

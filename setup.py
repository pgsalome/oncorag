from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="oncorag2",
    version="0.1.0",
    author="Pgsalome",
    author_email="pgsalome@gmail.com",
    description="Oncology Report Analysis with Generative AI - Version 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgsalome/oncorag2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "oncorag2-generate=oncorag2.feature_extraction.agent:main",
        ],
    },
    include_package_data=True,
)
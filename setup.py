from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.2",
    author="hemangdhanani",
    description="Donor Choose problem solutions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hemangdhanani",
    author_email="hemangdhanani@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "nltk",
        "seaborn",
        "sklearn"
    ]
)
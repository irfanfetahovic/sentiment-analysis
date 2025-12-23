from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Find packages in both src and root
sentiment_packages = find_packages(where="src")
script_packages = find_packages(include=["scripts"])

setup(
    name="sentiment-analysis",
    version="1.0.0",
    author="Irfan Fetahovic",
    author_email="irfan.fetahovic@example.com",
    description="Production-ready sentiment analysis using classical NLP and Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/irfanfetahovic/sentiment-analysis",
    packages=sentiment_packages + script_packages,
    package_dir={"sentiment_analysis": "src/sentiment_analysis"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentiment-train-classical=scripts.run_train_classical:main",
            "sentiment-train-transformer=scripts.run_train_transformer:main",
            "sentiment-evaluate=scripts.run_evaluate:main",
            "sentiment-predict=scripts.run_predict:main",
            "sentiment-pipeline=scripts.run_pipeline:main",
        ],
    },
)

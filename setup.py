from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="medical-search-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Medical Disease Name Search System using RAG with medical embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medical-search-system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.0.267",
        "langchain-huggingface>=0.0.6",
        "langchain-openai>=0.0.2",
        "transformers>=4.30.2",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "pydantic>=1.10.8",
        "openai>=0.27.8",
        "streamlit>=1.24.0",
        "pandas>=2.0.2",
        "scikit-learn>=1.2.2",
        "torch>=2.0.1",
        "tqdm>=4.65.0",
        "python-Levenshtein>=0.21.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "gpu": ["faiss-gpu>=1.7.4"],
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-search=app:main",
        ],
    },
)
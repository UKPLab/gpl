from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="gpl",
    version="0.1.1",
    author="Kexin Wang",
    author_email="kexin.wang.2049@gmail.com",
    description="GPL is an unsupervised domain adaptation method for training dense retrievers. It is based on query generation and pseudo labeling with powerful cross-encoders. To train a domain-adapted model, it needs only the unlabeled target corpus and can achieve significant improvement over zero-shot models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/UKPLab/gpl",
    project_urls={
        "Bug Tracker": "https://github.com/UKPLab/gpl/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        'beir',
        'easy-elasticsearch>=0.0.7'
    ],
)
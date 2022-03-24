import setuptools

REQUIRED_PKGS = [
    "numpy>=1.21.1",
    "allennlp>=2.8.0",
    "transformers>=4.15.0",
    "datasets>=1.14.0",
    "eaas>=0.1.9",
    "tqdm>=4.62.3"
    ]



setuptools.setup(
    name="omneval",
    version="0.0.1",
    author="Ziyun Xu(vincent1rookie)",
    author_email="vincentinfdu@hotmail.com",
    description="A zero-shot evaluation benchmark to assess innate ability of PLMs",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ExpressAI/omneval",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PKGS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'omneval = main:main'
        ]}
)

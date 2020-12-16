import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vietocr",
    version="0.3.4",
    description="Transformer base text recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'einops==0.2.0',
        'gdown==3.11.0',
        'prefetch_generator==1.0.1',
        'imgaug==0.4.0',
        'lmdb==1.0.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

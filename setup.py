import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="verdict",
    version="1.2",
    author="Zach Hafen",
    author_email="zachary.h.hafen@gmail.com",
    description="A class for flexible manipulation of nested data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhafen/verdict",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'h5py>=3.7.0',
        'h5sparse>=0.1.0',
        'mock>=4.0.3',
        'numpy>=1.23.5',
        'pandas>=1.5.2',
        'scipy>=1.9.3',
        'setuptools>=65.6.3',
        'six>=1.16.0',
        'tqdm>=4.64.1',
    ],
    py_modules=[ 'verdict', 'test_verdict' ],
)

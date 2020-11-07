import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="verdict",
    version="1.1.4",
    author="Zach Hafen",
    author_email="zachary.h.hafen@gmail.com",
    description="A version of Python's dictionary with additional features",
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
        'pandas>=0.20.3',
        'mock>=2.0.0',
        'numpy>=1.14.5',
        'six>=1.10.0',
        'setuptools>=28.8.0',
        'h5py>=2.7.0',
    ],
    py_modules=[ 'verdict', 'test_verdict' ],
)

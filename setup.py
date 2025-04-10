from setuptools import setup, find_packages

setup(
    name='drnet',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.0',
    ],
    author='Syam Gowtham Geddam',
    author_email='geddamgowtham4@gmail.com',
    description='A parameterized DRNet implementation for feature extraction and image processing.', 
    url='https://github.com/nameishyam/code-implementation',
)
from setuptools import setup,find_packages
from datetime import datetime

install_requires = ['numpy', 'torch','torchvision']

setup(
    name="maglab",
    version=0.1,
    description ="Python package for electron microscopy simulation.",
    author="BoyaoLyu",
    author_email="lvboyao@mail.ustc.edu.cn",
    packages=find_packages(include=['maglab', 'maglab.*']),
    install_requires=install_requires,
    python_requires='>=3',    
)
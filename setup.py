from setuptools import setup,find_packages
from datetime import datetime

install_requires = ['numpy', 'torch','torchvision']

filename = 'maglab/__init__.py'
with open(filename, 'r') as file:
    lines = file.readlines()

if lines[-1].startswith('print("install on: '):
    with open(filename, 'w') as file:
        lines[-1] = f'print("install on: {datetime.now()}")\n' 
        file.writelines(lines)
else:
    raise ValueError("file format problem")


setup(
    name="maglab",
    version=0.0,
    description ="Python package for electron microscopy simulation.",
    author="BoyaoLyu",
    author_email="lvboyao@mail.ustc.edu.cn",
    packages=find_packages(include=['maglab', 'maglab.*']),
    install_requires=install_requires,
    python_requires='>=3',    
)
from setuptools import find_packages
from setuptools import setup

setup(
    name='omni_bringup',
    version='0.0.0',
    packages=find_packages(
        include=('omni_bringup', 'omni_bringup.*')),
)

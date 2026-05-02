from setuptools import find_packages
from setuptools import setup

setup(
    name='omni_dofbot_bringup',
    version='0.0.0',
    packages=find_packages(
        include=('omni_dofbot_bringup', 'omni_dofbot_bringup.*')),
)

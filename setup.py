# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='simpy',
    version='0.1.0',
    description='Automate ODE solving for biology',
    long_description=readme,
    author='Haydn King',
    author_email='hjking734@gmail.com',
    url='https://github.com/haydnKing/pysim',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

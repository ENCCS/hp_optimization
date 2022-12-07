# coding: utf-8

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='hpo_workshop',
      version='0.1',
      description='Project to train QSAR models',
      url='https://github.com/eryl/hpo_workshop',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['hpo_workshop'],
      install_requires=[],
      dependency_links=[],
      zip_safe=False)

from setuptools import setup, find_packages
from subprocess import check_output
import os

from ilustrado import __version__

with open('requirements/requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f.readlines()]

# have to strip git link for setup.py/pip install
requirements = [req for req in requirements if not req.startswith('git')]

extra_requirements = dict()
for subreq in ['docs', 'voronoi', 'test']:
    with open('requirements/{}_requirements.txt'.format(subreq), 'r') as f:
        extra_requirements[subreq] = [line.strip() for line in f.readlines()]

setup(name='ilustrado',
      version=__version__,
      description='Simple genetic algorithm for crystal structure prediction.',
      long_description=open('README.rst').read(),
      author='Matthew Evans',
      author_email='me388@cam.ac.uk',
      license='MIT',
      packages=find_packages(),
      test_suite='ilustrado.tests',
      install_requires=requirements,
      extras_require=extra_requirements,
      zip_safe=False)

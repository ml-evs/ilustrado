from setuptools import setup, find_packages
from subprocess import check_output

try:
    __version__ = check_output(["git", "describe", "--tags"]).decode("utf-8").strip()
    __version__ += '+' + (check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
                          .decode('utf-8').strip())
except:
    __version__ = 'xxx'

setup(name='ilustrado',
      version=__version__,
      description='Simple genetic algorithm for crystal structure prediction.',
      long_description=open('README.md').read(),
      author='Matthew Evans',
      author_email='me388@cam.ac.uk',
      license='MIT',
      packages=find_packages(),
      test_suite='ilustrado.tests',
      install_requires=[
          'numpy>=1.10',
          'scipy>=0.18',
          # 'matador', requires a local install of matador
          'scikit-learn>=0.18',
          'matplotlib==1.5',
          'seaborn'
      ],
      extras_require={
          'matador': ['matador']
      },
      zip_safe=False)

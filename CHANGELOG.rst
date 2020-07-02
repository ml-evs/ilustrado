v0.4
====

- Updated matador dependency PyPI versions > 0.9.
- Added ``compute_mode="manual"`` to just generate structures without relaxing
- Added GitHub CI
- Rescale cell volume after crossover to maintain density
- QoL updates to printing and loading mutations
- Allow arbitrary ASE calculator to be used in relaxations

v0.3
====

- New keyword: ``sandbagging``. When enabled, fitness penalties will be applied to successive sampling of the same region of composition space. By default, the modifier is a multiplicative factor of 0.95 to all compositions within a hypersphere of radius 0.05.
- ``compute_mode='slurm'`` that makes use of array jobs for "infinite" horizontal scalability
- improved documentation and examples

.. |MIT License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/ml-evs/ilustrado/blob/master/LICENSE
.. |Documentation Status| image:: https://readthedocs.org/projects/ilustrado/badge/?version=latest
   :target: https://ilustrado.readthedocs.io/en/latest/?badge=latest

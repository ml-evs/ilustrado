# `ilustrado`

## Summary

`ilustrado` is a Python package that implements a highly-customisable massively-parallel genetic algorithm for *ab initio* crystal structure prediction (CSP), with a focus on mapping out compositional phase diagrams. The aim of `ilustrado` was to provide a method of extending CSP results generated with random structure searching (AIRSS) or extrapolating from known chemically-relevant systems (species-swapping/prototyping).

The API is [fully-documented](http://www.tcm.phy.cam.ac.uk/~me388/ilustrado) and the source code can be found on [BitBucket](https://bitbucket.org/me388/ilustrado). `ilustrado` makes extensive use of the [matador](https://tcm.phy.cam.ac.uk/~me388/matador) API and interfaces with [CASTEP](http://www.castep.org/) for DFT-level relaxations. Written by [Matthew Evans](http://www.tcm.phy.cam.ac.uk/~me388) (me388@cam.ac.uk).

By default, fitnesses are evaluated as the distance from a binary or ternary convex hull that is passed as input. Duplicate structures are filtered out post-relaxation based on pair distribution function overlap. The standard mutation operators are implemented (cell and position noise, vacancy, atom permutation/transmutation) and additionally a Voronoi-based mutation, whereby one elemental sublattice is removed and reinstated (with a randomly modified number of lattice points) at the Voronoi points of the remaining overall crystal. Crossover is performed with the standard cut-and-splice method to ensure transferrability over many types of material systems. Several physical constraints (minimum atomic separations, densities, cell shapes) are applied to the trial structures before relaxation to improve efficiency. In order to maintain population diversity as the simulation progresses, the user can optionally disfavour frequently-visited regions of composition space. The entrypoint is a Python script that creates an `ArtificialSelector` object with the desired parameters (documented [here](http://www.tcm.phy.cam.ac.uk/~me388/ilustrado/ilustrado.html#ilustrado.ilustrado.ArtificialSelector)). Many examples can be found in `examples/`.

There are two `compute_mode`s in which to run `ilustrado` (examples of both can be found in `examples/`):
- `compute_mode='direct'` involves one `ilustrado` processes spawning `mpirun` calls either on local or remote partitions (i.e. either a node list is passed for running on a local cluster, or `ilustrado` itself is submitted as a HPC job). In this case, the user must manually restart the GA when the job finishes.
- `compute_mode='slurm'` performs the GA in interruptible steps; submitting `ilustrado` as a job will lead to the submission of many slurm array jobs for the relaxation, and a dependency job that re-runs the `ilustrado` process to check on the relaxations. In this case, the user only needs to submit one job.

## New in v0.3b:

- `sandbagging` of composition space
- `compute_mode='slurm'` that makes use of array jobs for "infinite" horizontal scalability
- improved documentation and examples

## API Docs

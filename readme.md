## Machine Learning Guide for University of Aberdeen HPC Clusters

This repository provides information and examples for using the University of Aberdeen HPC clusters Maxwell and Macleod. See the accompanying presentation of this guide `Maxwell-HPC-ML-Intro.pdf`.

This guide is focused on GPU accelerated Machine and Deep Learning with [PyTorch](https://pytorch.org/) and does not include instructions for CPU and high-memory computation. We assume users of this guide are experienced in producing Pytorch codes and models, as well as GPU-accelerated training.

**NOTE:** HPC services are **NOT** intended as an environment for the development codes or the hosting of model inference, and visualisation dashboards.

### Macleod
The Macleod cluster is the University's own in-house HPC specifically dedicated to teaching and student research projects. All staff and students can access the cluster, staff can [request access here](https://forms.office.com/r/Fzukzia80T).

The `macleod/` directory of this repo contains a written introduction guide and example code to get started on using Macleod for GPU accelerated Machine Learning.

**Note:** Macleod should not be used for large-scale computing projects. If you require more GPU resources consider Maxwell if you have access.

### Maxwell
The Maxwell cluster is the University's own in-house HPC specifically dedicated to computationally expensive research projects. All staff can access the cluster, staff can request access by contacting [digital research](digitalresearch@abdn.ac.uk).

The `maxwell/` directory of this repo contains a written introduction guide and example code to get started on using Maxwell for GPU accelerated Machine Learning.

### Todo:

- [ ] Add screenshots
- [ ] Distributed computation guide

### Official Resources

Full Aberdeen HPC documentation:
https://www.abdn.ac.uk/it/documents-uni-only/OCF-User0-Manual-Abderdeen-Maxwell.pdf

Full SLURM documentation:
https://slurm.schedmd.com/documentation.html

More details regarding SLURM can be found at:
https://slurm.schedmd.com/tutorials.html

Conda documentation:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

### Errors, Issues, and Contributions

If you would like to contribute to these guides please fork the repository and make pull requests to propose your changes to the project. Alternatively, email specific changes to aiden.durrant@abdn.ac.uk for updates.

Errors can be raised through the GitHub issues or again emailed to aiden.durrant@abdn.ac.uk .

#### Author
Aiden Durrant

Department of Computing Science, School of Natural and Computing Science

aiden.durrant@abdn.ac.uk
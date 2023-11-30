## Maxwell Cluster Introduction Guide -
**v1.0**

A step-by-step introduction guide to running PyTorch GPU accelerated programs on Maxwell HPC.

## 1. Technical specifications

Maxwell is a Linux supercomputing cluster housed in the Edward Wright Datacentre and provides:
- 120 CPU cores and 1.2TB of RAM - Maximum 256GB per node
- A100 nodes: Four nodes each with 3x A100 GPU 80GB cards
- Comercial GPU nodes: Two GPU nodes each with 2x 2080ti 12GB cards.
- High-speed network – a 10Gb/s network

There currently exist two separate partitions of the A100 GPU nodes:
- `a100_full`: 3 nodes where each A100 GPU memory is not partitioned and remains full at 80Gb per card.
- `a100_mig`: 1 node where each A100 GPU is partitioned into 2, resulting in 6 GPU partitions each with 40GB.

It is important to note that you cannot employ different GPU partitions in a distributed fashion, hence the current maximum distributed setting is 3x `a100_full` nodes with 9x A100 80GB GPU's in total.

## 2. Accessing the cluster

Requesting access to Maxwell is done by directly contacting [digital research](digitalresearch@abdn.ac.uk), and is only available for Staff, PGR, and those who purchase resources. For UG and PGT students interested in HPC resources, see the Maxwell cluster guide.

### 2.1 Remote Access

Once access is granted, you can connect to Maxwell via SSH when connected to the University network (excluding eduroam). To access the university network remotely the f5 VPN provides seamless connection for managed devices. For personal devices, the Web VPN can be employed by following https://remote.abdn.ac.uk/ .

For more details regarding remote access, see https://www.abdn.ac.uk/staffnet/working-here/it-services/remote-access.php

### 2.2 SSH and SFTP

Any SSH client can be employed to connect to Maxwell, where PuTTy can be used on managed devices. Simply connect via the listed hostnames and ports below to access the login nodes for each HPC service. The username is your university username e.g. s01ab23.

Maxwell login nodes:
- Maxlogin1.abdn.ac.uk : port 22
- Maxlogin2.abdn.ac.uk : port 22

Transferring data can be achieved by SFTP via the same settings as previously outlined. Again, any SFTP client can be used, however, university-managed devices can use WinSCP.

**Note:** It is recommended that git version management tools are employed for managing code on the HPC services.

If you run into any issues at this stage contact servicedesk@abdn.ac.uk  

## 3. Data Storage

Once you are logged into Maxwell you will find yourself in the home directory `/uoa/home/<username>`.

The storage on both Maxwell and Macleod is split into two partitions `home/` and `sharedscratch/`. Home is accessible through the above path, and shared scratch can be accessed as a link through the home directory (`/uoa/home/<username>/sharedscratch/`) or through `/uoa/scratch/users/<username>/`.

The following defines the storage specifications:

| Storage    | HPC Specification |
| -------- | ------- |
| `home`   | <ul><li>50GB</li><li>Daily and Weekly Backups</li></ul> |
| `sharedscratch`   | <ul><li>1TB</li><li>No Backup</li></ul> |

Mode Details at (technical-specification-hpc-for-research)[https://uoa.freshservice.com/support/solutions/articles/50000135034-technical-specification-hpc-for-research-maxwell-]

We recommend you store all working code and data in the `sharedscratch` partition and only use `home` directory storage only when absolutely necessary.

## 4. Getting your data on Maxwell

It is recommended that code management is handled through version control software such as Git and hosted in cloud repositories such as GitHub. This provides an effective solution for managing code across multiple devices including Maxwell.

For data files (i.e. datasets, logs, etc.) any SFTP client can be employed when connected to the university network for uploading and downloading data from Maxwell.

`sftp <username>@maxlogin1.abdn.ac.uk -p 22`

For a graphical interface, SCP is recommended by IT services and can be installed on all managed devices through the software centre.

## 5. Initial setup, conda environment

Now you have connected to Maxwell you must access your required packages, if your packages are not present in the software list (https://www.abdn.ac.uk/it/documents-uni-only/Maxwell-Galaxy-Software.pdf ), then the easiest way to download them is to create an anaconda environment.

First load anaconda via:
`module load miniconda3`

Now you can create an environment using the following:
`conda create -n <your_env_name> python=3.11`

**Note:** For first-time installs you may be prompted to configure your shell, if you receive this error run the command `conda init bash` and then exit/close the terminal, ssh back into Maxwell and load the miniconda3 module to continue.

Once created, activate the environment
`conda activate <your_env_name>`

You should now see the `(<your_env_name>)` in the bash shell.

Now you can go ahead and install all your desired packages.
For this example, we will install the following (Note: when installing PyTorch use the current CUDA version installed on maxwell, currently 11.7):
`conda install PyTorch torchvision torchaudio cudatoolkit=11.7 -c PyTorch`
`conda install -c conda-forge matplotlib`
`conda install -c conda-forge tqdm`

Now you have a virtual environment created and all your packages installed, you can now use this environment every time you use Maxwell. Remember you must load the module anaconda3 before you activate your environment.

## 6. Intro to SLURM and running your code

Unlike your standard desktop computer, you must submit a “job” for execution to a scheduler. Here the Maxwell cluster uses SLURM workload manager to schedule the allocation of resources for jobs.

Some useful commands are as follows:
`sinfo` - Show summary information
`squeue` - Show job queue
`scontrol show partition gpu` - Show GPU partition details
`chkquota` - Show hard drive allocation used and free space

To run a job (run your program) on Maxwell you must submit a script to the SLURM scheduler. The SLURM script must contain 3 things:
1. Define the resource requirements for the job
2. Activate the environment we created earlier
3. Specify the script we wish to run

An example script `run_example.sh` for this introduction guide is given here:

```
#!/bin/bash
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=12 # number of cores
#SBATCH --mem=64G # memory pool for all cores

#SBATCH --ntasks-per-node=1 # one job per node
#SBATCH --gres=gpu:1 # 1 GPU out of 3
#SBATCH --partition=a100_full

#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR

#SBATCH --mail-type=ALL
#SBATCH --mail-user=<username>@abdn.ac.uk

module load miniconda3
source activate <your_env_name>

srun python example_script.py --epochs=25 --save /home/<username>/sharedscratch/
```

This script specifies we wish to use 1 GPU node, 1 of the 3 available GPU's on that node, 64GB of RAM, and 12 CPU cores.

When specifying the partition:
- For MIG GPU's users should use: `#SBATCH --partition=a100_mig` (see the following section)
- For standard/full a100 GPU's users should use: `#SBATCH --partition=a100_full`

To submit the above job script named `run_example.sh` to the Slurm scheduler we use the `sbatch` command:
`sbatch run_example.sh`

The job may run immediately or may take up to a few days to run depending on the number of existing jobs, you will receive an email notification if you use the SLURM command `email`.

To check the status of queued and running jobs, use the following:
`squeue -u <username>`

If you wish to cancel a job simply use:
`scancel <job_ID>`

### 6.2 MIG GPU Partition

Multi-Instance GPU (MIG) is a feature of Nvidia graphics cards that enables a single GPU to be virtually partitioned into separate GPU instances to provide users with a greater number of GPU instances for optimal utilization.

The MIG-enabled GPU are placed on the `a100_mig` node, which contains 3x 80Gb A100 GPUs, which are then virtualised into 6 x 40GB instances.

To access these resources simply utilise the following settings in the SLURM script. All other settings remain the same, and the implementation of PyTorch code does not change.

```
#SBATCH --gres=gpu:2 # 2 GPU instances out of 6
#SBATCH --partition=a100_mig
```

**Important!** The MIG node can only be used in isolation, and cannot be configured in a distributed manner with the other A100 nodes.

For more information on MIG see [here](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html)

## 7. Distributed Computation

To fully leverage the GPU resources available on Maxwell, it is essential code is correctly implemented with distributed computation in mind. Although you can request large computational resources and SLURM will allocate them to you, it is up to the user to ensure their codes utilize these resources.

By default PyTorch will not offload any computation to the GPUs, this has to be manually specified. Additional steps also need to be taken to enusre that when multiple GPUs are requested these are correctly employed.

There are 4 sensible configurations considered:
- For 1 node (#SBATCH --nodes=1):
  - No GPU training
  - Single GPU training (#SBATCH --gres=gpu:1) - [CUDA offload](https://PyTorch.org/docs/stable/notes/cuda.html#best-practices)
  - Multiple GPU training (#SBATCH --gres=gpu: > 1) - [Data Parallel](https://PyTorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)
- Multiple nodes (#SBATCH --nodes= > 1):
  - Multiple GPU training (#SBATCH --gres=gpu: > 1)- [Distributed Data Parallel](https://PyTorch.org/tutorials/intermediate/ddp_tutorial.html)

We advise users to follow the linked PyTorch documentation for each of these settings. Additionally, we provide an example PyTorch code in this repository to help you get started.

**Note:** We advise only experienced users with strong distributed computational knowledge to request the multiple node, multiple GPU setting given its more strenuous implementation and key adjustment to individual codes.

## 8. Example Code

Single node, single GPU examples can be found `abdn-hpc/maxwell/single_gpu/`

Single node, multiple GPU examples can be found `abdn-hpc/maxwell/multi_gpu/`

Multiple node, multiple GPU examples can be found `abdn-hpc/maxwell/multi_node_multi_gpu/`

## 9. Miscellaneous
### 9.1 Wall Clock Time
Wall clock time is the duration for which the nodes remain allocated to you.

All jobs have a maximum wall time of 24 hours, which is applied by default unless specified otherwise.

If your job requires computation that is longer than 24 hours, you must handle it
within the code/script. Typically this is done via "checkpointing", for more details see [Saving and Loading PyTorch Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

## Additional Resources

Full Aberdeen HPC documentation:
https://www.abdn.ac.uk/it/documents-uni-only/OCF-User0-Manual-Abderdeen-Maxwell.pdf

Full SLURM documentation:
https://slurm.schedmd.com/documentation.html

More details regarding SLURM can be found at:
https://slurm.schedmd.com/tutorials.html

Conda documentation:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
## Maxwell Cluster Introduction Guide -
**v0.1**


A step-by-step introduction guide to running PyTorch GPU accelerated programs on Macleod HPC.

### 1. Technical specifications

Macleod is a Linux supercomputing cluster housed in the Edward Wright Datacentre and provides:
- 120 CPU cores and 1.2TB of RAM - Maximum 256GB per node
- A100 nodes: Three nodes each with 3x A100 GPU 80GB cards
- Comercial GPU nodes: Two GPU nodes each with 2x 2080ti 12GB cards.
- High-speed network – a 10Gb network

There currently exists two seperate partitions of the A100 nodes:
- `a100_full`: 2 nodes where each A100 GPU memory is not partitioned and remains full at 80Gb per card.
- `a100_mig`: 1 node where each A100 GPU is partitioned into 3, resulting in 9 GPU partitions each with ??GB.

It is important to note that you cannot employ different GPU partitions in a distributed fashion, hence the current maximum distributed setting is 2 `a100_full` nodes with 9x A100 80GB GPU's in total.

### 2. Accessing the cluster

Maxwell can be accessed via SSH when connected to the University network (not including eduroam).

To access the 

If you run into any issues at this stage contact servicedesk@abdn.ac.uk  

### 3. Initial setup, conda environment

Once you are logged into Macleod you will find yourself in the home directory. The storage on Macleod is split into two partitions `home/` and `sharedscratch/` to keep things simple I recommend saving all your files to your personal `sharedscratch/` parition, this is due to the `sharedscratch` having a larger storage limit than the `home` partition. Simply change to the `sharedscratch/` directory from the home directory using `cd sharedscratch`. The full `sharedscratch` path is defined as: `/home/<username>/sharedscratch/`

Now you have connected to Macleod you must access your required packages, if your packages are not present in the software list (https://www.abdn.ac.uk/it/documents-uni-only/Maxwell-Galaxy-Software.pdf ), then the easiest way to download them is to create an anaconda environment. 

First load anaconda via:
`module load anaconda3`

Now you can create an environment using the following:
`conda create -n <your_env_name> python=3.8`

**Note:** For first time installs you may be prompted to configure your shell, if you receive this error run the command `conda init bash` and then exit / close the terminal, ssh back into macleod and load the anaconda3 module to continue.

Once created, activate the environment
`conda activate <your_env_name>` 

You should now see the `(<your_env_name>)` in the bash shell.

Now you can go ahead and install all your desired packages.
For this example we will install the following (Note: when installing PyTorch use CUDA version 11.3):
`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
`conda install -c conda-forge matplotlib`
`conda install -c conda-forge tqdm`

Now you have a virtual environment created and all your packages installed, you can now use this environment everytime you use macleod. Remember you must load the module anaconda3 before you activate your environment.

### 4. Getting your data on Macleod

The easiest solution to uploading your code is through github, to download from an online repository. This is not only simple, but using version control and web hosting is best practice and should be employed.

If you must upload from your local device you can use SFTP from your own device to Macloed by the following, which jumps through the ssh-gateway.

`sftp -J <username>@ssh-gateway.abdn.ac.uk <username>@macleod1.abdn.ac.uk`

### 5. Intro to SLURM and running your code

Unlike your standard desktop computer, you must submit a “job” for execution to a scheduler. Here the Macleod cluster uses SLURM workload manager to schedule the allocation of resources for jobs.

Some useful commands are as follows:
`sinfo` - Show summary information 
`squeue` - Show job queue
`scontrol show partition gpu` - Show gpu partition details
`chkquota` - Show hard drive allocation used and free space

To run a job (run your program) on Macleod you must submit a script to the SLURM scheduler. The SLURM script must contain 3 things:
1. Define the resource requirements for the job
2. Activate the environment we created earlier
3. Specify the script we wish to run

An example script `run_example.sh` for this introduction guide is given here:

```
#!/bin/bash
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=12 # number of cores
#SBATCH --mem=32G # memory pool for all cores

#SBATCH --ntasks-per-node=1 # one job per node
#SBATCH --gres=gpu:7 # 7 of the 21 paritions
#SBATCH --partition=ncs-staff 

#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=<username>@abdn.ac.uk 

module load anaconda3
source activate <your_env_name>

srun python example_script.py --epochs=25 --save /home/<username>/sharedscratch/
```

This script specifies we wish to use 1 GPU node, 7 of the 21 GPU partitions, 32GB of RAM, 12 CPU cores.

When specifying the partition:
- Students should use: `#SBATCH --partition=gpu`
- NCS staff should use: `#SBATCH --partition=ncs-staff`

To submit the above job script named `run_example.sh` to the Slurm scheduler we use the `sbatch` command:
`sbatch run_example.sh`

The job may run immediately or may take upto a few days to run depending on the number of existing jobs, you will receive an email notification if you use the SLURM command `email`.

To check the status of queued and running jobs, use the following:
`squeue -u <username>` 

If you wish to cancel a job simply use:
`scancel <job_ID>`

### 6. Info about your code (PyTorch)

Example PyTorch code is given in this repository to help you get started, but generally, there are a few rules to follow.

When selecting how many GPUS are available, do not specify device ID's, this will use all available GPUs. If you specify GPU devices you may run into errors, this is a current issue with the GPU partitioning and how SLURM identifies the GPUs.
`net = torch.nn.DataParallel(model)`


### Additional Resources 

Full Aberdeen HPC documentation: 
https://www.abdn.ac.uk/it/documents-uni-only/OCF-User0-Manual-Abderdeen-Maxwell.pdf

Full SLURM documentation:
https://slurm.schedmd.com/documentation.html 

More details regarding SLURM can be found at: 
https://slurm.schedmd.com/tutorials.html 

Conda documentation:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html 
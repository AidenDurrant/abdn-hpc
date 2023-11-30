## SLURM settings

```
#SBATCH --gres=gpu:7 # 7 of the 21 partitions
#SBATCH --partition=gpu
```

## PyTorch Code Changes

Check if GPU is available, if so, set device to 'cuda'.

```
# Check if GPUs are available, if so set device to CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Send the model to GPU and wrap model with [nn.dataparallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).

```
net = VGG()
net = net.to(device) # Send to CUDA aka GPU

if device == 'cuda':
    net = torch.nn.DataParallel(net) # Use all available GPUs in dataparallel mode
```

Send objective funtion and data to GPU

```
criterion = nn.CrossEntropyLoss().to(device) # Loss Function

...

inputs, targets = inputs.to(device), targets.to(device) # Send samples to device
```

### Common errors
If you recieve an error of the kind:

`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

It is likely that a component has not be cast to the GPU.
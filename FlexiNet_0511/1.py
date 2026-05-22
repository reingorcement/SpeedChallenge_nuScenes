import torch

cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()

print(f'PyTorch Version: {torch.__version__}')
print(f'PyTorch built with CUDA: {torch.version.cuda}')
print(f'CUDA available: {cuda_available}')
print(f'CUDA device count: {device_count}')
print(f'CUDA arch list: {torch.cuda.get_arch_list()}')

if cuda_available and device_count > 0:
	print(f'GPU Device: {torch.cuda.get_device_name(0)}')
	print(f'GPU Capability: {torch.cuda.get_device_capability(0)}')
else:
	print('GPU Device: N/A')

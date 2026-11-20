import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    props = torch.cuda.get_device_properties(0)
    memory = props.total_memory / (1024 ** 3)

    print("GPU memory:", round(memory, 2), "GB")

    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x

    print("GPU computation successful!")
    print("Result shape:", y.shape)

else:
    print("GPU not detected!")

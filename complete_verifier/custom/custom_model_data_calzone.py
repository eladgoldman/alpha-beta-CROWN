import os
import torch
from torchvision import transforms
from torchvision import datasets
import arguments

def calzone_mnist(spec, pixels_group):
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.MNIST(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False, num_workers=1)
    X, labels = next(iter(testloader))

    data_max_res = X.clone()
    data_min_res = X.clone()
    data_max =     data_max_res.view(-1)
    data_min =     data_min_res.view(-1)
    
    for pixel_index in pixels_group:
        data_max[pixel_index] = 1
        data_min[pixel_index] = 0

    for i in range(len(data_max)):
        data_max[i] = (data_max[i] - mean) / std
        data_min[i] = (data_min[i] - mean) / std

    # In this case, the epsilon does not matter here.
    ret_eps = None

    return X, labels, data_max_res, data_min_res, ret_eps
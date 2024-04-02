import abcrown
from load_model import load_model_onnx

import arguments
from utils import expand_path
from custom.custom_model_data_calzone import mnist_loader
from specifications import construct_vnnlib
import torch
from l_zero_gpu_worker import LZeroGpuWorker

def make_bounds(image, pixels_group ,mean, std):
    
    data_max_res = image.clone()
    data_min_res = image.clone()
    
    data_max =     data_max_res.view(-1)
    data_min =     data_min_res.view(-1)
    
    for pixel_index in pixels_group:
        data_max[pixel_index] = 1
        data_min[pixel_index] = 0

    for i in range(len(data_max)):
        data_max[i] = (data_max[i] - mean) / std
        data_min[i] = (data_min[i] - mean) / std

    return data_max_res, data_min_res


def l0_verify(model_ori, image, label, pixels_group, mean, std ):
    data_max_res, data_min_res = make_bounds(image, pixels_group ,mean, std)

    verification_dataset = {'X': torch.unsqueeze(image, 0), 
                            'labels': torch.unsqueeze(label, 0),
                            'data_max': torch.unsqueeze(data_max_res,0),
                            'data_min': torch.unsqueeze(data_min_res,0),
                            }
    
    vnnlib = construct_vnnlib(verification_dataset, [0])[0]
        
    # some stuff before every run
    model_ori.eval()
    vnnlib_shape = shape

    x_range = torch.tensor(vnnlib[0][0])
    data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
    data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
    x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
    
    device = 'cpu'
    model_ori = model_ori.to(device)
    x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)
    
    return abcrown.incomplete_verifier(
                    model_ori, x, data_ub=data_max, data_lb=data_min, vnnlib=vnnlib)

if __name__ == '__main__':
    
    model_ori, _ = load_model_onnx(expand_path(
        arguments.Config["model"]["onnx_path"]))
    

    # abcrown = ABCROWN(args=["--config", yaml_path, "--device" ,"cpu"])
    yaml_path = "exp_configs\\beta_crown\\calzone.yaml"

    l_zero_gpu_worker = LZeroGpuWorker(port=6000, means=[0.0], stds=[1.0], network=model_ori, config_path=yaml_path,
                                is_conv=False)    
    l_zero_gpu_worker.work()
    


    # with open(yaml_path, 'w') as outfile:
    #     yaml.dump(conf, outfile, default_flow_style=False)

    

    
    # X, labels, _ ,_,_ = mnist_loader()    
    # shape = [-1] + list(X.shape[1:])

    # image_index = 0
    
    # pixels_groups = [
    #     [12,36,5],
    #     [13,2,5],
    #     [14,57,88,99,246],
    #     [1,634,225,90,246],
    #     [124,58,87,235,246],
    #     [75,346,345,23,246],                    
    # ]
    
    # for pixels_group in pixels_groups:
    #     verified_status, ret = l0_verify(model_ori, X[image_index], labels[image_index], pixels_group, conf['data']['mean'], conf['data']['std'] )
    #     print("verified_status = " + str(verified_status) + ", ret = " + str(ret))   
    
    # abcrown.main()
    
    
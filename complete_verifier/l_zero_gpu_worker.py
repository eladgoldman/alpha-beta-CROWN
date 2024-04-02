import time
from multiprocessing.connection import Listener
from random import shuffle, sample
import numpy as np
from specifications import construct_vnnlib
import torch
import abcrown
from abcrown import ABCROWN
import yaml
from pathlib import Path

class LZeroGpuWorker:
    def __init__(self, port, config_path, network, means, stds, is_conv):
        self.__port = port
        self.__network = network
        self.__means = means
        self.__stds = stds
        self.__is_conv = is_conv
        self.__config = yaml.safe_load(Path(config_path).read_text())
        self.__abcrown = ABCROWN(args=["--config", config_path])

    def work(self):
        address = ('localhost', self.__port)
        with Listener(address, authkey=b'secret password') as listener:
            print(f"Waiting at port {self.__port}")
            with listener.accept() as conn:
                # Every iteration of this loop is one image
                message = conn.recv()
                while message != 'terminate':
                    image, label, sampling_lower_bound, sampling_upper_bound, repetitions = message
                    sampling_successes, sampling_time = self.__sample(image, label, sampling_lower_bound, sampling_upper_bound, repetitions)
                    conn.send((sampling_successes, sampling_time))
                    image, label, strategy, worker_index, number_of_workers = conn.recv()
                    coverings = self.__load_coverings(strategy)
                    self.__prove(conn, image, label, strategy, worker_index, number_of_workers, coverings)
                    message = conn.recv()

    def __sample(self, image, label, sampling_lower_bound, sampling_upper_bound, repetitions):
        population = list(range(0, len(image)))
        sampling_successes = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        sampling_time = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        for size in range(sampling_lower_bound, sampling_upper_bound + 1):
            for i in range(0, repetitions):
                pixels = sample(population, size)
                start = time.time()
                verified = self.verify_group(image, label, pixels)
                duration = time.time() - start
                sampling_time[size - sampling_lower_bound] += duration
                if verified:
                    sampling_successes[size - sampling_lower_bound] += 1

        return sampling_successes, sampling_time

    def __load_coverings(self, strategy):
        t = strategy[-1]
        coverings = dict()
        for size, broken_size in zip(strategy, strategy[1:]):
            covering = []
            with open(f'../../coverings/({size},{broken_size},{t}).txt',
                      'r') as coverings_file:
                for line in coverings_file:
                    # TODO: ignore last line of file
                    block = tuple(int(item) for item in line.split(','))
                    covering.append(block)
                coverings[size] = covering
        return coverings

    def __prove(self, conn, image, label, strategy, worker_index, number_of_workers, coverings):
        t = strategy[-1]
        with open(f'../../coverings/({len(image)},{strategy[0]},{t}).txt',
                  'r') as shared_covering:
            for line_number, line in enumerate(shared_covering):
                if conn.poll() and conn.recv() == 'stop':
                    conn.send('stopped')
                    return
                if line_number % number_of_workers == worker_index:
                    pixels = tuple(int(item) for item in line.split(','))
                    start = time.time()
                    verified = self.verify_group(image, label, pixels)
                    duration = time.time() - start
                    if verified:
                        conn.send((True, len(pixels), duration))
                    else:
                        conn.send((False, len(pixels), duration))
                        if len(pixels) not in coverings:
                            conn.send('adversarial-example-suspect')
                            conn.send(pixels)
                        else:
                            groups_to_verify = self.__break_failed_group(pixels, coverings[len(pixels)])
                            while len(groups_to_verify) > 0:
                                if conn.poll() and conn.recv() == 'stop':
                                    conn.send('stopped')
                                    return
                                group_to_verify = groups_to_verify.pop(0)
                                start = time.time()
                                verified = self.verify_group(image, label, group_to_verify)
                                duration = time.time() - start
                                if verified:
                                    conn.send((True, len(group_to_verify), duration))
                                else:
                                    conn.send((False, len(group_to_verify), duration))
                                    if len(group_to_verify) in coverings:
                                        groups_to_verify = self.__break_failed_group(group_to_verify, coverings[len(group_to_verify)]) + groups_to_verify
                                    else:
                                        conn.send('adversarial-example-suspect')
                                        conn.send(group_to_verify)
                    conn.send('next')
        conn.send("done")
        message = conn.recv()
        if message != 'stop':
            raise Exception('This should not happen')
        conn.send('stopped')

    def __break_failed_group(self, pixels, covering):
        permutation = list(pixels)
        shuffle(permutation)
        return [tuple(sorted(permutation[item] for item in block)) for block in covering]

    def verify_group(self, image, label, pixels_group):
        specLB = np.copy(image)
        specUB = np.copy(image)
        for pixel_index in pixels_group:
            specLB[pixel_index] = 0
            specUB[pixel_index] = 1
        self.normalize(specLB)
        self.normalize(specUB)

        return self.l0_verify(image, label, specUB, specLB)
        # return self.__network.test(specLB, specUB, label)

    def normalize(self, image):
        # normalization taken out of the network
        for i in range(len(image)):
            image[i] = (image[i] - self.__means[0]) / self.__stds[0]
       
                    
    def l0_verify(self, image, label, data_max_res, data_min_res):

        shape = [-1] + list(image.shape)
        verification_dataset = {'X': torch.unsqueeze(torch.from_numpy(image), 0), 
                                'labels': torch.unsqueeze(torch.from_numpy(label), 0),
                                'data_max': torch.unsqueeze(torch.from_numpy(data_max_res),0),
                                'data_min': torch.unsqueeze(torch.from_numpy(data_min_res),0),
                                }
        
        vnnlib = construct_vnnlib(verification_dataset, [0])[0]
            
        # some stuff before every run
        self.__network.eval()
        vnnlib_shape = shape

        x_range = torch.tensor(vnnlib[0][0])
        data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
        data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
        x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
        
        device = 'gpu'
        self.__network = self.__network.to(device)
        x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)
        
        res, _ = self.__abcrown.incomplete_verifier(
                        self.__network, x, data_ub=data_max, data_lb=data_min, vnnlib=vnnlib)
        
        return (res == 'safe-incomplete')
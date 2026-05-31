import numpy as np
from typing import Callable, Union, List
import matplotlib.pyplot as plt
def read_array_from_file(filename: str) -> np.ndarray:
    try:
        with open(filename, 'r') as file:
            array = []
            for line in file:
                line = line.strip()
                if line:  # 非空行
                    try:
                        array.append(float(line))
                    except ValueError:
                        print(f"警告: 跳过非数字内容: '{line}'")
            return np.array(array)
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到")
        return np.array([])

def add_poisson_noise(signal: Union[List[float], np.ndarray], 
                      scaling_factor: float = 1.0, 
                      seed: int = 19260817) -> np.ndarray:
    signal_arr = np.array(signal, dtype=float)
    
    if seed is not None:
        np.random.seed(seed)
    
    scaled_signal = signal_arr * scaling_factor
    
    scaled_signal = np.maximum(scaled_signal, 0)
    
    noisy_signal = np.random.poisson(scaled_signal)
    
    if scaling_factor != 1.0:
        noisy_signal = noisy_signal / scaling_factor
    
    return noisy_signal

def convolve_1d(array: Union[List[float], np.ndarray], 
                kernel_func: Callable[[int], float], 
                kernel_size: int) -> np.ndarray:
    
    # 转换为numpy数组以便处理
    arr = np.array(array)
    n = len(arr)
    
    if kernel_size % 2 == 0:
        raise ValueError("卷积核大小必须为奇数")
    
    half_kernel = kernel_size // 2
    
    kernel = np.array([kernel_func(i - half_kernel) for i in range(kernel_size)])
    
    result = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - half_kernel)
        end = min(n, i + half_kernel + 1)
        
        kernel_start = max(0, half_kernel - i)
        kernel_end = kernel_size - max(0, (i + half_kernel + 1) - n)
        kernel_end = kernel_size - max(0, (i + half_kernel + 1) - n)
        
        result[i] = np.sum(arr[start:end] * kernel[kernel_start:kernel_end])
    
    return result

if __name__ == "__main__":
    # 创建一个示例数组
    array = read_array_from_file('outputsimu.txt')
    standard1 = np.average(array[120:140])
    array = [x/standard1 for x in array]
    array2 = read_array_from_file('outputactual.txt') 
    standard2 = np.average(array2[120:140])
    array2 = [x/standard2 for x in array2]
    #array = array/np.average(array)
    def gaussian_kernel(x,sigma):
        return np.exp(-x**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
    nx=np.arange(-5,5,1)
    gx1=gaussian_kernel(nx,1)
    gx5=gaussian_kernel(nx,5)
    # result_gaussian = convolve_1d(array, gaussian_kernel, 9)
    result_gaussian1 = np.convolve(array, gx1, mode='valid')
    result_gaussian1=result_gaussian1/np.average(result_gaussian1)
    # plt.plot(result_gaussian1,label='convolved1')
    result_gaussian5 = np.convolve(array, gx5, mode='valid')
    result_gaussian5=result_gaussian5/np.average(result_gaussian5)
    # plt.plot(result_gaussian5,label='convolved5')
    array1k =add_poisson_noise(array,scaling_factor=3e3,seed=123)
    array5h =add_poisson_noise(array,scaling_factor=3e3,seed=456)
    print(np.std(array2[0:40]))
    print(np.std(array1k[120:140]))
    print(np.std(array5h[120:140]))
    plt.plot(array,label='simulation (no noise)')    
    #array2=array1/array
    # plt.plot(array1k,label='simulation (lambda=1000)')
    plt.plot(array5h,label='simulation (lambda=3e3)')
    plt.plot(array2,label='experiment')
    #plt.plot(result_gaussian)
    # plt.xlabel("Pixel ID")
    # plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.show()

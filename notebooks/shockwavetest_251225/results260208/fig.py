import numpy as np
import matplotlib.pyplot as plt

def process_txt_file(file_path, sigma=0.1):
    """
    处理txt文件中的一维数组
    
    参数:
    file_path: txt文件路径
    n2: 要扩展的行数
    sigma: 噪声的标准差（默认0.1）
    
    返回:
    处理后的二维数组
    """
    
    # 1. 读取txt文件中的1*n1一维数组，n1未知
    try:
        # 读取文件内容
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # 处理多种可能的分隔符（逗号、空格、制表符、换行符等）
        # 使用正则表达式分割数字
        import re
        numbers = re.split(r'[,\s\t\n]+', content)
        
        # 过滤空字符串并转换为浮点数
        data_1d = np.array([float(x) for x in numbers if x])
        n1 = len(data_1d)
        
        print(f"成功读取数据: n1 = {n1}")
        print(f"原始数据: {data_1d}")
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    # 2. 归一化（最小-最大归一化到[0,1]范围）
    if data_1d.max() == data_1d.min():
        # 如果所有值相同，归一化为0.5
        normalized_data = np.ones_like(data_1d) * 0.5
        print("警告：所有数据值相同，归一化为0.5")
    else:
        normalized_data = (data_1d - data_1d.min()) / (data_1d.max() - data_1d.min())
    
    # print(f"归一化后数据: {normalized_data}")
    # print(f"归一化范围: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")

    noise=np.random.normal(0, sigma, normalized_data.shape)
    # 添加噪声
    noisy_data = normalized_data + noise
    
    # 可选：将数据限制在[0,1]范围内（如果添加噪声后超出范围）
    # noisy_data = np.clip(noisy_data, 0, 1)
    
    # print(f"添加噪声后的数组形状: {noisy_data.shape}")
    # print(f"噪声统计: 均值={noise.mean():.6f}, 标准差={noise.std():.6f}")
    # print(f"最终数据范围: [{noisy_data.min():.4f}, {noisy_data.max():.4f}]")
    
    return noisy_data

def save_results(data, output_file='result.npy', output_txt='result.txt'):
    """
    保存处理结果
    
    参数:
    data: 要保存的数据
    output_file: numpy格式输出文件
    output_txt: 文本格式输出文件
    """
    # 保存为numpy格式
    np.save(output_file, data)
    print(f"结果已保存为numpy格式: {output_file}")
    
    # 保存为文本格式（方便查看）
    np.savetxt(output_txt, data, fmt='%.6f', delimiter=',')
    print(f"结果已保存为文本格式: {output_txt}")

# 示例使用
if __name__ == "__main__":
    # 配置参数
    times=np.arange(1,393)
    for ii in times:
        print(ii)
        input_file = '260208_output_shocksample_shot_timestamp'+str(ii)+'.txt'  # 输入文件路径
        sigma = 1/(262*np.sqrt(0.0018))             # 噪声的标准差
        result = process_txt_file(input_file, sigma=sigma)
        result = result[np.newaxis]
        if ii==1:
            total = result
        else:
            total = np.concatenate((total, result), axis=0)
    print(f"扩展后的数组形状: {total.shape}")
    total=np.rot90(total)
    fig, ax = plt.subplots(figsize=(10, 6))
    num_columns = total.shape[1]
    num_rows = total.shape[0]
    time_step = 0.02  # ns
    position_step = 3.33 #um
    total_time = num_columns * time_step
    total_length = num_rows * position_step
    im = ax.imshow(total, aspect='auto', origin='upper', 
                extent=[0, total_time, 0, total_length],cmap='grey')
    ax.set_xlabel('Time/ns')
    ax.set_ylabel('Position/um')
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.AutoLocator()) 
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
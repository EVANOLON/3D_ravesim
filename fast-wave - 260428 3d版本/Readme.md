<!-- Copyright (c) 2024, ETH Zurich -->

# Fast Wave

Sacrifice the ability to compute arbitrarily large simulations for perforance gains.

This program is separated into various CMake targets:

* `fwcuda`: located in the `fwcuda/` directory, this is the part that directly calls CUDA kernels. This is the only part of the main that is compiled with nvcc. We separate this out so that the rest of the code can use more C++ features and libraries without being limited by nvcc bugs.
* `fastwave_lib`: The library that implements the main application logic of running the simulation according to the given configuration
* `fastwave`: The executable that the end user calls from the command line to use `fastwave_lib`.
* `fwtest`: The unit testing executable. This part is also compiled with nvcc so that we can directly call kernels from the tests.

This code is sparsely documented. Much of the functionality is a one-to-one translation from the Python version
in `big-wave`, so check there if anything is unclear.

## Usage

Once the `fastwave` program has been compiled (see build instructions below), you can run simulations with a similar workflow to what is used for `big-wave`.

Use `sim_dir = multisim.setup_simulation(...)` from `big-wave` to create a simulation directory. Then run the simulation like that:

```bash
./fastwave /path/to/sim/dir -s 123
```

where 123 is the source index (0-based). To run multiple source points just use a for loop.

To run on a specific GPU, pass the 0-based gpu index as an env variable:

```bash
# run on the second gpu
CUDA_VISIBLE_DEVICES=1 ./fastwave /path/to/sim/dir -s 123
```

The history can be saved by using the optional argument `--history_dz 0.025`.

Remember to recompile if the c++ or cuda code has changed.

## Build Instructions

The project is built in the normal CMake way. On Euler you'll have to load a recent cuda version first.

```bash
# this first line is only needed on euler
env2lmod && module load gcc/9.3.0 cuda/12.1.1

cd fast-wave

mkdir build-Release
cd build-Release

# On Hari add the -DCUDAToolkit_INCLUDE_DIR option. This might not be
# required on other machines.
cmake -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_INCLUDE_DIR=/usr/include ..

# The above step doesn't compile the program yet, it's just the CMake configure step. This next line
# is the build step. If the code has changed, only this next line has to be re-run.
cmake --build . --target fastwave

# Run the tests. This requires c++20 and therefore does not work
# on euler unless you want to compile your own non-ancient gcc.
cmake --build . --target fwtest
ctest
```

The cuda toolkit has to be installed for this to work. Other dependencies are fetched from the internet during configure time (during the first of the cmake commands above).

If you run into git submodule errors while running CMake, it might be that you don't have an ssh key for github set up on that machine. One of the dependencies uses git submodules with an ssh address instead of an https address, which breaks in that case. You can fix this issue by adding these lines to your `~/.gitconfig`:

```
[url "https://github.com/"]
    insteadOf = git@github.com:
```

This project does not depend on big-fourier or big-wave, you don't have to build them to use this.

## todo:

does it make sense to keep the data in a chunked format? Govindaraju et al.: Our hierarchical FFT minimizes the number of memory accesses by combining transpose operations with the FFT computation.

## update:
相比1230版增加了propagate_convolve_step, apply_sample_factors与 square_and_downsample 的 2D版本，修改了inner_2d
可通过is2d实现2d/3d仿真的切换。需手动设置
3d版本存在Segmentation fault
原因：
1. GPU内存相关问题（最常见）
序号	原因	具体表现
1.1	内存分配失败	cudaMalloc 返回错误，但未检查
1.2	内存访问越界	核函数访问超出分配范围的内存
1.3	内存未初始化	使用未初始化的设备指针
1.4	内存释放后访问	释放GPU内存后仍尝试访问
1.5	内存对齐问题	访问未对齐的内存地址
1.6	内存拷贝错误	cudaMemcpy 参数错误（方向、大小）
2. 核函数与线程配置问题
序号	原因	具体表现
2.1	线程索引计算错误	ix, iy 计算错误，导致越界
2.2	网格/块尺寸配置错误	gridDim, blockDim 设置不当
2.3	线程数超出限制	每个块超过1024个线程
2.4	共享内存溢出	分配的共享内存超过限制
2.5	寄存器溢出	核函数使用过多寄存器
3. 参数传递问题
序号	原因	具体表现
3.1	std::optional 处理错误	设备代码中 get_optional_value 实现错误
3.2	结构体填充/对齐	SimParams 等结构体在主机/设备内存中的布局不一致
3.3	指针传递错误	传递主机指针给核函数，或反之
3.4	参数类型不匹配	声明和定义参数类型不一致
4. CUDA API调用问题
序号	原因	具体表现
4.1	未检查CUDA错误	cudaGetLastError() 未正确调用
4.2	异步操作未同步	核函数执行和内存拷贝未正确同步
4.3	多流操作冲突	多个CUDA流访问相同资源
4.4	上下文错误	CUDA上下文创建/销毁问题
5. 代码逻辑问题
序号	原因	具体表现
5.1	除零错误	在核函数中进行除法运算，分母可能为0
5.2	浮点异常	无效的浮点运算（如sqrt负数）
5.3	无限递归/循环	设备代码中的逻辑错误
5.4	数据竞争	多个线程同时写入同一内存位置
5.5	分支分歧过大	核函数中if-else分支导致warp内线程执行不同路径
6. 编译链接问题
序号	原因	具体表现
6.1	模板实例化缺失	缺少必要的模板实例化声明
6.2	声明定义不匹配	函数签名不一致
6.3	编译器优化错误	某些优化级别导致问题
6.4	ABI不兼容	CUDA版本和GCC版本不匹配
7. 资源限制问题
序号	原因	具体表现
7.1	GPU内存不足	分配的总内存超过GPU容量
7.2	计算资源耗尽	同时启动的核函数过多
7.3	系统内存不足	主机内存分配失败
7.4	文件描述符耗尽	打开文件过多
8. 特定于2D仿真的问题
序号	原因	具体表现
<!-- 8.1	get_optional_value 设备实现错误	设备代码中无法正确读取 std::optional -->
8.2	2D索引到1D的映射错误	idx = iy * nx + ix 计算错误
8.3	边界检查不完整	2D核函数中边界条件处理错误
8.4	2D FFT问题	2D FFT的参数配置错误
# Linux shared library build notes

这个目录用于在没有 MATLAB 的 Linux 环境中，把 Windows MATLAB Coder 已生成的 C++ 源码编译成 `libextractAllFeatures.so`。

当前源码对应 64 维 feature schema。不要继续使用旧 62 维 `.so`；Python 侧的输出缓冲区和 `feature_map.json` 也必须同步为 64 维。

## 1. Windows 侧准备

在 Windows MATLAB 中进入本工程目录：

```matlab
cd('D:\bj_project\matlab_poj\extractor_feature_03')
build_windows_dll
package_codegen_source_for_linux
```

`build_windows_dll` 负责生成 MATLAB Coder C++ 源码和 Windows DLL。
`package_codegen_source_for_linux` 只打包源码和 C ABI wrapper，不会生成 Linux `.so`。

打包输出目录：

```text
build/linux_source_package/
├── src/
├── native_api/
├── build_linux_so.sh
├── README_linux_build.md
└── source_manifest.txt
```

`src/` 中同时包含 Coder 生成源码和独立编译需要的 `tmwtypes.h`，Linux 目标机不需要安装 MATLAB。

## 2. 复制到 Linux

把整个 `build/linux_source_package` 目录复制到 Linux，例如：

```bash
scp -r build/linux_source_package user@linux-host:/tmp/iq_feature_linux_source
```

Linux 侧不需要安装 MATLAB，只需要 C++ 编译器。

## 3. Linux 侧编译

在 Linux 上运行：

```bash
cd /tmp/iq_feature_linux_source
chmod +x build_linux_so.sh
./build_linux_so.sh --clean
```

输出：

```text
out/libextractAllFeatures.so
```

编译完成后，把动态库和 C ABI 头文件分别同步到当前 Python 核心包：

```bash
cp out/libextractAllFeatures.so \
  /home/dianci/projects/signal_fusion/src/signal_fusion/feature_extraction/assets/native/linux/
cp native_api/iq_feature_c_api.h \
  /home/dianci/projects/signal_fusion/src/signal_fusion/feature_extraction/assets/include/
```

安装后目录中至少应有：

```text
/home/dianci/projects/signal_fusion/src/signal_fusion/feature_extraction/assets/native/linux/libextractAllFeatures.so
/home/dianci/projects/signal_fusion/src/signal_fusion/feature_extraction/assets/include/iq_feature_c_api.h
```

## 4. 检查依赖

编译脚本结束时会自动执行：

```bash
ldd out/libextractAllFeatures.so
```

如果看到 `not found`，说明 Linux 运行环境缺少依赖库。优先安装系统依赖，例如：

```bash
sudo apt-get update
sudo apt-get install -y build-essential libgomp1
```

如果没有 sudo 权限，也可以把缺失的 `.so` 放到 `native/linux`，或运行前设置：

```bash
export LD_LIBRARY_PATH=/path/to/native/linux:$LD_LIBRARY_PATH
```

## 5. 在核心包中测试

```bash
cd /home/dianci/projects/signal_fusion
PYTHONPATH=src python -m signal_fusion.feature_extraction.cli \
  --data_path /path/to/input_dataset.npz \
  --output_path /path/to/output_features.npz \
  --data_format npz \
  --x_key X \
  --sample_rate 100000000 \
  --max_samples 2
```

默认情况下，Linux 版 Python 后端会查找：

```text
src/signal_fusion/feature_extraction/assets/native/linux/libextractAllFeatures.so
```

如果动态库放在其他位置，可以显式传入：

```bash
--dll_path /absolute/path/to/libextractAllFeatures.so
```

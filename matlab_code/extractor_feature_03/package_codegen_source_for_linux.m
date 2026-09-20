function package_codegen_source_for_linux()
%PACKAGE_CODEGEN_SOURCE_FOR_LINUX 打包 MATLAB Coder 生成的 C++ 源码，供 Linux 编译 .so。
%
% 使用场景：
%   1. Windows MATLAB 已经运行 build_windows_dll.m，生成 build/windows_x86_64。
%   2. Linux 机器没有 MATLAB，只负责用 g++ 编译共享库。
%
% 输出目录：
%   build/linux_source_package/
%       src/              MATLAB Coder 生成的 .cpp/.h 源码
%       native_api/       Python ctypes 使用的 C ABI wrapper
%       build_linux_so.sh Linux 编译脚本
%       README_linux_build.md 使用说明
%       source_manifest.txt 文件清单

clc;

projectRoot = fileparts(mfilename('fullpath'));
generatedSourceDir = fullfile(projectRoot, 'build', 'windows_x86_64');
nativeApiDir = fullfile(projectRoot, 'native_api');
packageDir = fullfile(projectRoot, 'build', 'linux_source_package');
srcOutDir = fullfile(packageDir, 'src');
nativeOutDir = fullfile(packageDir, 'native_api');

fprintf('[package] project root: %s\n', projectRoot);
fprintf('[package] generated source: %s\n', generatedSourceDir);

if ~isfolder(generatedSourceDir)
    error('Generated source directory not found: %s\nRun build_windows_dll.m first.', generatedSourceDir);
end

if ~isfolder(nativeApiDir)
    error('native_api directory not found: %s', nativeApiDir);
end

requiredGeneratedFiles = [ ...
    "extractAllFeatures.cpp", ...
    "extractAllFeatures.h", ...
    "extractAllFeatures_initialize.h", ...
    "extractAllFeatures_terminate.h" ...
];

for i = 1:numel(requiredGeneratedFiles)
    requiredPath = fullfile(generatedSourceDir, requiredGeneratedFiles(i));
    if ~isfile(requiredPath)
        error('Required generated file not found: %s', requiredPath);
    end
end

requiredNativeFiles = ["iq_feature_c_api.cpp", "iq_feature_c_api.h"];
for i = 1:numel(requiredNativeFiles)
    requiredPath = fullfile(nativeApiDir, requiredNativeFiles(i));
    if ~isfile(requiredPath)
        error('Required native API file not found: %s', requiredPath);
    end
end

% build/linux_source_package 是可再生成产物，重新打包前先清理旧包。
if isfolder(packageDir)
    rmdir(packageDir, 's');
end
mkdir(packageDir);
mkdir(srcOutDir);
mkdir(nativeOutDir);

manifestLines = strings(0, 1);

sourcePatterns = ["*.cpp", "*.c", "*.h", "*.hpp"];
for p = 1:numel(sourcePatterns)
    files = dir(fullfile(generatedSourceDir, sourcePatterns(p)));
    for k = 1:numel(files)
        src = fullfile(files(k).folder, files(k).name);
        dst = fullfile(srcOutDir, files(k).name);
        copyfile(src, dst);
        manifestLines(end + 1, 1) = "src/" + string(files(k).name); %#ok<AGROW>
    end
end

% R2025b 生成的 rtwtypes.h 会引用 MATLAB 公共类型头文件 tmwtypes.h。
% Linux 目标机不安装 MATLAB，因此在 Windows 打包阶段一并复制该头文件。
supportHeaders = "tmwtypes.h";
matlabIncludeDir = fullfile(matlabroot, 'extern', 'include');

for i = 1:numel(supportHeaders)
    supportPath = fullfile(matlabIncludeDir, supportHeaders(i));
    if ~isfile(supportPath)
        error('Required MATLAB support header not found: %s', supportPath);
    end

    copyfile(supportPath, fullfile(srcOutDir, supportHeaders(i)));
    manifestLines(end + 1, 1) = "src/" + supportHeaders(i); %#ok<AGROW>
end

nativePatterns = ["*.cpp", "*.c", "*.h", "*.hpp"];
for p = 1:numel(nativePatterns)
    files = dir(fullfile(nativeApiDir, nativePatterns(p)));
    for k = 1:numel(files)
        src = fullfile(files(k).folder, files(k).name);
        dst = fullfile(nativeOutDir, files(k).name);
        copyfile(src, dst);
        manifestLines(end + 1, 1) = "native_api/" + string(files(k).name); %#ok<AGROW>
    end
end

helperFiles = ["build_linux_so.sh", "README_linux_build.md"];
for i = 1:numel(helperFiles)
    helperPath = fullfile(projectRoot, helperFiles(i));
    if isfile(helperPath)
        copyfile(helperPath, fullfile(packageDir, helperFiles(i)));
        manifestLines(end + 1, 1) = string(helperFiles(i)); %#ok<AGROW>
    else
        warning('Helper file not found, skip copy: %s', helperPath);
    end
end

manifestPath = fullfile(packageDir, 'source_manifest.txt');
fid = fopen(manifestPath, 'w');
if fid < 0
    error('Cannot write manifest: %s', manifestPath);
end
cleanupObj = onCleanup(@() fclose(fid));
for i = 1:numel(manifestLines)
    fprintf(fid, '%s\n', manifestLines(i));
end
clear cleanupObj;

fprintf('[package] copied %d files.\n', numel(manifestLines));
fprintf('[package] package directory: %s\n', packageDir);
fprintf('\nNext steps on Linux:\n');
fprintf('  cd <copied linux_source_package>\n');
fprintf('  chmod +x build_linux_so.sh\n');
fprintf('  ./build_linux_so.sh --clean\n');
end

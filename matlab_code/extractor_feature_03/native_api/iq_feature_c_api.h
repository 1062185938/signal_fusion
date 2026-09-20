#ifndef IQ_FEATURE_C_API_H
#define IQ_FEATURE_C_API_H

#include <stdint.h>

/*
 * 跨平台动态库导出宏
 *
 * Windows:
 *   __declspec(dllexport)
 *
 * Linux:
 *   default visibility
 */
#if defined(_WIN32) || defined(__CYGWIN__)
    #define IQ_FEATURE_API __declspec(dllexport)
#else
    #define IQ_FEATURE_API __attribute__((visibility("default")))
#endif


/*
 * C ABI
 *
 * 即使整个动态库使用 C++ 编译，
 * 对 Python 暴露的三个接口仍使用稳定的 C linkage。
 */
#ifdef __cplusplus
extern "C" {
#endif


/*
 * 初始化特征提取器。
 *
 * 一个进程中通常只需要调用一次。
 */
IQ_FEATURE_API void iqFeatureInitialize(void);


/*
 * 提取一段 IQ 信号的 64 维特征。
 *
 * Input:
 *
 *   iData
 *       长度至少为 16384 的 float 缓冲区。
 *       前 signalLength 个点为有效 I 数据。
 *
 *   qData
 *       长度至少为 16384 的 float 缓冲区。
 *       前 signalLength 个点为有效 Q 数据。
 *
 *   signalLength
 *       实际有效 IQ 点数。
 *       当前要求：
 *
 *           32 <= signalLength <= 16384
 *
 *   sampleRate
 *       采样率，单位 Hz。
 *
 * Output:
 *
 *   features
 *       长度至少为 64 的 float 缓冲区。
 *
 * Return:
 *
 *    0   成功
 *   -1   signalLength 非法
 *   -2   sampleRate 非法
 *   -3   输入存在 NaN / Inf
 *   其他值保留给以后扩展。
 */
IQ_FEATURE_API int32_t iqFeatureExtract(
    const float *iData,
    const float *qData,
    int32_t signalLength,
    double sampleRate,
    float *features
);


/*
 * 释放 MATLAB Coder 生成代码使用的内部资源。
 *
 * 程序结束前调用一次。
 */
IQ_FEATURE_API void iqFeatureTerminate(void);


#ifdef __cplusplus
}
#endif

#endif

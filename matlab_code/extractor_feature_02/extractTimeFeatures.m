function features = extractTimeFeatures(x)
%#codegen
% EXTRACTTIMEFEATURES
% 从单个复数 IQ 段中提取 12 个时域和统计特征。
%
% Input:
%   x        复数 IQ 向量。输入被视为一个完整的段。
%
% Output:
%   features 1×12 单精度特征向量。
%
% Feature order:
%    1  RMS                     均方根
%    2  Standard deviation      标准差
%    3  PAPR_dB                 峰均功率比
%    4  Amplitude skewness      幅度偏度
%    5  Amplitude kurtosis      幅度峰度  
%    6  Envelope CV             包络变异系数
%    7  Normalized Amplitude  entropy      归一化幅度熵
%    8  Normalized C20          归一化 C20
%    9  Normalized C40          归一化 C40
%   10  Normalized C41          归一化 C41
%   11  Normalized C42          归一化 C42
%   12  Differential phase standard deviation  差分相位标准差  
%
%
%注意事项
%   1   删去Clearance factor裕度因子，Crest factor峰值因子，同样是描述峰值突出程度，与PARP重合度太高  
%   2   删去Impulse factor，脉冲因子，与当前业务契合度不够




%% 1. 输入

xIn = single(x(:));


epsilon = eps('single');

%% 2. 去除直流分量

x0 = xIn - mean(xIn);
absX0 = abs(x0);

%% 3. 基本幅度和峰值相关特征

%IQ信号的均方根
fRMS = rms(x0);

% 标准差，可以用于描述信号幅度的整体波动程度。
fStandardDeviation = std(x0);

maximumAmplitude = max(absX0);
meanAmplitude = mean(absX0);
meanPower = mean(absX0.^2);

% 峰均功率比：
%
%   PAPR = max(|x|^2) / mean(|x|^2)
%
% 使用 dB 表示，更符合通信信号分析中的常见定义。

if meanPower <= epsilon
    fPAPRdB = single(0);
else
    paprLinear = ...
        maximumAmplitude.^2 / ...
        (meanPower + epsilon);

    fPAPRdB = single(10) * log10(paprLinear + epsilon);
end

% 包络变异系数：
%
%   Envelope CV = std(|x|) / mean(|x|)
%
% 该特征不反映绝对幅度大小，而是描述包络相对波动程度。

if meanAmplitude <= epsilon
    fEnvelopeCV = single(0);
else
    fEnvelopeCV = ...
        fStandardDeviation / ...
        (meanAmplitude + epsilon);
end


%% 4. 幅度偏度和峰度
%
% 
%
% skewness:
%   E[(x - mu)^3] / (E[(x - mu)^2])^(3/2)
%
% kurtosis:
%   E[(x - mu)^4] / (E[(x - mu)^2])^2
%


centeredAmplitude = absX0 - meanAmplitude;

amplitudeSecondMoment = mean(centeredAmplitude.^2);
amplitudeThirdMoment = mean(centeredAmplitude.^3);
amplitudeFourthMoment = mean(centeredAmplitude.^4);

if amplitudeSecondMoment <= epsilon
    fAmplitudeSkewness = single(0);
    fAmplitudeKurtosis = single(0);
else
    fAmplitudeSkewness = ...
        amplitudeThirdMoment / ...
        (amplitudeSecondMoment * sqrt(amplitudeSecondMoment) + epsilon);

    fAmplitudeKurtosis = ...
        amplitudeFourthMoment / ...
        (amplitudeSecondMoment.^2 + epsilon);
end

%% 5. 高阶矩和累积量
%对高阶累积量进行能量归一化，是为了减小接收幅度或增益变化的影响；取模则可以减小未知载波相位带来的影响。
M20 = mean(x0.^2);
M21 = mean(absX0.^2);
M40 = mean(x0.^4);
M41 = mean((x0.^3) .* conj(x0));
M42 = mean(absX0.^4);

powerSquared = M21.^2 + epsilon;

fC20 = abs(M20) / (M21 + epsilon);

fC40 = ...
    abs(M40 - 3 * M20.^2) / powerSquared;

fC41 = ...
    abs(M41 - 3 * M20 * M21) / powerSquared;

fC42 = ...
    abs(M42 - abs(M20).^2 - 2 * M21.^2) / ...
    powerSquared;

%% 6. 归一化幅度熵
%
% 将幅度范围划分为 10 个等宽区间。
%
% 原始香农熵的最大值为：
%
%   log2(10)
%
% 因此，将计算结果除以 log2(10)，使输出大致位于 [0,1]。

numberOfBins = 10;
fAmplitudeEntropy = single(0);

if maximumAmplitude > epsilon

    edges = linspace( ...
        single(0), ...
        maximumAmplitude, ...
        numberOfBins + 1 ...
    );

    counts = zeros(1, numberOfBins, 'single');

    for binIndex = 1:numberOfBins

        if binIndex == numberOfBins
            counts(binIndex) = single(sum( ...
                absX0 >= edges(binIndex) & ...
                absX0 <= edges(binIndex + 1) ...
            ));
        else
            counts(binIndex) = single(sum( ...
                absX0 >= edges(binIndex) & ...
                absX0 < edges(binIndex + 1) ...
            ));
        end
    end

    totalCount = sum(counts);

    if totalCount > 0

        for binIndex = 1:numberOfBins

            probability = ...
                counts(binIndex) / totalCount;

            if probability > 0
                fAmplitudeEntropy = ...
                    fAmplitudeEntropy - ...
                    probability * log2(probability);
            end
        end

        fAmplitudeEntropy = ...
            fAmplitudeEntropy / ...
            log2(single(numberOfBins));
    end
end

%% 7. 差分相位标准差
%
% Differential phase:
%
%   deltaPhi[n] = angle(x[n] * conj(x[n-1]))
%
% 这种计算方式可以消除恒定初始相位的影响。
%
% 这里不对整段相位进行 unwrap，因为差分相位本身已经通过 angle()
% 限制在 [-pi, pi] 内。该特征仍可能受到载波频偏以及低幅度噪声
% 样本的影响。

if numel(x0) >= 2

    differentialPhase = angle( ...
        x0(2:end) .* conj(x0(1:end-1)) ...
    );

    fDifferentialPhaseStandardDeviation = ...
        std(differentialPhase);
else
    fDifferentialPhaseStandardDeviation = single(0);
end


%% 8. 固定尺寸输出

features = zeros(1, 12, 'single');

features(1)  = single(fRMS);
features(2)  = single(fStandardDeviation);
features(3)  = single(fPAPRdB);
features(4)  = single(fAmplitudeSkewness);
features(5)  = single(fAmplitudeKurtosis);
features(6)  = single(fEnvelopeCV);
features(7)  = single(fAmplitudeEntropy);
features(8)  = single(fC20);
features(9)  = single(fC40);
features(10) = single(fC41);
features(11) = single(fC42);
features(12) = single(fDifferentialPhaseStandardDeviation);


%% 9. 数值保护

for featureIndex = 1:numel(features)

    if isnan(features(featureIndex)) || ...
            isinf(features(featureIndex))

        features(featureIndex) = single(0);
    end
end

end
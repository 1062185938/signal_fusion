function features = extractTimeFrequencyFeatures(x, sampleRate)
%#codegen
% EXTRACTTIMEFREQUENCYFEATURES
% 从一段 IQ 信号中提取 28 个时频和帧活动特征
% Input:
%   x           复数 IQ 向量。输入被视为一个完整的段。
%
%   sampleRate  采样率
%
% Output:
%   features    1-by-28 的单精度特征向量
%
% Feature order:
%    1  SpectralKurtosisMean 谱峰度均值
%    2  SpectralKurtosisStd 谱峰度标准差
%    3  SpectralSkewnessMean 谱偏度均值
%    4  SpectralSkewnessStd 谱偏度标准差
%    5  SpectralCrestMean 谱峰因子均值
%    6  SpectralCrestStd 谱峰因子标准差
%    7  SpectralFlatnessMean 谱平坦度均值
%    8  SpectralFlatnessStd 谱平坦度标准差
%    9  SpectralEntropyMean 谱熵均值
%   10  SpectralEntropyStd 谱熵标准差
%   11  NormalizedSTFTRidgeFrequencyStd STFT归一化脊频率标准差
%   12  NormalizedSTFTRidgeSlope STFT归一化脊频率斜率
%   13  NormalizedSpectralCentroidStd  归一化谱质心标准差
%   14  NormalizedSpectralSpreadMean  归一化频谱扩展均值
%   15  NormalizedSpectralSpreadStd  归一化频谱扩展标准差
%   16  STFTFrameEnergyCV            STFT帧能量变异系数
%   17  STFTActiveFrameRatio         STFT活动帧比例
%   18  STFTNormalizedEnergyTransitionCount  STFT能量状态跃迁比例
%
%   19  NormalizedFSSTRidgeFrequencyStd
%                                      FSST归一化脊频率标准差
%   20  NormalizedFSSTRidgeSlope
%                                      FSST归一化脊频率斜率
%   21  NormalizedFSSTRidgeCurvatureRMS
%                                      FSST归一化脊线曲率RMS
%   22  FSSTRidgeEnergyRatio
%                                      FSST脊线能量占比
%   23  NormalizedFSSTTimeFrequencyEntropy
%                                      FSST归一化全局时频熵
%
%   24  NormalizedWSSTRidgeFrequencyStd
%                                      WSST归一化脊频率标准差
%   25  NormalizedWSSTRidgeSlope
%                                      WSST归一化脊频率斜率
%   26  NormalizedWSSTBandwidthMean
%                                      WSST归一化局部带宽均值
%   27  NormalizedWSSTBandwidthStd
%                                      WSST归一化局部带宽标准差
%   28  WSSTRidgeEnergyRatio
%                                      WSST脊线能量占比





%% 1. 输入特征

xIn = complex( ...
    single(real(x(:))), ...
    single(imag(x(:))) ...
);

signalLength = numel(xIn);

%检验输入长度和采样率检查
minimumSignalLength = 32;
maximumSignalLength = 16384;

if signalLength < minimumSignalLength
    error('INPUT_LENGTH_TOO_SHORT');
end

if signalLength > maximumSignalLength
    error('INPUT_LENGTH_EXCEEDS_16384');
end

if ~isscalar(sampleRate) || ...
        isnan(sampleRate) || isinf(sampleRate) || ...
        sampleRate <= 0
    error('INVALID_SAMPLE_RATE');
end

features = zeros(1, 28, 'single');

tinyValue = single(1.0e-20);
sampleRateSingle = single(sampleRate);

%% 2. 去除直流偏量

x0 = xIn - mean(xIn);

signalEnergy = sum(abs(x0).^2);

% 对于零信号或极弱信号，返回全零的有限特征向量。
if signalEnergy <= tinyValue
    return;
end

%% 3. 配置语谱图
%
% 窗长约为输入长度的四分之一：
%
%   128 个采样点  -> 窗长 32 点
%   512 个采样点  -> 窗长 128 点
%   4096 个采样点 -> 窗长 256 点
%
% 最大窗长限制为 256 点。
% 使用 50% 重叠和固定的 512 点 FFT

windowLength = floor(signalLength / 4);

if windowLength < 32
    windowLength = 32;
elseif windowLength > 256
    windowLength = 256;
end

% 确保窗长不超过信号长度。
if windowLength > signalLength
    windowLength = signalLength;
end


windowLength = 2 * floor(windowLength / 2);

overlapLength = floor(windowLength / 2);
nfft = 512;

analysisWindow = single(hamming(windowLength, 'periodic'));

%% 4. 计算中心化的复数 IQ 语谱图

[stftMatrix, frequencyAxis, ~] = spectrogram( ...
    x0, ...
    analysisWindow, ...
    overlapLength, ...
    nfft, ...
    sampleRate, ...
    'centered' ...
);

% 将 STFT 系数转换为非负的时频功率分布。
powerMatrix = single(abs(stftMatrix).^2);

frequencyAxis = single(frequencyAxis(:));

numFrequencyBins = size(powerMatrix, 1);
numTimeFrames = size(powerMatrix, 2);

if numTimeFrames == 0 || numFrequencyBins < 2
    return;
end

%% 5. 为局部频谱特征分配序列存储空间
%因为要转换成C代码，所以数组的大小不能在运行时动态增长，所以在使用这个数组之前，需要先定下长度

spectralKurtosisSequence = ...
    zeros(numTimeFrames, 1, 'single');

spectralSkewnessSequence = ...
    zeros(numTimeFrames, 1, 'single');

spectralCrestSequence = ...
    zeros(numTimeFrames, 1, 'single');

spectralFlatnessSequence = ...
    zeros(numTimeFrames, 1, 'single');

spectralEntropySequence = ...
    zeros(numTimeFrames, 1, 'single');

ridgeFrequencySequence = ...
    zeros(numTimeFrames, 1, 'single');

spectralCentroidSequence = ...
    zeros(numTimeFrames, 1, 'single');

spectralSpreadSequence = ...
    zeros(numTimeFrames, 1, 'single');

frameEnergySequence = ...
    zeros(numTimeFrames, 1, 'single');

%% 6. 在每个时间帧内计算五种频谱描述子
%
% 此处显式实现描述子，不移动负频率轴。
% 直接使用中心化的物理频率轴进行谱偏度和谱峰度的计算。

for frameIndex = 1:numTimeFrames

    localPower = powerMatrix(:, frameIndex);
    localTotalPower = sum(localPower);

    frameEnergySequence(frameIndex) = ...
        single(localTotalPower);

    if localTotalPower <= tinyValue
        continue;
    end

     %% 6.1 局部频谱质心

    localCentroid = ...
        sum(frequencyAxis .* localPower) / localTotalPower;

    spectralCentroidSequence(frameIndex) = ...
        single(localCentroid);

    frequencyDifference = ...
        frequencyAxis - localCentroid;

    %% 6.2 局部频谱扩展

    localSecondMoment = ...
        sum((frequencyDifference.^2) .* localPower) / ...
        localTotalPower;

    localSecondMoment = ...
        max(localSecondMoment, single(0));

    localSpread = sqrt(localSecondMoment);

    spectralSpreadSequence(frameIndex) = ...
        single(localSpread);

    %% 6.3 谱偏度和谱峰度
    if localSecondMoment > tinyValue

        localThirdMoment = ...
            sum((frequencyDifference.^3) .* localPower) / ...
            localTotalPower;

        localFourthMoment = ...
            sum((frequencyDifference.^4) .* localPower) / ...
            localTotalPower;

        spectralSkewnessSequence(frameIndex) = ...
            localThirdMoment / ...
            (localSpread.^3 + tinyValue);


        spectralKurtosisSequence(frameIndex) = ...
            localFourthMoment / ...
            (localSecondMoment.^2 + tinyValue);
    end

    %% 6.4 谱峰因子

    localMeanPower = ...
        localTotalPower / single(numFrequencyBins);

    localMaximumPower = max(localPower);

    spectralCrestSequence(frameIndex) = ...
        localMaximumPower / ...
        (localMeanPower + tinyValue);

    %% 6.5 谱平坦度
    %

    localPowerFloor = max( ...
        localMeanPower * single(1.0e-12), ...
        tinyValue ...
    );

    limitedLocalPower = ...
        max(localPower, localPowerFloor);

    localGeometricMean = ...
        exp(mean(log(limitedLocalPower)));

    spectralFlatnessSequence(frameIndex) = ...
        localGeometricMean / ...
        (localMeanPower + localPowerFloor);

    %% 6.6 归一化谱熵

    probabilitySpectrum = ...
        localPower / localTotalPower;

    entropyValue = -sum( ...
        probabilitySpectrum .* ...
        log(probabilitySpectrum + tinyValue) ...
    );

    entropyNormalizer = ...
        log(single(numFrequencyBins));

    if entropyNormalizer > 0
        spectralEntropySequence(frameIndex) = ...
            entropyValue / entropyNormalizer;
    end

    [~, localPeakIndex] = max(localPower);

    ridgeFrequencySequence(frameIndex) = ...
        frequencyAxis(localPeakIndex);
end

%% 7. 局部频谱描述子的时间统计量

[kurtosisMean, kurtosisStd] = ...
    localMeanAndStd(spectralKurtosisSequence);

[skewnessMean, skewnessStd] = ...
    localMeanAndStd(spectralSkewnessSequence);

[crestMean, crestStd] = ...
    localMeanAndStd(spectralCrestSequence);

[flatnessMean, flatnessStd] = ...
    localMeanAndStd(spectralFlatnessSequence);

[entropyMean, entropyStd] = ...
    localMeanAndStd(spectralEntropySequence);

%% 7.1 STFT 帧活动特征
%
% 帧能量 CV 描述局部能量起伏；活动帧定义为能量不低于全段最大
% 帧能量的 10%。活动比例和状态跃迁比例用于描述连续占用与突发性。

[frameEnergyMean, frameEnergyStd] = ...
    localMeanAndStd(frameEnergySequence);

stftFrameEnergyCV = single(0);
stftActiveFrameRatio = single(0);
stftNormalizedEnergyTransitionCount = single(0);

if frameEnergyMean > tinyValue
    stftFrameEnergyCV = ...
        frameEnergyStd / (frameEnergyMean + tinyValue);
end

maximumFrameEnergy = max(frameEnergySequence);

if maximumFrameEnergy > tinyValue
    activeThreshold = single(0.1) * maximumFrameEnergy;
    activeFrameFlags = frameEnergySequence >= activeThreshold;

    stftActiveFrameRatio = ...
        sum(single(activeFrameFlags)) / single(numTimeFrames);

    if numTimeFrames >= 2
        transitionCount = single(0);

        for frameIndex = 2:numTimeFrames
            if activeFrameFlags(frameIndex) ~= ...
                    activeFrameFlags(frameIndex - 1)
                transitionCount = transitionCount + single(1);
            end
        end

        stftNormalizedEnergyTransitionCount = ...
            transitionCount / single(numTimeFrames - 1);
    end
end

%% 8. 提取主导时频脊线

% Use the peak-power frequency in each STFT frame as a deterministic ridge.
normalizedRidgeFrequency = ...
    ridgeFrequencySequence / sampleRateSingle;

[~, normalizedstftRidgeStd] = ...
    localMeanAndStd(normalizedRidgeFrequency);

%% 9. 计算归一化脊频率斜率
%
% 时间轴归一化到 [0,1]，因此斜率表示
% 在整个 IQ 片段持续时间内归一化频率的变化量。

normalizedstftRidgeSlope = single(0);
numRidgePoints = numel(normalizedRidgeFrequency);

if numRidgePoints >= 2

    normalizedTime = ...
        single((0:numRidgePoints-1).') / ...
        single(numRidgePoints-1);

    timeMean = mean(normalizedTime);
    ridgeMean = mean(normalizedRidgeFrequency);

    centeredTime = normalizedTime - timeMean;
    centeredRidge = ...
        normalizedRidgeFrequency - ridgeMean;

    slopeDenominator = ...
        sum(centeredTime.^2);

    if slopeDenominator > tinyValue
        normalizedstftRidgeSlope = ...
            sum(centeredTime .* centeredRidge) / ...
            slopeDenominator;
    end
end

%% 10. 瞬时频率
%
% instfreq 计算时频功率分布的第一条件谱矩。

% Use the power-weighted frequency centroid of each STFT frame.
normalizedSpectralCentroid = ...
    spectralCentroidSequence / sampleRateSingle;

[~, NormalizedSpectralCentroidStd] = ...
    localMeanAndStd(normalizedSpectralCentroid);

%% 11. 瞬时带宽
%
% ScaleFactor = 1 使结果等于频谱标准差，
% 而不使用 MATLAB 默认的额外比例因子。

% Use the power-weighted spectral spread of each STFT frame.
normalizedSpectralSpread = ...
    spectralSpreadSequence / sampleRateSingle;

[NormalizedSpectralSpreadMean, ...
 NormalizedSpectralSpreadStd] = ...
    localMeanAndStd(normalizedSpectralSpread);


%% 12. FSST 时频特征
%
% Fourier Synchrosqueezed Transform
%
% FSST 可以直接处理复数 IQ 信号。
% 使用与 Spectrogram 相同的分析窗长度，使两种时频表示
% 的基本时间尺度保持一致。
%
% MATLAB Coder 要求 FSST 的分析窗为 double 类型。

normalizedFSSTRidgeSlope = single(0);
normalizedFSSTRidgeCurvatureRMS = single(0);
fsstRidgeEnergyRatio = single(0);
normalizedFSSTTimeFrequencyEntropy = single(0);


%% 12.1 计算 FSST

fsstWindow = double( ...
    hamming(windowLength, 'periodic') ...
);

[fsstMatrix, fsstFrequencyAxis, ~] = fsst( ...
    x0, ...
    sampleRate, ...
    fsstWindow ...
);

fsstPowerMatrix = ...
    single(abs(fsstMatrix).^2);

fsstFrequencyAxis = ...
    single(fsstFrequencyAxis(:));

numFSSTFrequencyBins = ...
    size(fsstPowerMatrix, 1);

numFSSTTimeFrames = ...
    size(fsstPowerMatrix, 2);

fsstTotalEnergy = ...
    sum(fsstPowerMatrix(:));


%% 12.2 提取 FSST 主脊线
%
% 每个时间位置选取功率最大的频率作为主脊线。
%
% 同时统计主脊线所在频率点及其上下各一个频率 bin 的能量，
% 用于计算 Ridge Energy Ratio。

fsstRidgeFrequencySequence = ...
    zeros(numFSSTTimeFrames, 1, 'single');

fsstRidgeEnergy = single(0);

if fsstTotalEnergy > tinyValue

    for frameIndex = 1:numFSSTTimeFrames

        localPower = ...
            fsstPowerMatrix(:, frameIndex);

        localTotalPower = ...
            sum(localPower);

        if localTotalPower <= tinyValue
            continue;
        end

        [~, ridgeIndex] = ...
            max(localPower);

        fsstRidgeFrequencySequence(frameIndex) = ...
            fsstFrequencyAxis(ridgeIndex);


        % 主脊线上下各取一个相邻频率 bin。
        lowerRidgeIndex = ...
            max(ridgeIndex - 1, 1);

        upperRidgeIndex = ...
            min( ...
                ridgeIndex + 1, ...
                numFSSTFrequencyBins ...
            );

        fsstRidgeEnergy = ...
            fsstRidgeEnergy + ...
            sum( ...
                localPower( ...
                    lowerRidgeIndex:upperRidgeIndex ...
                ) ...
            );

    end
end


%% 12.3 FSST 归一化脊频率标准差

normalizedFSSTRidgeFrequency = ...
    fsstRidgeFrequencySequence / ...
    sampleRateSingle;

[~, normalizedFSSTRidgeFrequencyStd] = ...
    localMeanAndStd( ...
        normalizedFSSTRidgeFrequency ...
    );


%% 12.4 FSST 归一化脊频率斜率
%
% 时间轴归一化到 [0,1]。
%
% 正斜率：
%   主频总体向高频方向移动
%
% 负斜率：
%   主频总体向低频方向移动

if numFSSTTimeFrames >= 2

    normalizedFSSTTime = ...
        single((0:numFSSTTimeFrames-1).') / ...
        single(numFSSTTimeFrames-1);

    fsstTimeMean = ...
        mean(normalizedFSSTTime);

    fsstRidgeMean = ...
        mean(normalizedFSSTRidgeFrequency);

    centeredFSSTTime = ...
        normalizedFSSTTime - fsstTimeMean;

    centeredFSSTRidge = ...
        normalizedFSSTRidgeFrequency - ...
        fsstRidgeMean;

    fsstSlopeDenominator = ...
        sum(centeredFSSTTime.^2);

    if fsstSlopeDenominator > tinyValue

        normalizedFSSTRidgeSlope = ...
            sum( ...
                centeredFSSTTime .* ...
                centeredFSSTRidge ...
            ) / ...
            fsstSlopeDenominator;

    end
end


%% 12.5 FSST 脊线曲率 RMS
%
% 使用归一化频率脊线的离散二阶差分：
%
%   f[n+1] - 2*f[n] + f[n-1]
%
% 表示脊线偏离恒定斜率变化的程度。
%
% 数值较小：
%   脊线更接近直线或平稳频率
%
% 数值较大：
%   脊线存在弯曲、跳变或扫频速率变化

if numFSSTTimeFrames >= 3

    fsstRidgeSecondDifference = ...
        normalizedFSSTRidgeFrequency(3:end) - ...
        single(2) * ...
        normalizedFSSTRidgeFrequency(2:end-1) + ...
        normalizedFSSTRidgeFrequency(1:end-2);

    normalizedFSSTRidgeCurvatureRMS = ...
        sqrt( ...
            mean( ...
                fsstRidgeSecondDifference.^2 ...
            ) ...
        );

end


%% 12.6 FSST 主脊线能量占比
%
% 描述 FSST 能量是否集中在主要时频轨迹附近。

if fsstTotalEnergy > tinyValue

    fsstRidgeEnergyRatio = ...
        fsstRidgeEnergy / ...
        (fsstTotalEnergy + tinyValue);

end


%% 12.7 FSST 全局归一化时频熵
%
% 将整张 FSST 时频功率图归一化为概率分布：
%
%   p(i,j) = P(i,j) / sum(P)
%
% 再计算整张时频图的归一化 Shannon entropy。
%
% 较小：
%   能量集中在较少时频位置
%
% 较大：
%   时频能量分布更分散、更复杂

if fsstTotalEnergy > tinyValue

    fsstProbabilityMatrix = ...
        fsstPowerMatrix / ...
        fsstTotalEnergy;

    fsstEntropyValue = ...
        -sum( ...
            fsstProbabilityMatrix(:) .* ...
            log( ...
                fsstProbabilityMatrix(:) + ...
                tinyValue ...
            ) ...
        );

    fsstEntropyNormalizer = ...
        log( ...
            single(numel(fsstPowerMatrix)) ...
        );

    if fsstEntropyNormalizer > 0

        normalizedFSSTTimeFrequencyEntropy = ...
            fsstEntropyValue / ...
            fsstEntropyNormalizer;

    end
end





%% 13. WSST 时频特征
%
% Wavelet Synchrosqueezed Transform
%
% MATLAB 的 wsst() 只支持实数输入。
%
% 对复数 IQ 信号分别计算：
%
%   WSST(I)
%   WSST(Q)
%
% 然后构造联合时频功率：
%
%   Pwsst = |WSST(I)|^2 + |WSST(Q)|^2
%
% WSST 内部使用 double，以提高 MATLAB 与生成 C/C++
% 结果之间的一致性。

normalizedWSSTRidgeSlope = single(0);
wsstRidgeEnergyRatio = single(0);


%% 13.1 分别计算 I/Q 两路 WSST

wsstInputI = ...
    double(real(x0));

wsstInputQ = ...
    double(imag(x0));


% 'amor' 为固定的解析 Morlet 小波。
% 字面量在代码生成时作为编译期常量。

[wsstMatrixI, wsstFrequencyAxis] = ...
    wsst( ...
        wsstInputI, ...
        double(sampleRate), ...
        'amor' ...
    );

[wsstMatrixQ, ~] = ...
    wsst( ...
        wsstInputQ, ...
        double(sampleRate), ...
        'amor' ...
    );


%% 13.2 构造 I/Q 联合 WSST 功率图

wsstPowerMatrix = single( ...
    abs(wsstMatrixI).^2 + ...
    abs(wsstMatrixQ).^2 ...
);

wsstFrequencyAxis = ...
    single(wsstFrequencyAxis(:));

numWSSTFrequencyBins = ...
    size(wsstPowerMatrix, 1);

numWSSTTimeFrames = ...
    size(wsstPowerMatrix, 2);

wsstTotalEnergy = ...
    sum(wsstPowerMatrix(:));


%% 13.3 分配 WSST 局部特征序列

wsstRidgeFrequencySequence = ...
    zeros(numWSSTTimeFrames, 1, 'single');

normalizedWSSTBandwidthSequence = ...
    zeros(numWSSTTimeFrames, 1, 'single');

wsstRidgeEnergy = single(0);


%% 13.4 每个时间位置提取脊线和局部带宽

if wsstTotalEnergy > tinyValue

    for frameIndex = 1:numWSSTTimeFrames

        localPower = ...
            wsstPowerMatrix(:, frameIndex);

        localTotalPower = ...
            sum(localPower);

        if localTotalPower <= tinyValue
            continue;
        end


        %% WSST 主脊频率

        [~, ridgeIndex] = ...
            max(localPower);

        wsstRidgeFrequencySequence(frameIndex) = ...
            wsstFrequencyAxis(ridgeIndex);


        %% WSST 主脊线邻域能量

        lowerRidgeIndex = ...
            max(ridgeIndex - 1, 1);

        upperRidgeIndex = ...
            min( ...
                ridgeIndex + 1, ...
                numWSSTFrequencyBins ...
            );

        wsstRidgeEnergy = ...
            wsstRidgeEnergy + ...
            sum( ...
                localPower( ...
                    lowerRidgeIndex:upperRidgeIndex ...
                ) ...
            );


        %% WSST 局部频谱质心

        localCentroid = ...
            sum( ...
                wsstFrequencyAxis .* ...
                localPower ...
            ) / ...
            localTotalPower;


        %% WSST 局部频率扩展 / 带宽

        frequencyDifference = ...
            wsstFrequencyAxis - ...
            localCentroid;

        localSecondMoment = ...
            sum( ...
                (frequencyDifference.^2) .* ...
                localPower ...
            ) / ...
            localTotalPower;

        localSecondMoment = ...
            max( ...
                localSecondMoment, ...
                single(0) ...
            );

        localBandwidth = ...
            sqrt(localSecondMoment);

        normalizedWSSTBandwidthSequence(frameIndex) = ...
            localBandwidth / ...
            sampleRateSingle;

    end
end


%% 13.5 WSST 归一化脊频率标准差

normalizedWSSTRidgeFrequency = ...
    wsstRidgeFrequencySequence / ...
    sampleRateSingle;

[~, normalizedWSSTRidgeFrequencyStd] = ...
    localMeanAndStd( ...
        normalizedWSSTRidgeFrequency ...
    );


%% 13.6 WSST 归一化脊频率斜率

if numWSSTTimeFrames >= 2

    normalizedWSSTTime = ...
        single((0:numWSSTTimeFrames-1).') / ...
        single(numWSSTTimeFrames-1);

    wsstTimeMean = ...
        mean(normalizedWSSTTime);

    wsstRidgeMean = ...
        mean(normalizedWSSTRidgeFrequency);

    centeredWSSTTime = ...
        normalizedWSSTTime - ...
        wsstTimeMean;

    centeredWSSTRidge = ...
        normalizedWSSTRidgeFrequency - ...
        wsstRidgeMean;

    wsstSlopeDenominator = ...
        sum(centeredWSSTTime.^2);

    if wsstSlopeDenominator > tinyValue

        normalizedWSSTRidgeSlope = ...
            sum( ...
                centeredWSSTTime .* ...
                centeredWSSTRidge ...
            ) / ...
            wsstSlopeDenominator;

    end
end


%% 13.7 WSST 归一化局部带宽均值和标准差

[normalizedWSSTBandwidthMean, ...
 normalizedWSSTBandwidthStd] = ...
    localMeanAndStd( ...
        normalizedWSSTBandwidthSequence ...
    );


%% 13.8 WSST 主脊线能量占比

if wsstTotalEnergy > tinyValue

    wsstRidgeEnergyRatio = ...
        wsstRidgeEnergy / ...
        (wsstTotalEnergy + tinyValue);

end














%% 14. 组装28维特征

features(1)  = kurtosisMean;
features(2)  = kurtosisStd;

features(3)  = skewnessMean;
features(4)  = skewnessStd;

features(5)  = crestMean;
features(6)  = crestStd;

features(7)  = flatnessMean;
features(8)  = flatnessStd;

features(9)  = entropyMean;
features(10) = entropyStd;

features(11) = normalizedstftRidgeStd;
features(12) = normalizedstftRidgeSlope;
features(13) = NormalizedSpectralCentroidStd;

features(14) = NormalizedSpectralSpreadMean;
features(15) = NormalizedSpectralSpreadStd;

features(16) = stftFrameEnergyCV;
features(17) = stftActiveFrameRatio;
features(18) = stftNormalizedEnergyTransitionCount;

%% FSST

features(19) = ...
    normalizedFSSTRidgeFrequencyStd;

features(20) = ...
    normalizedFSSTRidgeSlope;

features(21) = ...
    normalizedFSSTRidgeCurvatureRMS;

features(22) = ...
    fsstRidgeEnergyRatio;

features(23) = ...
    normalizedFSSTTimeFrequencyEntropy;


%% WSST

features(24) = ...
    normalizedWSSTRidgeFrequencyStd;

features(25) = ...
    normalizedWSSTRidgeSlope;

features(26) = ...
    normalizedWSSTBandwidthMean;

features(27) = ...
    normalizedWSSTBandwidthStd;

features(28) = ...
    wsstRidgeEnergyRatio;






features = localReplaceNonfinite(features);

end


% =========================================================================
% 局部函数：均值和总体标准差
% =========================================================================
function [meanValue, standardDeviation] = localMeanAndStd(values)

values = values(:);

if isempty(values)
    meanValue = single(0);
    standardDeviation = single(0);
    return;
end

meanValue = mean(values);

centeredValues = values - meanValue;

varianceValue = mean(centeredValues.^2);
varianceValue = max(varianceValue, single(0));

standardDeviation = sqrt(varianceValue);

end


% =========================================================================
% 局部函数：将 NaN 和 Inf 替换为零
% =========================================================================
function values = localReplaceNonfinite(values)

for index = 1:numel(values)
    if isnan(values(index)) || isinf(values(index))
        values(index) = single(0);
    end
end

end

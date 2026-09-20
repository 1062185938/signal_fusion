function features = extractFrequencyFeatures(x, sampleRate)
%#codegen
% EXTRACTFREQUENCYFEATURES
% 从单个复数 IQ 段中提取 21 个频域统计特征。
%
% Input:
%   x           复数 IQ 向量。输入被视为一个完整的段。
%   sampleRate  采样率
%
% Output:
%   features    1-by-21 单精度特征向量
%
% Feature order:
%    1  Normalized mean frequency       归一化平均频率                                   
%    2  Normalized median frequency     归一化中值频率                                      
%    3  Normalized spectral spread      归一化频谱扩展
%    4  Normalized 90% occupied bandwidth       归一化 90% 占用带宽
%    5  Normalized peak frequency       归一化峰值频率
%    6  Log10 band power                对数带内功率
%    7  Global normalized spectral entropy      全局归一化频谱熵
%    8  Global spectral crest factor    全局频谱峰值因子
%    9  Global spectral flatness        全局频谱平坦度
%   10  Normalized spectral skewness    归一化频谱偏度
%   11  PSD standard deviation          PSD 标准差
%   12  Log mean bispectrum magnitude
%                                      对数平均双谱幅值
%   13  Log maximum bispectrum magnitude
%                                      对数最大双谱幅值
%   14  Log bispectral energy
%                                      对数双谱能量
%   15  Normalized bispectral entropy
%                                      归一化双谱熵
%   16  Normalized bispectrum peak frequency 1
%                                      双谱峰值频率 1
%   17  Normalized bispectrum peak frequency 2
%                                      双谱峰值频率 2
%   18  Mean squared bicoherence
%                                      平均平方双相干
%   19  Maximum squared bicoherence
%                                      最大平方双相干
%   20  Normalized bicoherence entropy
%                                      归一化双相干熵
%   21  Bicoherence-weighted biphase concentration
%                                      双相干加权双相位集中度

% Frequency normalization:
%   normalizedFrequency = frequencyHz / sampleRate
%
% 注意：
%   - 进行PSD之前会去除直流分量
%   - Welch 分段仅用于PSD估计
%   - 输入最大长度限制为 16384 点
%   - Welch 最大窗长为 1024 点
%   - FFT 点数固定为 4096
%   - PSD standard deviation 保留原始 PSD 数值尺度 


%% 1. 输入验证

xIn = complex( ...
    single(real(x(:))), ...
    single(imag(x(:))) ...
);

signalLength = numel(xIn);
minimumSignalLength = 32;
maximumSignalLength = 16384;

if signalLength < minimumSignalLength
    error('INPUT_LENGTH_TOO_SHORT');
end

if signalLength > maximumSignalLength
    error('INPUT_LENGTH_EXCEEDS_16384');
end

if any(isnan(real(xIn))) || any(isnan(imag(xIn))) || ...
        any(isinf(real(xIn))) || any(isinf(imag(xIn)))
    error('INPUT_CONTAINS_NAN_OR_INF');
end

if ~isscalar(sampleRate) || ...
        isnan(sampleRate) || isinf(sampleRate) || ...
        sampleRate <= 0
    error('INVALID_SAMPLE_RATE');
end

tinyValue = single(1.0e-20);

%% 2. 去除直流分量

x0 = xIn - mean(xIn);

%% 3.  Welch PSD 配置
%
% 使用最大长度为 1024 点的汉明窗。
% 输入长度更短时则使用整个输入长度作为窗长。
%
% FFT 点数固定为 4096
% 输入长度增加时，Welch 可使用更多数据段进行平均

maximumWindowLength = 1024;
nfft = 4096;

windowLength = min(signalLength, maximumWindowLength);
overlapLength = floor(windowLength / 2);

window = single(hamming(windowLength, 'periodic'));

%% 4. 计算中心化双边 Welch PSD

[powerSpectrum, frequencyAxis] = pwelch( ...
    x0, ...
    window, ...
    overlapLength, ...
    nfft, ...
    sampleRate, ...
    'centered' ...
);

powerSpectrum = single(real(powerSpectrum));
frequencyAxis = single(frequencyAxis(:));
powerSpectrum = powerSpectrum(:);


%避免由于数值误差产生极小负值
powerSpectrum = max(powerSpectrum, single(0));

totalSpectralWeight = sum(powerSpectrum);

%% 5. 处理零信号或极弱信号

features = zeros(1, 21, 'single');

if totalSpectralWeight <= tinyValue
    features(6) = log10(tinyValue);
    return;
end

%% 6. 归一化 PSD
%
% 将 PSD 转换为总和为 1 的功率权重。
%
% normalizedPowerSpectrum 用于计算：
%   - 频谱熵
%   - 频谱偏度
%
% 这样这些统计量不会受到信号整体功率尺度的直接影响。

normalizedPowerSpectrum = ...
    powerSpectrum / totalSpectralWeight;

%% 6. 平均频率 / 频谱质心

meanFrequency = ...
    sum(frequencyAxis .* powerSpectrum) / ...
    totalSpectralWeight;

%% 7. 中值频率

normalizedCumulativePower = ...
    cumsum(powerSpectrum) / totalSpectralWeight;

medianIndex = 1;

for index = 1:numel(normalizedCumulativePower)
    if normalizedCumulativePower(index) >= single(0.5)
        medianIndex = index;
        break;
    end
end

medianFrequency = frequencyAxis(medianIndex);

%% 8. 频谱扩展
%
% 以平均频率为中心的功率加权标准差。

frequencyDeviation = frequencyAxis - meanFrequency;

spectralVariance = ...
    sum((frequencyDeviation.^2) .* powerSpectrum) / ...
    totalSpectralWeight;

spectralSpread = sqrt(max(spectralVariance, single(0)));

%% 9. 90% 占用带宽
%
% 从累计功率 5% 到 95% 之间的频率范围，
% 对应总功率的 90%。相比 99% 带宽，它对噪声底和边缘泄漏更稳健。

lowerThreshold = single(0.05);
upperThreshold = single(0.95);

lowerIndex = 1;
upperIndex = numel(normalizedCumulativePower);

for index = 1:numel(normalizedCumulativePower)
    if normalizedCumulativePower(index) >= lowerThreshold
        lowerIndex = index;
        break;
    end
end

for index = 1:numel(normalizedCumulativePower)
    if normalizedCumulativePower(index) >= upperThreshold
        upperIndex = index;
        break;
    end
end

occupiedBandwidth = ...
    frequencyAxis(upperIndex) - frequencyAxis(lowerIndex);

occupiedBandwidth = max(occupiedBandwidth, single(0));

%% 10. 峰值频率

[~, peakIndex] = max(powerSpectrum);
peakFrequency = frequencyAxis(peakIndex);

%% 11. 带内功率
%
% 对频率轴上的 PSD 进行积分
frequencyResolution = abs( ...
    frequencyAxis(2) - frequencyAxis(1) ...
);

bandPower = sum(powerSpectrum) * frequencyResolution;

logBandPower = log10(max(bandPower, tinyValue));


%% 13. 全局归一化频谱熵
%
% Spectral entropy:
%
%   H = -sum(p(k) * log2(p(k))) / log2(K)
%
% p(k) 为归一化 PSD。
%
% 输出范围大致位于 [0, 1]：
%
%   较小：频谱能量集中
%   较大：频谱能量分散

spectralEntropy = single(0);

numberOfFrequencyBins = numel(normalizedPowerSpectrum);

for index = 1:numberOfFrequencyBins

    probability = ...
        normalizedPowerSpectrum(index);

    if probability > 0

        spectralEntropy = ...
            spectralEntropy - ...
            probability * log2(probability);

    end
end

spectralEntropy = ...
    spectralEntropy / ...
    log2(single(numberOfFrequencyBins));


%% 14. 全局频谱峰值因子
%
% Spectral crest factor:
%
%   max(PSD) / mean(PSD)
%
% 描述最大谱峰相对于平均谱水平的突出程度。

meanPSD = mean(powerSpectrum);

if meanPSD <= tinyValue
    spectralCrestFactor = single(0);
else

    spectralCrestFactor = ...
        max(powerSpectrum) / ...
        (meanPSD + tinyValue);

end


%% 15. 全局频谱平坦度
%
% Spectral flatness:
%
%   geometric mean(PSD) / arithmetic mean(PSD)
%
% 输出范围通常位于 [0, 1]：
%
%   接近 0：频谱存在明显峰值
%   接近 1：频谱较为平坦
%
% 使用与平均 PSD 成比例的最小值，避免 log(0)。

spectralFloor = ...
    max( ...
        meanPSD * single(1.0e-12), ...
        tinyValue ...
    );

logSpectrumSum = single(0);

for index = 1:numel(powerSpectrum)

    logSpectrumSum = ...
        logSpectrumSum + ...
        log(max(powerSpectrum(index), spectralFloor));

end

geometricMeanPSD = ...
    exp( ...
        logSpectrumSum / ...
        single(numel(powerSpectrum)) ...
    );

if meanPSD <= tinyValue
    spectralFlatness = single(0);
else

    spectralFlatness = ...
        geometricMeanPSD / ...
        (meanPSD + tinyValue);

end


%% 16. 归一化频谱偏度
%
% Spectral skewness:
%
%   sum(p(k) * (f(k)-meanFrequency)^3)
%   ----------------------------------
%              spectralSpread^3
%
% 该统计量本身是无量纲的，因此不需要再除以 sampleRate。
%
%   接近 0：频谱相对对称
%   > 0：向正频率侧偏
%   < 0：向负频率侧偏

if spectralSpread <= tinyValue

    spectralSkewness = single(0);

else

    spectralThirdMoment = ...
        sum( ...
            (frequencyDeviation.^3) .* ...
            normalizedPowerSpectrum ...
        );

    spectralSkewness = ...
        spectralThirdMoment / ...
        (spectralSpread.^3 + tinyValue);

end


%% 17. PSD 标准差
%
% 直接对 Welch PSD 的频率点数值计算标准差。
%
% 该特征描述 PSD 在不同频率位置上的整体起伏程度。
%
% 与归一化频谱统计不同，本特征保留 PSD 的原始功率尺度，
% 因此会受到信号整体功率和接收增益的影响。


psdStandardDeviation = ...
    std(powerSpectrum);

%% 18. 双谱特征初始化
%
% 双谱用于描述普通功率谱无法反映的三阶频率耦合关系。
%
% Bispectrum:
%
%   B(f1,f2) =
%       E[X(f1) X(f2) conj(X(f1+f2))]
%
% 双谱分析与前面的 Welch PSD 独立进行。
%
% 为控制二维双谱的计算量：
%
%   window length = 256
%   overlap       = 50%
%   FFT length    = 256
%
% 当输入长度小于 512 点时，不计算双谱特征，
% 对应特征全部保持为 0。

fLogMeanBispectrumMagnitude = single(0);
fLogMaximumBispectrumMagnitude = single(0);
fLogBispectralEnergy = single(0);
fNormalizedBispectralEntropy = single(0);

fNormalizedBispectrumPeakFrequency1 = single(0);
fNormalizedBispectrumPeakFrequency2 = single(0);

fMeanSquaredBicoherence = single(0);
fMaximumSquaredBicoherence = single(0);
fNormalizedBicoherenceEntropy = single(0);

fBicoherenceWeightedBiphaseConcentration = single(0);


%% 19. 双谱配置

minimumBispectrumSignalLength = 512;

bispectrumWindowLength = 256;
bispectrumHopLength = 128;
bispectrumNfft = 256;

halfBispectrumNfft = ...
    bispectrumNfft / 2;

% 最大输入长度 16384 点时：
%
% floor((16384 - 256) / 128) + 1 = 127
%
% 因此固定预分配最多 127 个 FFT 子段。

maximumBispectrumSegments = 127;


%% 20. 双谱计算

if signalLength >= minimumBispectrumSignalLength

    numberOfBispectrumSegments = ...
        floor( ...
            (signalLength - bispectrumWindowLength) / ...
            bispectrumHopLength ...
        ) + 1;

    bispectrumWindow = single( ...
        hamming(bispectrumWindowLength, 'periodic') ...
    );

    bispectrumWindowNormalization = ...
        sum(bispectrumWindow);

    % 保存所有子段的中心化 FFT。
    %
    % 频率索引对应：
    %
    %   -Fs/2 ... 0 ... Fs/2
    %
    % 固定尺寸预分配有利于后续 MATLAB Coder。

    segmentSpectra = complex( ...
        zeros( ...
            bispectrumNfft, ...
            maximumBispectrumSegments, ...
            'single' ...
        ), ...
        zeros( ...
            bispectrumNfft, ...
            maximumBispectrumSegments, ...
            'single' ...
        ) ...
    );


    %% 20.1 每个子段计算 FFT

    for segmentIndex = 1:numberOfBispectrumSegments

        startIndex = ...
            (segmentIndex - 1) * ...
            bispectrumHopLength + 1;

        endIndex = ...
            startIndex + ...
            bispectrumWindowLength - 1;

        segment = ...
            x0(startIndex:endIndex);

        % 再次去除当前子段的局部直流分量，
        % 减少零频附近对双谱的影响。

        segment = ...
            segment - mean(segment);

        segment = ...
            segment .* bispectrumWindow;

        segmentSpectrum = ...
            fft(segment, bispectrumNfft);

        % 手动进行 fftshift。
        %
        % 对固定 256 点 FFT：
        %
        %   [129:256, 1:128]

        centeredSegmentSpectrum = [
            segmentSpectrum(halfBispectrumNfft + 1:end);
            segmentSpectrum(1:halfBispectrumNfft)
        ];

        % 对窗函数增益进行简单归一化，
        % 提高不同信号之间原始双谱幅值的可比性。

        centeredSegmentSpectrum = ...
            centeredSegmentSpectrum / ...
            (bispectrumWindowNormalization + tinyValue);

        segmentSpectra(:, segmentIndex) = ...
            centeredSegmentSpectrum;

    end


    %% 20.2 双谱和平方双相干矩阵
    %
    % 对中心化 FFT：
    %
    %   k = -NFFT/2, ..., NFFT/2-1
    %
    % 仅计算满足：
    %
    %   k3 = k1 + k2
    %
    % 且 k3 位于有效 FFT 频率范围内的点。
    %
    % 同时由于：
    %
    %   B(f1,f2) = B(f2,f1)
    %
    % 只计算 index2 >= index1 的区域，
    % 避免完全重复的频率组合。

    bispectrumMatrix = complex( ...
        zeros( ...
            bispectrumNfft, ...
            bispectrumNfft, ...
            'single' ...
        ), ...
        zeros( ...
            bispectrumNfft, ...
            bispectrumNfft, ...
            'single' ...
        ) ...
    );

    squaredBicoherenceMatrix = ...
        zeros( ...
            bispectrumNfft, ...
            bispectrumNfft, ...
            'single' ...
        );


    bispectrumMagnitudeSum = single(0);
    bispectrumEnergy = single(0);
    maximumBispectrumMagnitude = single(0);

    squaredBicoherenceSum = single(0);
    maximumSquaredBicoherence = single(0);

    validBispectrumPointCount = 0;

    peakBispectrumIndex1 = 1;
    peakBispectrumIndex2 = 1;

    for index1 = 1:bispectrumNfft

        frequencyBin1 = ...
            index1 - ...
            halfBispectrumNfft - 1;

        for index2 = index1:bispectrumNfft

            frequencyBin2 = ...
                index2 - ...
                halfBispectrumNfft - 1;

            frequencyBin3 = ...
                frequencyBin1 + ...
                frequencyBin2;

            % f1 + f2 必须仍位于 Nyquist 范围内。

            if frequencyBin3 >= -halfBispectrumNfft && ...
                    frequencyBin3 <= halfBispectrumNfft - 1

                index3 = ...
                    frequencyBin3 + ...
                    halfBispectrumNfft + 1;

                tripleProductSum = ...
                    complex(single(0), single(0));

                pairPowerSum = single(0);
                sumFrequencyPower = single(0);


                %% 对所有数据段平均

                for segmentIndex = 1:numberOfBispectrumSegments

                    spectrum1 = ...
                        segmentSpectra( ...
                            index1, ...
                            segmentIndex ...
                        );

                    spectrum2 = ...
                        segmentSpectra( ...
                            index2, ...
                            segmentIndex ...
                        );

                    spectrum3 = ...
                        segmentSpectra( ...
                            index3, ...
                            segmentIndex ...
                        );

                    pairProduct = ...
                        spectrum1 * spectrum2;

                    tripleProduct = ...
                        pairProduct * conj(spectrum3);

                    tripleProductSum = ...
                        tripleProductSum + ...
                        tripleProduct;

                    pairPowerSum = ...
                        pairPowerSum + ...
                        abs(pairProduct).^2;

                    sumFrequencyPower = ...
                        sumFrequencyPower + ...
                        abs(spectrum3).^2;

                end


                %% 双谱

                bispectrumValue = ...
                    tripleProductSum / ...
                    single(numberOfBispectrumSegments);

                bispectrumMatrix(index1,index2) = ...
                    bispectrumValue;

                bispectrumMagnitude = ...
                    abs(bispectrumValue);


                %% 平方双相干
                %
                % b^2(f1,f2) =
                %
                % |sum X1 X2 conj(X3)|^2
                % ----------------------------------
                % sum|X1 X2|^2 * sum|X3|^2
                %
                % 理论范围为 [0,1]。

                squaredBicoherence = ...
                    abs(tripleProductSum).^2 / ...
                    ( ...
                        pairPowerSum * ...
                        sumFrequencyPower + ...
                        tinyValue ...
                    );

                squaredBicoherence = ...
                    min( ...
                        max( ...
                            real(squaredBicoherence), ...
                            single(0) ...
                        ), ...
                        single(1) ...
                    );

                squaredBicoherenceMatrix(index1,index2) = ...
                    squaredBicoherence;


                %% 双谱基本统计累积

                validBispectrumPointCount = ...
                    validBispectrumPointCount + 1;

                bispectrumMagnitudeSum = ...
                    bispectrumMagnitudeSum + ...
                    bispectrumMagnitude;

                bispectrumEnergy = ...
                    bispectrumEnergy + ...
                    bispectrumMagnitude.^2;


                %% 最大双谱幅值及其位置

                if bispectrumMagnitude > ...
                        maximumBispectrumMagnitude

                    maximumBispectrumMagnitude = ...
                        bispectrumMagnitude;

                    peakBispectrumIndex1 = index1;
                    peakBispectrumIndex2 = index2;

                end


                %% 双相干统计累积

                squaredBicoherenceSum = ...
                    squaredBicoherenceSum + ...
                    squaredBicoherence;

                if squaredBicoherence > ...
                        maximumSquaredBicoherence

                    maximumSquaredBicoherence = ...
                        squaredBicoherence;
                end

            end
        end
    end


    %% 21. 原始双谱幅值统计

    if validBispectrumPointCount > 0

        meanBispectrumMagnitude = ...
            bispectrumMagnitudeSum / ...
            single(validBispectrumPointCount);
        %双谱平均幅度的对数，反映整体耦合强度的平均水平
        fLogMeanBispectrumMagnitude = ...
            log10( ...
                max( ...
                    meanBispectrumMagnitude, ...
                    tinyValue ...
                ) ...
            );
        %最大耦合峰值的对数，反映是否存在极强的单次耦合
        fLogMaximumBispectrumMagnitude = ...
            log10( ...
                max( ...
                    maximumBispectrumMagnitude, ...
                    tinyValue ...
                ) ...
            );
        %双谱总能量的对数，反映高阶耦合总强度
        fLogBispectralEnergy = ...
            log10( ...
                max( ...
                    bispectrumEnergy, ...
                    tinyValue ...
                ) ...
            );

    end


    %% 22. 双谱峰值频率
    %
    % centered FFT 中：
    %
    %   f = k * Fs / NFFT
    %
    % 再除以 Fs 后：
    %
    %   normalizedFrequency = k / NFFT
    %
    % 因此无需显式使用 sampleRate。
    % 对于fNormalizedBispectrumPeakFrequency1 和 fNormalizedBispectrumPeakFrequency2 
    % 给出了最强的二次相位耦合发生在哪两个频率上。

    peakFrequencyBin1 = ...
        peakBispectrumIndex1 - ...
        halfBispectrumNfft - 1;

    peakFrequencyBin2 = ...
        peakBispectrumIndex2 - ...
        halfBispectrumNfft - 1;
       
    fNormalizedBispectrumPeakFrequency1 = ...
        single(peakFrequencyBin1) / ...
        single(bispectrumNfft);

    fNormalizedBispectrumPeakFrequency2 = ...
        single(peakFrequencyBin2) / ...
        single(bispectrumNfft);


    %% 23. 双相干均值和标准差

    if validBispectrumPointCount > 0
        
        %平均平方双相干，反映整体耦合显著性（0~1）。越接近1，说明大部分频率对都存在强耦合
        fMeanSquaredBicoherence = ...
            squaredBicoherenceSum / ...
            single(validBispectrumPointCount);
        %最大耦合显著性，用于判断是否存在极其确定的单一耦合
        fMaximumSquaredBicoherence = ...
            maximumSquaredBicoherence;

    end
    
    %% 24. 双谱熵、双相干熵和双相位集中度

    bispectrumEntropy = single(0);
    bicoherenceEntropy = single(0);

    biphaseWeightedSum = ...
        complex(single(0), single(0));

    for index1 = 1:bispectrumNfft

        frequencyBin1 = ...
            index1 - ...
            halfBispectrumNfft - 1;

        for index2 = index1:bispectrumNfft

            frequencyBin2 = ...
                index2 - ...
                halfBispectrumNfft - 1;

            frequencyBin3 = ...
                frequencyBin1 + ...
                frequencyBin2;

            if frequencyBin3 >= -halfBispectrumNfft && ...
                    frequencyBin3 <= halfBispectrumNfft - 1

                bispectrumValue = ...
                    bispectrumMatrix(index1,index2);

                bispectrumMagnitude = ...
                    abs(bispectrumValue);

                squaredBicoherence = ...
                    squaredBicoherenceMatrix(index1,index2);


                %% 归一化双谱熵
                %
                % 使用 |B|^2 构造概率分布。

                if bispectrumEnergy > tinyValue

                    bispectrumProbability = ...
                        bispectrumMagnitude.^2 / ...
                        (bispectrumEnergy + tinyValue);

                    if bispectrumProbability > 0

                        bispectrumEntropy = ...
                            bispectrumEntropy - ...
                            bispectrumProbability * ...
                            log2(bispectrumProbability);

                    end
                end


                %% 归一化双相干熵

                if squaredBicoherenceSum > tinyValue

                    bicoherenceProbability = ...
                        squaredBicoherence / ...
                        ( ...
                            squaredBicoherenceSum + ...
                            tinyValue ...
                        );

                    if bicoherenceProbability > 0

                        bicoherenceEntropy = ...
                            bicoherenceEntropy - ...
                            bicoherenceProbability * ...
                            log2(bicoherenceProbability);

                    end
                end


                %% 双相干加权双相位
                %
                % 不直接对 angle(B) 计算普通平均值，
                % 而使用复平面单位向量表示双相位。

                if bispectrumMagnitude > tinyValue && ...
                        squaredBicoherence > 0

                    unitBiphase = ...
                        bispectrumValue / ...
                        bispectrumMagnitude;

                    biphaseWeightedSum = ...
                        biphaseWeightedSum + ...
                        squaredBicoherence * ...
                        unitBiphase;

                end

            end
        end
    end


    %% 25. 熵归一化

    if validBispectrumPointCount > 1

        entropyNormalization = ...
            log2( ...
                single(validBispectrumPointCount) ...
            );

        if entropyNormalization > 0

            fNormalizedBispectralEntropy = ...
                bispectrumEntropy / ...
                entropyNormalization;

            fNormalizedBicoherenceEntropy = ...
                bicoherenceEntropy / ...
                entropyNormalization;

        end
    end


    %% 26. 双相位集中度

    if squaredBicoherenceSum > tinyValue

        fBicoherenceWeightedBiphaseConcentration = ...
            abs(biphaseWeightedSum) / ...
            (squaredBicoherenceSum + tinyValue);

    end


end










%% 28. 固定尺寸输出

sampleRateSingle = single(sampleRate);

features(1) = ...
    meanFrequency / sampleRateSingle;

features(2) = ...
    medianFrequency / sampleRateSingle;

features(3) = ...
    spectralSpread / sampleRateSingle;

features(4) = ...
    occupiedBandwidth / sampleRateSingle;

features(5) = ...
    peakFrequency / sampleRateSingle;

features(6) = ...
    logBandPower;

features(7) = ...
    spectralEntropy;

features(8) = ...
    spectralCrestFactor;

features(9) = ...
    spectralFlatness;

features(10) = ...
    spectralSkewness;

features(11) = ...
    psdStandardDeviation;


%% 双谱特征输出

features(12) = ...
    fLogMeanBispectrumMagnitude;

features(13) = ...
    fLogMaximumBispectrumMagnitude;

features(14) = ...
    fLogBispectralEnergy;

features(15) = ...
    fNormalizedBispectralEntropy;

features(16) = ...
    fNormalizedBispectrumPeakFrequency1;

features(17) = ...
    fNormalizedBispectrumPeakFrequency2;

features(18) = ...
    fMeanSquaredBicoherence;

features(19) = ...
    fMaximumSquaredBicoherence;

features(20) = ...
    fNormalizedBicoherenceEntropy;

features(21) = ...
    fBicoherenceWeightedBiphaseConcentration;

%% 19. 数值保护

for featureIndex = 1:numel(features)

    if isnan(features(featureIndex)) || ...
            isinf(features(featureIndex))

        features(featureIndex) = single(0);

    end
end

end

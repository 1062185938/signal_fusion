function [features, status] = extractAllFeatures( ...
    iData, qData, signalLength, sampleRate)
%#codegen
% EXTRACTALLFEATURES
% IQ 特征提取动态库统一入口。
%
% Input:
%   iData         16384×1 single，I 路输入缓冲区
%   qData         16384×1 single，Q 路输入缓冲区
%   signalLength  int32，实际有效 IQ 采样点数量
%   sampleRate    double，采样率 Hz
%
% Output:
%   features      1×64 single 特征向量
%   status        int32 状态码
%
% Feature order:
%
%    1 - 15   Time-domain and correlation features
%   16 - 36   Frequency-domain features
%   37 - 64   Time-frequency and activity features
%
% Status:
%    0   正常
%   -1   输入长度非法
%   -2   采样率非法
%   -3   输入包含 NaN 或 Inf


%% 1. 固定输出

features = zeros(1, 64, 'single');
status = int32(0);


%% 2. 输入长度检查

minimumSignalLength = int32(32);
maximumSignalLength = int32(16384);

if signalLength < minimumSignalLength || ...
        signalLength > maximumSignalLength

    status = int32(-1);
    return;
end


%% 3. 采样率检查

if ~isscalar(sampleRate) || ...
        isnan(sampleRate) || ...
        isinf(sampleRate) || ...
        sampleRate <= 0

    status = int32(-2);
    return;
end


%% 4. 提取实际有效 IQ 数据


n = double(signalLength);

iSegment = ...
    iData(1:n);

qSegment = ...
    qData(1:n);


%% 5. 输入数据检查

if any(isnan(iSegment)) || ...
        any(isinf(iSegment)) || ...
        any(isnan(qSegment)) || ...
        any(isinf(qSegment))

    status = int32(-3);
    return;
end


%% 6. 构造复数 IQ

x = complex( ...
    iSegment, ...
    qSegment ...
);


%% 7. 时域特征
%
% 输出：
%   1×15 single

timeFeatures = ...
    extractTimeFeatures(x);


%% 8. 频域特征
%
% 输出：
%   1×21 single

frequencyFeatures = ...
    extractFrequencyFeatures( ...
        x, ...
        sampleRate ...
    );


%% 9. 时频特征
%
% 输出：
%   1×28 single

timeFrequencyFeatures = ...
    extractTimeFrequencyFeatures( ...
        x, ...
        sampleRate ...
    );


%% 10. 拼接固定 64 维输出

features(1:15) = ...
    timeFeatures;

features(16:36) = ...
    frequencyFeatures;

features(37:64) = ...
    timeFrequencyFeatures;

end

#include "iq_feature_c_api.h"

#include "extractAllFeatures.h"
#include "extractAllFeatures_initialize.h"
#include "extractAllFeatures_terminate.h"


void iqFeatureInitialize(void)
{
    extractAllFeatures_initialize();
}


int32_t iqFeatureExtract(
    const float *iData,
    const float *qData,
    int32_t signalLength,
    double sampleRate,
    float *features)
{
    int status = 0;

    extractAllFeatures(
        iData,
        qData,
        static_cast<int>(signalLength),
        sampleRate,
        features,
        &status
    );

    return static_cast<int32_t>(status);
}


void iqFeatureTerminate(void)
{
    extractAllFeatures_terminate();
}
//
// extractAllFeatures_spec.h
//
// Code generation for function 'extractAllFeatures'
//

#ifndef EXTRACTALLFEATURES_SPEC_H
#define EXTRACTALLFEATURES_SPEC_H

// Include files
#if defined(BUILDING_EXE_EXTRACTALLFEATURES)
#if defined(_WIN32) || defined(__LCC__)
#define EXTRACTALLFEATURES_DLL_EXPORT __declspec(dllimport)
#else
#define EXTRACTALLFEATURES_DLL_EXPORT __attribute__((visibility("default")))
#endif
#elif defined(BUILDING_EXTRACTALLFEATURES)
#if defined(_WIN32) || defined(__LCC__)
#define EXTRACTALLFEATURES_DLL_EXPORT __declspec(dllexport)
#else
#define EXTRACTALLFEATURES_DLL_EXPORT __attribute__((visibility("default")))
#endif
#else
#define EXTRACTALLFEATURES_DLL_EXPORT
#endif

#endif
// End of code generation (extractAllFeatures_spec.h)

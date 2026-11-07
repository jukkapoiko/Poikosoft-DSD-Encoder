#ifndef POIKOSOFT_SDM_CPU_H
#define POIKOSOFT_SDM_CPU_H

// Only the instruction sets needed by the SDM kernels.
struct CPUInfo
{
    bool sse2;
    bool avx;
};

void cpu_info(CPUInfo *info);

#endif

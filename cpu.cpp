/* Poikosoft SDM CPU dispatch: one check at context initialization.
 * SSE2 is used for two-channel processing. AVX1 requires CPU and OS
 * support for 256-bit YMM register state. No newer SIMD or FMA requirement.
 */
#include "cpu.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif
#endif

void cpu_info(CPUInfo *info)
{
    info->sse2 = false;
    info->avx = false;

#if defined(_MSC_VER)
    int regs[4] = {};
    __cpuid(regs, 1);
    info->sse2 = (regs[3] & (1 << 26)) != 0;
    if ((regs[2] & (1 << 27)) != 0 && (regs[2] & (1 << 28)) != 0)
        info->avx = (_xgetbv(0) & 0x06) == 0x06;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        info->sse2 = (edx & bit_SSE2) != 0;
        // The compiler builtin checks both CPU support and OS YMM state.
        info->avx = __builtin_cpu_supports("avx");
    }
#endif
}

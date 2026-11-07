// Private state layout shared by the scalar/SSE2 and AVX1 kernels.
#ifndef POIKOSOFT_SDM_PRIVATE_H
#define POIKOSOFT_SDM_PRIVATE_H

#include "sdm.h"
#include "cpu.h"
#include <cmath>
#include <cstring>

struct alignas(32) sdm_filter_t {
    double a[8];
    double g[4];
    unsigned freq;
};

typedef void (*sdm_process_func)(int32_t, sdm_t *, const float *, float *, unsigned);

struct alignas(32) sdm {
    double state[2][8];
    double gainm6db;
    unsigned freq;
    unsigned idx;
    const sdm_filter_t *filter;
    double prev_y;
    sdm_process_func process;
    CPUInfo cpuinfo;

    uint8_t risk_window[64];
    unsigned risk_position;
    unsigned risk_count;
    unsigned guard_active;
    unsigned stable_samples;
    uint64_t guard_activations;
    uint64_t guard_samples;
    uint64_t guard_failures;

    uint32_t peak_history;
    unsigned peak_ones;
    uint64_t peak_forced_bits;
};

static inline double sdm_clip_input(double x)
{
    const double limit = 1.4125375446227544; // +3.000 dBFS before -6 dB gain
    if (x > limit) return limit;
    if (x < -limit) return -limit;
    if (x != x) return 0.0; // NaN input is replaced by silence
    return x;
}

static inline double sdm_peak_guard(sdm_t *p, double v)
{
    unsigned bit = v < 0.0 ? 0u : 1u;
    const unsigned proposed = bit;
    const unsigned count = p->peak_ones + bit;
    if (count > 24) bit = 0;
    else if (count < 4) bit = 1;
    p->peak_forced_bits += (bit != proposed);
    const unsigned oldest = (p->peak_history >> 26) & 1u;
    p->peak_history = ((p->peak_history << 1) | bit) & 0x07ffffffu;
    p->peak_ones += bit - oldest;
    return bit ? 1.0 : -1.0;
}

/* Scalar fallback uses the same rule but keeps its three fields local to
 * the encoding block, avoiding repeated loads/stores of the SDM context. */
static inline double sdm_peak_guard_cached(double v,
    uint32_t &history, unsigned &ones, uint64_t &forced)
{
    const unsigned proposed = v < 0.0 ? 0u : 1u;
    const unsigned count = ones + proposed;
    const unsigned bit = (proposed & unsigned(count <= 24u)) |
        unsigned(count < 4u);
    forced += (bit != proposed);
    const unsigned oldest = (history >> 26) & 1u;
    history = ((history << 1) | bit) & 0x07ffffffu;
    ones += bit - oldest;
    return bit ? 1.0 : -1.0;
}

/*
 * Packed 2-channel, bit-exact 28-bit modulation guard (SSE2).
 *
 * Each 64-bit lane holds its own 27-bit history, running ones count,
 * and 64-bit forced-bit counter.  Operations are integer-only except for
 * the sign test and reconstruction of the exact +/-1.0 feedback levels.
 *
 * Keep all three vectors local to an encoding block. Commit them once at
 * the end: no per-sample per-channel state loads or stores are necessary.
 * The ordered comparison treats NaNs as non-negative, exactly as (v < 0.0)
 * in the original scalar quantizer.
 */
#include <emmintrin.h>

static inline __m128d sdm_peak_guard_2ch_sse2(__m128d v,
    __m128i &history, __m128i &ones, __m128i &forced)
{
    const __m128i one = _mm_set1_epi64x(1);
    const __m128i max_ones = _mm_set1_epi32(24);
    const __m128i min_ones = _mm_set1_epi32(4);

    // Proposed bit: 0 for negative values, 1 for >= 0 and NaN.
    const __m128i negative = _mm_srli_epi64(
        _mm_castpd_si128(_mm_cmplt_pd(v, _mm_setzero_pd())), 63);
    const __m128i proposed = _mm_xor_si128(negative, one);
    // Equivalent to testing (ones + proposed) against [4, 24]:
    // force 1 if ones < 4, otherwise accept a proposed 1 only if ones < 24.
    // This removes the dependent add before the two comparisons.
    const __m128i can_add_one = _mm_cmpgt_epi32(max_ones, ones);
    const __m128i must_add_one = _mm_cmpgt_epi32(min_ones, ones);
    const __m128i bit = _mm_or_si128(
        _mm_and_si128(can_add_one, proposed),
        _mm_and_si128(must_add_one, one));

    forced = _mm_add_epi64(forced, _mm_xor_si128(bit, proposed));
    const __m128i oldest = _mm_and_si128(_mm_srli_epi64(history, 26), one);
    history = _mm_and_si128(_mm_or_si128(_mm_slli_epi64(history, 1), bit),
        _mm_set1_epi64x(0x07ffffff));
    ones = _mm_sub_epi64(_mm_add_epi64(ones, bit), oldest);

    // 0 -> -1.0; 1 -> +1.0. No FP rounding or conversion needed.
    return _mm_xor_pd(_mm_set1_pd(-1.0),
        _mm_castsi128_pd(_mm_slli_epi64(bit, 63)));
}

static inline void sdm_peak_guard_2ch_load(const sdm_t *p0, const sdm_t *p1,
    __m128i &history, __m128i &ones, __m128i &forced)
{
    history = _mm_set_epi64x(p1->peak_history, p0->peak_history);
    ones = _mm_set_epi64x(p1->peak_ones, p0->peak_ones);
    forced = _mm_set_epi64x(p1->peak_forced_bits, p0->peak_forced_bits);
}

static inline void sdm_peak_guard_2ch_store(sdm_t *p0, sdm_t *p1,
    __m128i history, __m128i ones, __m128i forced)
{
    // SSE2 only: unpack the low and high 64-bit elements without SSE4.x.
    p0->peak_history = static_cast<uint32_t>(_mm_cvtsi128_si32(history));
    p1->peak_history = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(history, 8)));
    p0->peak_ones = static_cast<unsigned>(_mm_cvtsi128_si32(ones));
    p1->peak_ones = static_cast<unsigned>(_mm_cvtsi128_si32(_mm_srli_si128(ones, 8)));
    p0->peak_forced_bits = static_cast<uint64_t>(_mm_cvtsi128_si64(forced));
    p1->peak_forced_bits = static_cast<uint64_t>(
        _mm_cvtsi128_si64(_mm_srli_si128(forced, 8)));
}

static inline bool sdm_soft_guard_needed(const sdm_t *p, double v)
{
    return p->risk_count != 0 || p->guard_active != 0 || !(std::fabs(v) <= 2.5);
}

static inline double sdm_soft_guard(sdm_t *p, double x, double v, double *d, const double *a)
{
    const double av = std::fabs(v);
    if (p->risk_count == 0 && p->guard_active == 0 && av <= 2.5)
        return v;

    const unsigned pos = p->risk_position;
    const unsigned risk = !(av <= 2.5) ? 1u : 0u;
    p->risk_count += risk;
    p->risk_count -= p->risk_window[pos];
    p->risk_window[pos] = uint8_t(risk);
    p->risk_position = (pos + 1) & 63;
    if (!p->guard_active && (p->risk_count >= 16 || !(av <= 8.0))) {
        p->guard_active = 1;
        p->stable_samples = 0;
        p->guard_activations++;
    }
    if (p->guard_active) {
        // Smoothly limit coefficient-weighted next states only.
        const double threshold = 1.1;
        const double limit = 3.0;
        for (int i = 0; i < 8; ++i) {
            const double w = a[i] * d[i];
            const double aw = std::fabs(w);
            if (aw > threshold) {
                const double mag = threshold + (limit - threshold) *
                    std::tanh((aw - threshold) / (limit - threshold));
                d[i] = std::copysign(mag, w) / a[i];
            }
        }
        v = x + a[0] * d[0];
        for (int i = 1; i < 8; ++i) v += a[i] * d[i];
        p->guard_samples++;
        if (av < 1.45 && p->risk_count < 4) p->stable_samples++;
        else p->stable_samples = 0;
        if (p->stable_samples >= 64) {
            p->guard_active = 0;
            p->stable_samples = 0;
        }
    }
    if (!std::isfinite(v)) p->guard_failures++;
    return v;
}

#endif

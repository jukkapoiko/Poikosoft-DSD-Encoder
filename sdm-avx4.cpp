/* Optional four-channel AVX SDM kernel. Compile this translation unit with AVX enabled,
 * and keep sdm.cpp compiled for the application's minimum supported ISA.
 * No FMA contraction: the SSE2 reference bitstream must remain unchanged.
 */
#include <immintrin.h>
#include "sdm-private.h"

#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("avx"))), apply_to=function)
#endif

/* Four AVX1 lanes share one packed SSE2 32-bit modulation guard.
 * The forcing delta is accumulated in 32-bit lanes only within this call,
 * then added to each channel's 64-bit lifetime counter. len <= UINT_MAX,
 * hence no accumulation overflow is possible within one call.
 */
static inline __m256d sdm_peak_guard_4ch_avx1(__m256d v,
    __m128i &history, __m128i &ones, __m128i &forced)
{
    const __m128d zero = _mm_setzero_pd();
    const __m128i one = _mm_set1_epi32(1);
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);

    const __m128i negative01 = _mm_srli_epi64(
        _mm_castpd_si128(_mm_cmplt_pd(lo, zero)), 63);
    const __m128i negative23 = _mm_srli_epi64(
        _mm_castpd_si128(_mm_cmplt_pd(hi, zero)), 63);
    const __m128i negative = _mm_unpacklo_epi64(
        _mm_shuffle_epi32(negative01, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm_shuffle_epi32(negative23, _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i proposed = _mm_xor_si128(negative, one);
    // Exact valid-state equivalent of checking (ones + proposed):
    // force one below 4, disallow a proposed one at 24 or above.
    const __m128i can_add_one = _mm_cmpgt_epi32(_mm_set1_epi32(24), ones);
    const __m128i must_add_one = _mm_cmpgt_epi32(_mm_set1_epi32(4), ones);
    const __m128i bit = _mm_or_si128(
        _mm_and_si128(can_add_one, proposed),
        _mm_and_si128(must_add_one, one));

    forced = _mm_add_epi32(forced, _mm_xor_si128(bit, proposed));
    const __m128i oldest = _mm_and_si128(_mm_srli_epi32(history, 26), one);
    history = _mm_and_si128(_mm_or_si128(_mm_slli_epi32(history, 1), bit),
        _mm_set1_epi32(0x07ffffff));
    ones = _mm_sub_epi32(_mm_add_epi32(ones, bit), oldest);

    const __m128d y01 = _mm_xor_pd(_mm_set1_pd(-1.0),
        _mm_castsi128_pd(_mm_slli_epi64(_mm_unpacklo_epi32(bit, _mm_setzero_si128()), 63)));
    const __m128d y23 = _mm_xor_pd(_mm_set1_pd(-1.0),
        _mm_castsi128_pd(_mm_slli_epi64(_mm_unpackhi_epi32(bit, _mm_setzero_si128()), 63)));
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(y01), y23, 1);
}

void sdm_process_interleaved_4ch_avx(int32_t channels, sdm_t *p0, sdm_t *p1, sdm_t *p2, sdm_t *p3, const float *input, float *output, unsigned int len)
{
    const double *a = p0->filter->a;
    const double *g = p0->filter->g;
    const __m256d c0 = _mm256_set1_pd(a[0]);
    const __m256d c1 = _mm256_set1_pd(a[1]);
    const __m256d c2 = _mm256_set1_pd(a[2]);
    const __m256d c3 = _mm256_set1_pd(a[3]);
    const __m256d c4 = _mm256_set1_pd(a[4]);
    const __m256d c5 = _mm256_set1_pd(a[5]);
    const __m256d c6 = _mm256_set1_pd(a[6]);
    const __m256d c7 = _mm256_set1_pd(a[7]);
    const __m256d g0 = _mm256_set1_pd(g[0]);
    const __m256d g2 = _mm256_set1_pd(g[1]);
    const __m256d g4 = _mm256_set1_pd(g[2]);
    const __m256d g6 = _mm256_set1_pd(g[3]);
    const __m256d gain = _mm256_set_pd(p3->gainm6db, p2->gainm6db, p1->gainm6db, p0->gainm6db);

    const double *state0 = p0->state[p0->idx];
    const double *state1 = p1->state[p1->idx];
    const double *state2 = p2->state[p2->idx];
    const double *state3 = p3->state[p3->idx];
    __m256d s0 = _mm256_set_pd(state3[0], state2[0], state1[0], state0[0]);
    __m256d s1 = _mm256_set_pd(state3[1], state2[1], state1[1], state0[1]);
    __m256d s2 = _mm256_set_pd(state3[2], state2[2], state1[2], state0[2]);
    __m256d s3 = _mm256_set_pd(state3[3], state2[3], state1[3], state0[3]);
    __m256d s4 = _mm256_set_pd(state3[4], state2[4], state1[4], state0[4]);
    __m256d s5 = _mm256_set_pd(state3[5], state2[5], state1[5], state0[5]);
    __m256d s6 = _mm256_set_pd(state3[6], state2[6], state1[6], state0[6]);
    __m256d s7 = _mm256_set_pd(state3[7], state2[7], state1[7], state0[7]);
    __m256d y = _mm256_set_pd(p3->prev_y, p2->prev_y, p1->prev_y, p0->prev_y);
    const unsigned count = len;
    __m128i peak_history = _mm_set_epi32(p3->peak_history, p2->peak_history,
        p1->peak_history, p0->peak_history);
    __m128i peak_ones = _mm_set_epi32(p3->peak_ones, p2->peak_ones,
        p1->peak_ones, p0->peak_ones);
    __m128i peak_forced = _mm_setzero_si128();
    while (len--) {
        // Each four-channel group occupies four adjacent interleaved samples.
        const __m128 samples = _mm_loadu_ps(input);
        __m256d pcm = _mm256_cvtps_pd(samples);
        pcm = _mm256_andnot_pd(_mm256_cmp_pd(pcm, pcm, _CMP_UNORD_Q), pcm);
        pcm = _mm256_max_pd(_mm256_set1_pd(-1.4125375446227544),
            _mm256_min_pd(pcm, _mm256_set1_pd(1.4125375446227544)));
        const __m256d x = _mm256_mul_pd(pcm, gain);

        const __m256d xmy = _mm256_sub_pd(x, y);
        const __m256d d0 = _mm256_sub_pd(_mm256_add_pd(s0, xmy), _mm256_mul_pd(s1, g0));
        const __m256d d1 = _mm256_add_pd(s1, s0);
        const __m256d d2 = _mm256_sub_pd(_mm256_add_pd(s2, s1), _mm256_mul_pd(s3, g2));
        const __m256d d3 = _mm256_add_pd(s3, s2);
        const __m256d d4 = _mm256_sub_pd(_mm256_add_pd(s4, s3), _mm256_mul_pd(s5, g4));
        const __m256d d5 = _mm256_add_pd(s5, s4);
        const __m256d d6 = _mm256_sub_pd(_mm256_add_pd(s6, s5), _mm256_mul_pd(s7, g6));
        const __m256d d7 = _mm256_add_pd(s7, s6);
        s0 = d0; s1 = d1; s2 = d2; s3 = d3; s4 = d4; s5 = d5; s6 = d6; s7 = d7;
        __m256d vv = _mm256_mul_pd(d0, c0);
        vv = _mm256_add_pd(vv, _mm256_mul_pd(d1, c1));
        vv = _mm256_add_pd(vv, _mm256_mul_pd(d2, c2));
        vv = _mm256_add_pd(vv, _mm256_mul_pd(d3, c3));
        vv = _mm256_add_pd(vv, _mm256_mul_pd(d4, c4));
        vv = _mm256_add_pd(vv, _mm256_mul_pd(d5, c5));
        vv = _mm256_add_pd(vv, _mm256_mul_pd(d6, c6));
        vv = _mm256_add_pd(vv, _mm256_mul_pd(d7, c7));
        __m256d v = _mm256_add_pd(x, vv);
        // Compare the four magnitudes directly in AVX1. A NaN must trigger
        // the slow path, exactly as the scalar !(fabs(v) <= 2.5) test.
        const __m256d abs_v = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
        const int normal_mask = _mm256_movemask_pd(_mm256_cmp_pd(abs_v,
            _mm256_set1_pd(2.5), _CMP_LE_OQ));
        if (p0->risk_count != 0 || p0->guard_active != 0 ||
            p1->risk_count != 0 || p1->guard_active != 0 ||
            p2->risk_count != 0 || p2->guard_active != 0 ||
            p3->risk_count != 0 || p3->guard_active != 0 || normal_mask != 15) {
            double original_v[4];
            _mm256_storeu_pd(original_v, v);
            sdm_t *contexts[4] = {p0, p1, p2, p3};
            double by_channel[4][8];
            const __m256d stages[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
            for (int i = 0; i < 8; i++) {
                double lanes[4];
                _mm256_storeu_pd(lanes, stages[i]);
                for (int c = 0; c < 4; c++)
                    by_channel[c][i] = lanes[c];
            }
            double xv[4], new_v[4];
            _mm256_storeu_pd(xv, x);
            bool changed = false;
            for (int c = 0; c < 4; c++) {
                const unsigned long long before = contexts[c]->guard_samples;
                new_v[c] = sdm_soft_guard(contexts[c], xv[c], original_v[c],
                    by_channel[c], a);
                if (contexts[c]->guard_samples != before)
                    changed = true;
            }
            if (changed) {
                s0 = _mm256_set_pd(by_channel[3][0], by_channel[2][0], by_channel[1][0], by_channel[0][0]);
                s1 = _mm256_set_pd(by_channel[3][1], by_channel[2][1], by_channel[1][1], by_channel[0][1]);
                s2 = _mm256_set_pd(by_channel[3][2], by_channel[2][2], by_channel[1][2], by_channel[0][2]);
                s3 = _mm256_set_pd(by_channel[3][3], by_channel[2][3], by_channel[1][3], by_channel[0][3]);
                s4 = _mm256_set_pd(by_channel[3][4], by_channel[2][4], by_channel[1][4], by_channel[0][4]);
                s5 = _mm256_set_pd(by_channel[3][5], by_channel[2][5], by_channel[1][5], by_channel[0][5]);
                s6 = _mm256_set_pd(by_channel[3][6], by_channel[2][6], by_channel[1][6], by_channel[0][6]);
                s7 = _mm256_set_pd(by_channel[3][7], by_channel[2][7], by_channel[1][7], by_channel[0][7]);
                v = _mm256_loadu_pd(new_v);
            }
        }

        // AVX1 arithmetic with one packed SSE2 4-lane integer history.
        y = sdm_peak_guard_4ch_avx1(v, peak_history, peak_ones, peak_forced);
        const __m128 floats = _mm256_cvtpd_ps(y);
        if (channels == 4) {
            _mm_storeu_ps(output, floats);
        }
        else {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, floats);
            output[0] = tmp[0]; output[1] = tmp[1];
            output[2] = tmp[2]; output[3] = tmp[3];
        }
        input += channels;
        output += channels;
    }
    sdm_t *ps[4] = {p0, p1, p2, p3};
    __m256d states[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    double tmp[4];
    for (int i = 0; i < 8; i++) {
        _mm256_storeu_pd(tmp, states[i]);
        for (int c = 0; c < 4; c++) ps[c]->state[ps[c]->idx ^ (count & 1)][i] = tmp[c];
    }
    _mm256_storeu_pd(tmp, y);
    for (int c = 0; c < 4; c++) {ps[c]->idx ^= count & 1; ps[c]->prev_y = tmp[c]; }
    alignas(16) uint32_t history32[4], ones32[4], forced32[4];
    _mm_store_si128(reinterpret_cast<__m128i *>(history32), peak_history);
    _mm_store_si128(reinterpret_cast<__m128i *>(ones32), peak_ones);
    _mm_store_si128(reinterpret_cast<__m128i *>(forced32), peak_forced);
    for (int c = 0; c < 4; ++c) {
        ps[c]->peak_history = history32[c];
        ps[c]->peak_ones = ones32[c];
        ps[c]->peak_forced_bits += forced32[c];
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

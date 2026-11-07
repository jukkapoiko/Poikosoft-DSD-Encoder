/*
 * Poikosoft 8th-order CRFF DSD encoder.
 * Input PCM is hard-limited to +3 dBFS, attenuated by exactly 6 dB,
 * then processed using per-channel soft-state limiting and a strict
 * zero-latency 28-bit output modulation guard. Its final DSD bit is
 * always used in the feedback loop. Supports independent 1..8 channels.
 *
 * Original sigma-delta modulator copyright and license notice:
 * Copyright (c) 2015 Mans Rullgard <mans@mansr.com>
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */
#include <string.h>
#include <stdlib.h>
#ifdef _MSC_VER
#include <malloc.h>
#endif
#include <cmath>
#include "cpu.h"
#include "sdm.h"
#include "sdm-private.h"

#include "sdm-filters.h"

#include <immintrin.h>
#include <emmintrin.h>

#define SWAPHL(x) \
  _mm_castsi128_pd(_mm_shuffle_epi32(_mm_castpd_si128(x), 0x4e))

#define SDM_FILTER_SSE2(s0, s1, s2, s3, src, x) do {            \
    __m128d tx, t0, t1, t2, t3;                                 \
    __m128d sx;                                                 \
                                                                \
    sx = _mm_set1_pd(x);                                        \
    s0 = _mm_load_pd(src);                                      \
    s1 = _mm_load_pd(src + 2);                                  \
    s2 = _mm_load_pd(src + 4);                                  \
    s3 = _mm_load_pd(src + 6);                                  \
                                                                \
    tx = _mm_shuffle_pd(sx, s0, 1);                             \
    t0 = _mm_shuffle_pd(s0, s1, 1);                             \
    t1 = _mm_shuffle_pd(s1, s2, 1);                             \
    t2 = _mm_shuffle_pd(s2, s3, 1);                             \
    t3 = _mm_shuffle_pd(s3, s3, 1);                             \
                                                                \
    s0 = _mm_add_pd(s0, tx);                                    \
    s1 = _mm_add_pd(s1, t0);                                    \
    s2 = _mm_add_pd(s2, t1);                                    \
    s3 = _mm_add_pd(s3, t2);                                    \
                                                                \
    s0 = _mm_sub_sd(s0, _mm_mul_sd(t0, _mm_load_sd(g)));        \
    s1 = _mm_sub_sd(s1, _mm_mul_sd(t1, _mm_load_sd(g + 1)));    \
    s2 = _mm_sub_sd(s2, _mm_mul_sd(t2, _mm_load_sd(g + 2)));    \
    s3 = _mm_sub_sd(s3, _mm_mul_sd(t3, _mm_load_sd(g + 3)));    \
  } while (0)

void sdm_process_interleaved_sse2(int32_t channels, sdm_t *p, const float *ibuf, float *obuf, unsigned int len)
{
    const double *a = p->filter->a;
    const double *g = p->filter->g;
    const double gain6mdb = p->gainm6db;
    double x;
    const double *s;
    double *d;
    double v;
    double y = p->prev_y;
    uint32_t peak_history = p->peak_history;
    unsigned peak_ones = p->peak_ones;
    uint64_t peak_forced = p->peak_forced_bits;

    while (len--) {

        x = sdm_clip_input(*ibuf) * gain6mdb;

        s = p->state[p->idx];
        p->idx ^= 1;
        d = p->state[p->idx];

        __m128d s0, s1, s2, s3;
        __m128d v0, v1;

        SDM_FILTER_SSE2(s0, s1, s2, s3, s, x - y);

        _mm_store_pd(d, s0);
        _mm_store_pd(d + 2, s1);
        _mm_store_pd(d + 4, s2);
        _mm_store_pd(d + 6, s3);

        v0 = _mm_mul_pd(s0, _mm_load_pd(a));
        v1 = _mm_mul_pd(s1, _mm_load_pd(a + 2));
        v0 = _mm_add_pd(v0, _mm_mul_pd(s2, _mm_load_pd(a + 4)));
        v1 = _mm_add_pd(v1, _mm_mul_pd(s3, _mm_load_pd(a + 6)));
        v0 = _mm_add_pd(v0, v1);
        v0 = _mm_add_pd(v0, SWAPHL(v0));

        v = x + _mm_cvtsd_f64(v0);
        v = sdm_soft_guard(p, x, v, d, a);

        *obuf = y = sdm_peak_guard_cached(v, peak_history, peak_ones, peak_forced);

        obuf += channels;
        ibuf += channels;
    }
    p->prev_y = y;
    p->peak_history = peak_history;
    p->peak_ones = peak_ones;
    p->peak_forced_bits = peak_forced;
}

void sdm_process_interleaved_c(int32_t channels, sdm_t *p, const float *ibuf, float *obuf, unsigned int len)
{
    const double *a = p->filter->a;
    const double *g = p->filter->g;
    const double gain6mdb = p->gainm6db;
    double x;
    const double *s;
    double *d;
    double v;
    double y = p->prev_y;
    uint32_t peak_history = p->peak_history;
    unsigned peak_ones = p->peak_ones;
    uint64_t peak_forced = p->peak_forced_bits;

    while (len--) {

        x = sdm_clip_input(*ibuf) * gain6mdb;

        s = p->state[p->idx];
        p->idx ^= 1;
        d = p->state[p->idx];

        // Only four CRFF resonator feedback connections exist: stages 0, 2, 4, 6.
        d[0] = (s[0] + (x - y)) - g[0] * s[1];
        d[1] = s[1] + s[0];
        d[2] = (s[2] + s[1]) - g[1] * s[3];
        d[3] = s[3] + s[2];
        d[4] = (s[4] + s[3]) - g[2] * s[5];
        d[5] = s[5] + s[4];
        d[6] = (s[6] + s[5]) - g[3] * s[7];
        d[7] = s[7] + s[6];
        v = a[0] * d[0];
        for (int i = 1; i < 8; ++i)
            v += a[i] * d[i];
        v = x + v;
        v = sdm_soft_guard(p, x, v, d, a);

        *obuf = y = sdm_peak_guard_cached(v, peak_history, peak_ones, peak_forced);

        obuf += channels;
        ibuf += channels;
    }
    p->prev_y = y;
    p->peak_history = peak_history;
    p->peak_ones = peak_ones;
    p->peak_forced_bits = peak_forced;
}

void sdm_process_interleaved(int32_t channels, sdm_t *p, const float *ibuf, float *obuf, unsigned int len)
{
    p->process(channels, p, ibuf, obuf, len);
}

static const sdm_filter_t *sdm_find_filter(unsigned freq)
{
    for (unsigned i = 0; i < sizeof(sdm_filters)/sizeof(sdm_filters[0]); ++i)
        if (sdm_filters[i].freq == freq) return &sdm_filters[i];
    return nullptr;
}

sdm_t *sdm_init(unsigned sourcefreq, unsigned freq)
{
    (void)sourcefreq;
    const sdm_filter_t *filter = sdm_find_filter(freq);
    if (!filter) return nullptr;
#ifdef _MSC_VER
    sdm_t *p = static_cast<sdm_t *>(_aligned_malloc(sizeof(sdm_t), 32));
#else
    void *memory = nullptr;
    if (posix_memalign(&memory, 32, sizeof(sdm_t)) != 0) return nullptr;
    sdm_t *p = static_cast<sdm_t *>(memory);
#endif
    if (!p) return nullptr;
    memset(p, 0, sizeof(*p));
    p->filter = filter;
    p->freq = freq;
    p->gainm6db = 0.5011872336272722;
    p->peak_history = 0x05555555u;
    p->peak_ones = 14;
    cpu_info(&p->cpuinfo);
    p->process = p->cpuinfo.sse2 ? sdm_process_interleaved_sse2 : sdm_process_interleaved_c;
    return p;
}

void sdm_reset(sdm_t *p, unsigned sourcefreq)
{
    (void)sourcefreq;
    if (!p) return;
    const sdm_filter_t *filter = p->filter;
    CPUInfo cpu = p->cpuinfo;
    const unsigned freq = p->freq;
    memset(p, 0, sizeof(*p));
    p->filter = filter;
    p->cpuinfo = cpu;
    p->freq = freq;
    p->gainm6db = 0.5011872336272722;
    p->peak_history = 0x05555555u;
    p->peak_ones = 14;
    p->process = p->cpuinfo.sse2 ? sdm_process_interleaved_sse2 : sdm_process_interleaved_c;
}

void sdm_close(sdm_t *p)
{
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
}

static void sdm_process_interleaved_2ch(int32_t channels, sdm_t *p0, sdm_t *p1, const float *input, float *output, unsigned int len)
{
    if (!len) return;

    if (p0->filter != p1->filter ||
        !p0->cpuinfo.sse2 || !p1->cpuinfo.sse2) {
        sdm_process_interleaved(channels, p0, input, output, len);
        sdm_process_interleaved(channels, p1, input + 1, output + 1, len);
        return;
    }
    const double *a = p0->filter->a;
    const double *g = p0->filter->g;
    const __m128d c0 = _mm_set1_pd(a[0]);
    const __m128d c1 = _mm_set1_pd(a[1]);
    const __m128d c2 = _mm_set1_pd(a[2]);
    const __m128d c3 = _mm_set1_pd(a[3]);
    const __m128d c4 = _mm_set1_pd(a[4]);
    const __m128d c5 = _mm_set1_pd(a[5]);
    const __m128d c6 = _mm_set1_pd(a[6]);
    const __m128d c7 = _mm_set1_pd(a[7]);
    const __m128d g0 = _mm_set1_pd(g[0]);
    const __m128d g2 = _mm_set1_pd(g[1]);
    const __m128d g4 = _mm_set1_pd(g[2]);
    const __m128d g6 = _mm_set1_pd(g[3]);
    const __m128d gain = _mm_set_pd(p1->gainm6db, p0->gainm6db);

    const double *state0 = p0->state[p0->idx];
    const double *state1 = p1->state[p1->idx];

    __m128d s0 = _mm_set_pd(state1[0], state0[0]);
    __m128d s1 = _mm_set_pd(state1[1], state0[1]);
    __m128d s2 = _mm_set_pd(state1[2], state0[2]);
    __m128d s3 = _mm_set_pd(state1[3], state0[3]);
    __m128d s4 = _mm_set_pd(state1[4], state0[4]);
    __m128d s5 = _mm_set_pd(state1[5], state0[5]);
    __m128d s6 = _mm_set_pd(state1[6], state0[6]);
    __m128d s7 = _mm_set_pd(state1[7], state0[7]);
    __m128d y = _mm_set_pd(p1->prev_y, p0->prev_y);

    const unsigned count = len;
    __m128i peak_history, peak_ones, peak_forced;
    sdm_peak_guard_2ch_load(p0, p1, peak_history, peak_ones, peak_forced);
    while (len--) {
        __m128d pcm = _mm_cvtps_pd(_mm_loadl_pi(_mm_setzero_ps(), (const __m64 *)input));
        pcm = _mm_andnot_pd(_mm_cmpunord_pd(pcm, pcm), pcm);
        pcm = _mm_max_pd(_mm_set1_pd(-1.4125375446227544), _mm_min_pd(pcm, _mm_set1_pd(1.4125375446227544)));

        const __m128d x = _mm_mul_pd(pcm, gain);
        const __m128d xmy = _mm_sub_pd(x, y);
        const __m128d d0 = _mm_sub_pd(_mm_add_pd(s0, xmy), _mm_mul_pd(s1, g0));
        const __m128d d1 = _mm_add_pd(s1, s0);
        const __m128d d2 = _mm_sub_pd(_mm_add_pd(s2, s1), _mm_mul_pd(s3, g2));
        const __m128d d3 = _mm_add_pd(s3, s2);
        const __m128d d4 = _mm_sub_pd(_mm_add_pd(s4, s3), _mm_mul_pd(s5, g4));
        const __m128d d5 = _mm_add_pd(s5, s4);
        const __m128d d6 = _mm_sub_pd(_mm_add_pd(s6, s5), _mm_mul_pd(s7, g6));
        const __m128d d7 = _mm_add_pd(s7, s6);
        s0 = d0; s1 = d1; s2 = d2; s3 = d3; s4 = d4; s5 = d5; s6 = d6; s7 = d7;
        __m128d vv = _mm_mul_pd(d0, c0);
        vv = _mm_add_pd(vv, _mm_mul_pd(d1, c1));
        vv = _mm_add_pd(vv, _mm_mul_pd(d2, c2));
        vv = _mm_add_pd(vv, _mm_mul_pd(d3, c3));
        vv = _mm_add_pd(vv, _mm_mul_pd(d4, c4));
        vv = _mm_add_pd(vv, _mm_mul_pd(d5, c5));
        vv = _mm_add_pd(vv, _mm_mul_pd(d6, c6));
        vv = _mm_add_pd(vv, _mm_mul_pd(d7, c7));
        __m128d v = _mm_add_pd(x, vv);
        const double v0s = _mm_cvtsd_f64(v);
        const double v1s = _mm_cvtsd_f64(_mm_unpackhi_pd(v, v));
        if (sdm_soft_guard_needed(p0, v0s) || sdm_soft_guard_needed(p1, v1s)) {

            double state0[8], state1[8];
            const __m128d stages[8] = {s0, s1, s2, s3, s4, s5, s6, s7};

            for (int i = 0; i < 8; i++) {
                state0[i] = _mm_cvtsd_f64(stages[i]);
                state1[i] = _mm_cvtsd_f64(_mm_unpackhi_pd(stages[i], stages[i]));
            }

            const unsigned long long before0 = p0->guard_samples;
            const unsigned long long before1 = p1->guard_samples;
            const double vo0 = sdm_soft_guard(p0, _mm_cvtsd_f64(x), v0s, state0, a);
            const double vo1 = sdm_soft_guard(p1, _mm_cvtsd_f64(_mm_unpackhi_pd(x, x)), v1s, state1, a);

            if (p0->guard_samples != before0 || p1->guard_samples != before1) {
                s0 = _mm_set_pd(state1[0], state0[0]);
                s1 = _mm_set_pd(state1[1], state0[1]);
                s2 = _mm_set_pd(state1[2], state0[2]);
                s3 = _mm_set_pd(state1[3], state0[3]);
                s4 = _mm_set_pd(state1[4], state0[4]);
                s5 = _mm_set_pd(state1[5], state0[5]);
                s6 = _mm_set_pd(state1[6], state0[6]);
                s7 = _mm_set_pd(state1[7], state0[7]);
                v = _mm_set_pd(vo1, vo0);
            }
        }

        // Final DSD bits must be fed back separately for each lane.
        y = sdm_peak_guard_2ch_sse2(v, peak_history, peak_ones, peak_forced);
        const __m128 floats = _mm_cvtpd_ps(y);
        _mm_storel_pi((__m64 *)output, floats);
        input += channels;
        output += channels;
    }
    p0->idx ^= count & 1;
    p1->idx ^= count & 1;
    double *t0 = p0->state[p0->idx];
    double *t1 = p1->state[p1->idx];
    __m128d a0[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    for (int i = 0; i < 8; i++) {
        t0[i] = _mm_cvtsd_f64(a0[i]);
        t1[i] = _mm_cvtsd_f64(_mm_unpackhi_pd(a0[i], a0[i]));
    }
    p0->prev_y = _mm_cvtsd_f64(y);
    p1->prev_y = _mm_cvtsd_f64(_mm_unpackhi_pd(y, y));
    sdm_peak_guard_2ch_store(p0, p1, peak_history, peak_ones, peak_forced);
}

/* The four-channel kernel uses AVX1 only. */
extern void sdm_process_interleaved_4ch_avx(int32_t channels, sdm_t *p0, sdm_t *p1,
    sdm_t *p2, sdm_t *p3, const float *input, float *output, unsigned int len);

static void sdm_process_interleaved_4ch(int32_t channels, sdm_t *p0, sdm_t *p1,
    sdm_t *p2, sdm_t *p3, const float *input, float *output, unsigned int len)
{
    if (!len) return;

    if (p0->cpuinfo.avx && p1->cpuinfo.avx && p2->cpuinfo.avx && p3->cpuinfo.avx &&
        p0->filter == p1->filter && p0->filter == p2->filter &&
        p0->filter == p3->filter) {
        sdm_process_interleaved_4ch_avx(channels, p0, p1, p2, p3, input, output, len);
    }
    else {
        sdm_process_interleaved_2ch(channels, p0, p1, input, output, len);
        sdm_process_interleaved_2ch(channels, p2, p3, input + 2, output + 2, len);
    }
}

/* One interleaved 1..8-channel entry point.
 * Uses AVX1 for groups of four when the CPU and OS support it,
 * SSE2 for remaining pairs, and the original path for odd channels.
 * Preserves each channel's continuous modulator feedback history.
 */
int sdm_process_interleaved_channels(unsigned channels, sdm_t *const *contexts,
    const float *input, float *output, unsigned int frames)
{
    if (channels < 1 || channels > 8 || !contexts || !input || !output)
        return 0;

    for (unsigned c = 0; c < channels; ++c) {
        if (!contexts[c] || !contexts[c]->filter)
            return 0;
        for (unsigned i = 0; i < c; ++i) {
            if (contexts[c] == contexts[i])
                return 0;
        }
    }

    unsigned c = 0;
    for (; c + 3 < channels; c += 4) {
        sdm_process_interleaved_4ch((int32_t)channels,
            contexts[c], contexts[c + 1], contexts[c + 2], contexts[c + 3],
            input + c, output + c, frames);
    }
    for (; c + 1 < channels; c += 2) {
        sdm_process_interleaved_2ch((int32_t)channels,
            contexts[c], contexts[c + 1], input + c, output + c, frames);
    }
    for (; c < channels; ++c) {
        sdm_process_interleaved((int32_t)channels,
            contexts[c], input + c, output + c, frames);
    }
    return 1;
}

int sdm_get_leakage_stats(const sdm_t *p, unsigned long long *activations,
    unsigned long long *protected_samples, unsigned long long *nonfinite_outputs)
{
    if (!p || !activations || !protected_samples || !nonfinite_outputs) return 0;
    *activations = p->guard_activations;
    *protected_samples = p->guard_samples;
    *nonfinite_outputs = p->guard_failures;
    return 1;
}
unsigned long long sdm_get_peak_forced_bits(const sdm_t *p)
{
    return p ? p->peak_forced_bits : 0;
}
unsigned sdm_get_frequency(const sdm_t *p) { return p ? p->freq : 0u; }


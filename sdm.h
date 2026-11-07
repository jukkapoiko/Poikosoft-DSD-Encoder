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
#ifndef POIKOSOFT_SDM_H
#define POIKOSOFT_SDM_H

#include <stdint.h>
#include <stddef.h>

typedef struct sdm sdm_t;

// Create an encoder for the requested DSD output rate.
sdm_t *sdm_init(unsigned sourcefreq, unsigned targetfreq);
void sdm_reset(sdm_t *s, unsigned sourcefreq);
void sdm_close(sdm_t *s);

// A single channel from an interleaved frame (stride = channels).
void sdm_process_interleaved(int32_t channels, sdm_t *s,
    const float *input, float *output, unsigned frames);

// 1..8 independent contexts; AVX1 groups of four, SSE2 pairs and scalar tails.
// In-place input == output is allowed. Returns 0 on invalid arguments.
int sdm_process_interleaved_channels(unsigned channels, sdm_t *const *contexts,
    const float *input, float *output, unsigned frames);

unsigned long long sdm_get_peak_forced_bits(const sdm_t *s);
int sdm_get_leakage_stats(const sdm_t *s, unsigned long long *activations,
    unsigned long long *protected_samples, unsigned long long *nonfinite_outputs);
unsigned sdm_get_frequency(const sdm_t *s);
const char *sdm_get_filter_name(const sdm_t *s);

#endif

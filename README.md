# Poikosoft DSD Encoder v14

**A high-performance, eighth-order CRFF sigma-delta modulator for converting oversampled PCM into a 1-bit Direct Stream Digital (DSD) signal.**

Developed and maintained by [Poikosoft](https://www.poikosoft.com), with extensive implementation changes by Jukka Poikolainen. The encoder **derives from the sigma-delta modulation work of Måns Rullgård in SoX**; the original author's contribution is acknowledged and remains licensed under the LGPL.

## What it does

The Poikosoft DSD Encoder is a standalone, streaming **8th-order cascade-of-resonators with distributed feedback (CRFF)** modulator. It accepts **32-bit floating-point, interleaved PCM samples at the intended DSD bit-clock rate** and produces one `float` result per input sample, representing a DSD bit as **`-1.0f` or `+1.0f`**.

It supports one to eight independently maintained channels, including mono, stereo, 4.0, 5.0, 5.1 and other channel counts within that range.

**Scope:** This source implements the sigma-delta modulation stage. It does **not** resample audio to the DSD rate, pack the output into bytes, or write DSF/DFF files. A separate resampler (for example Poikosoft SoX Resampler) and a DSD bit-packing/file-output stage can be used around it.

## How it differs from the original SoX `sdm.c`

The starting point is the SoX sigma-delta modulator developed by **Måns Rullgård**:

- [Original `sdm.c` source](https://github.com/mansr/sox/blob/master/src/sdm.c)
- [Original SoX repository](https://github.com/mansr/sox)

The original code provides sigma-delta processing within SoX, including configurable filter and trellis-related functionality. Poikosoft v14 is **a specialized, substantially reworked 8th-order CRFF engine** with a different integration API and data path. It is **not** a drop-in replacement for every feature or public interface of the original SoX effect.

### Optimized scalar and SIMD encoding

- Reworked scalar C++ implementation
- **SSE2:** parallel processing of two independent audio channels
- **AVX1:** parallel processing of four independent audio channels, without requiring AVX2 or FMA3
- A single interleaved multichannel entry point that combines four-channel groups, two-channel pairs and scalar tails for **1–8 channels**
- CPU/OS instruction-set detection and automatic dispatch
- Independent integrator state, output history and feedback for every channel
- Support for processing **in place** (`input == output`)
- Consistent output-bit decisions across scalar, SSE2 and AVX processing paths for equivalent states and inputs

AVX code is isolated in `sdm-avx4.cpp` so that an AVX-capable CPU is not required just to load the scalar/SSE2 portion of the library.

### Additional protection for high-level input

- Input hard-limiting at **+3.0 dBFS** (approximately `±1.4125375446` relative to full scale), followed by a fixed **−6 dB input gain**
- Adaptive, per-channel **soft-state limiting** for stressed eighth-order integrator states
- A **zero-lookahead 28-bit modulation guard** that keeps at least four zero bits and four one bits in every consecutive 28-bit output window
- The **actual guarded output bit** is always fed back into the modulator, keeping output and feedback consistent
- Counters for forced modulation bits and soft-state guard activity, useful for diagnostics
- No added trellis lookahead or extra buffering latency in this modulator stage

The 28-bit guard is intended to enforce a common SACD-style modulation-density restriction. This description is **not** a claim of formal SACD certification or a guarantee of overall DSD file-format compliance. As with any high-order 1-bit modulator, stability and measured audio quality still require validation for the intended signals and settings.

### DSD rate-specific filters

The supplied coefficient table (`sdm-filters.h`) contains 8th-order CRFF configurations for:

| Format | DSD bit-clock rate
| --- | --- |
| **DSD64** | 2,822,400 Hz
| DSD128 | 5,644,800 Hz
| DSD256 | 11,289,600 Hz
| DSD512 | 22,579,200 Hz
| DSD1024 | 45,158,400 Hz
| DSD2048 | 90,316,800 Hz

## API and data format

The public API is in [`sdm.h`](sdm.h). The important entry points are:

- `sdm_init(sourcefreq, targetfreq)` — allocate an independent modulator instance for a supported target DSD rate; returns `nullptr` for unsupported rates or allocation failure
- `sdm_process_interleaved_channels(channels, contexts, input, output, frames)` — process an interleaved block; returns 1 for valid arguments, 0 for invalid arguments
- `sdm_reset(context, sourcefreq)` — reset a context's modulation history
- `sdm_close(context)` — free the context
- `sdm_get_peak_forced_bits()` and `sdm_get_leakage_stats()` — obtain guard statistics

**Important:** In this implementation, `sourcefreq` is currently ignored by `sdm_init()` and `sdm_reset()`. **The encoder does not perform sample-rate conversion.** Feed it PCM that has already been resampled to the target DSD bit-clock rate.

Example for interleaved stereo PCM already at DSD64's 2,822,400 Hz rate:

```cpp
#include "sdm.h"

sdm_t *contexts[2] = {
    sdm_init(2822400, 2822400),
    sdm_init(2822400, 2822400)
};

if (contexts[0] && contexts[1]) {
    // input and output each contain frames * 2 interleaved floats.
    // output may also point to input for in-place processing.
    const int ok = sdm_process_interleaved_channels(
        2, contexts, input, output, frames);

    // Each output float is -1.0f or +1.0f; pack sign bits separately.
    (void)ok;
}

sdm_close(contexts[0]);
sdm_close(contexts[1]);
```

Retain each context across consecutive blocks of the same audio stream so that feedback and output history remain continuous. Do not share a context between channels or process the same context concurrently from multiple threads.

## Building and integration

This is a **C++ source component**, not a complete executable. The supplied files are `sdm.cpp`, `sdm-avx4.cpp`, `sdm-filters.h`, `sdm-private.h`, `sdm.h`, `cpu.cpp` and `cpu.h`.

- Compile `sdm.cpp` and `cpu.cpp` for the application's baseline x86/x64 target (SSE2 is available on x64)
- Compile `sdm-avx4.cpp` separately with **AVX1** enabled, not AVX2/FMA3
- Preserve the intended floating-point operation ordering; avoid aggressive floating-point reassociation/FMA contraction if bitstream equivalence matters
- Link all three `.cpp` translation units into the application or library
- On MSVC, use `/arch:AVX` only for `sdm-avx4.cpp`; on GCC/Clang, use `-mavx -mno-fma` for that source file

These source files do not contain a build-system project or file-format writer. Integration and packaging are the responsibility of the embedding application.

## Original authorship and licensing

**Original sigma-delta modulation work:** Copyright © 2015 **Måns Rullgård** (`mans@mansr.com`), as stated in the original SoX `sdm.c` and retained in the Poikosoft `sdm.cpp` and `sdm.h` notices.

**Poikosoft modifications and additional implementation:** © 2026 **Jukka Poikolainen / Poikosoft**, including the specialized 8th-order CRFF pipeline, revised coefficients and filter selection, multichannel SIMD kernels, PCM input handling, feedback-consistent modulation guard, adaptive state protection, and integration API.

The revised architecture and substantial new code do **not** erase the provenance or copyright of reused upstream material. Both sets of contributions are acknowledged.

## License

This derivative library is distributed under the **GNU Lesser General Public License, version 2.1 or (at your option) any later version** (`LGPL-2.1-or-later`). The full license text is provided in [`LICENSE`](LICENSE).

The original SoX-derived code retains its original copyright and LGPL notices. Poikosoft's modifications to that code are distributed under the same LGPL terms. If you redistribute a binary incorporating this LGPL-covered component, comply with the applicable source-code, notices and relinking requirements.

This project is independently maintained by Poikosoft and is **not an official SoX or Måns Rullgård release**.

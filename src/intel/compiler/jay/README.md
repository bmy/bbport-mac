# Jay

Jay is a modern compiler for Intel's Xe GPUs. Jay uses a novel modelling of the
Intel register file enabling a straightforward implementation of a decoupled
SSA-based register allocator, using a Colombet allocator with a Braun-Hack
spiller. As seen at [XDC2026 in
Toronto](https://indico.freedesktop.org/event/12/contributions/527/).

It currently supports Xe2 (Lunar Lake and Battlemage) and Xe3 (Pantherlake).
Additional hardware support is in progress.

Jay is enabled by default on these platforms. For debugging, set
`INTEL_DEBUG=no-jay` to get the old brw compiler.

## Contribution policy

We're committed to delivering high-quality software and nurturing healthy
software teams. To that end, we have the following guidelines for contributors.

* Follow the [upstream Mesa
  policy](https://docs.mesa3d.org/submittingpatches.html#expectations-on-contributors).
* Submit high-quality patches, regardless of what tools you used.
* You may use LLMs to improve quality (patch review & static analysis).
* Please do not use LLMs to try to improve velocity at the expense of quality
  (**no vibecoding**).
* Maintainability matters more than short-term crunch in the long run.
* People matter more than code.
* This policy will continue to evolve over time.

Selected background reading:

* <https://daniel.haxx.se/blog/2026/04/22/high-quality-chaos/>
* <https://dl.acm.org/doi/epdf/10.1145/3442188.3445922>
* <https://sourcehut.org/blog/2026-08-27-tos-changes-and-llms/>
* <https://kusma.xyz/blog/2026/03/26/open-source-and-ai/>

## Advice for debugging

Sometimes, a game will work on our old compiler (brw) but not Jay. In that case,
you can follow this checklist to narrow the issue:

Reposting the checklist earlier this summer:

* does it reproduce locally on Xe2 or Xe3 with upstream Mesa?
* does it go away if you disable Jay? (`INTEL_DEBUG=no-jay` env var)
* does it reproduce in a debug build of Mesa?
* in a debug build, are there any assertion/validation failures)?
* ...if so, can you capture a fossil?
* ...can you reproduce the assertion failure on the fossil?
* if not, does the issue go away with some combination of Jay debug flags?
  (`JAY_DEBUG=sync,strict,noopt,nosched` is a "safe" set).
* can you bisect it down to a specific shader hash, by selecting between brw and
  Jay with `nir_shader_bisect_select()` and the `nir_shader_bisect.py` script?
  See the comment in the `intel_use_jay()` function in
  `src/intel/dev/intel_debug.c` for how to enable this.
* can you dump the affected shader with INTEL_DEBUG=(vs|fs|cs)?
* can you dump the assembly of that shader with brw?
* can you diff the before/after assembly to see what Jay is doing different?
* if the issue goes away with some JAY_DEBUG flag, can you isolate which one?
* ..and then dump and diff the Jay assembly with/without that flag?

This should get you from "this trace fails" to "this specific shader fails and
here's the specific miscompiling assembly to study". From there, the issue is
hopefully easy to spot. If you get stuck there, send those results to a Jay
developer for further analysis.

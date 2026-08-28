# Third-party provenance

GKX is distributed under the MIT license in `LICENSE`. This file records
third-party code that has been translated, adapted, or retained in substantial
part inside GKX. Scientific references used to implement equations are cited
in the documentation and are not, by themselves, copied software.

## GX geometry helpers

Repository: <https://bitbucket.org/gyrokinetics/gx>

GKX commit `58ff86c859c1955faecdb3291745bc1d7712852a` introduced an
internal geometry backend explicitly described as progressively ported from
GX. The imported implementation translated helper logic from GX's Miller and
VMEC/PyVMEC geometry scripts into NumPy/JAX code. Later GKX commits renamed,
split, and refactored that implementation; current descendants live principally
under `src/gkx/geometry/`.

The precise GX revision used for the original translation was not recorded in
that commit. GX revision `96e42569fa9ffc392a46ddedddf5d24a27b8de39` is the
last revision in the available GX history before GKX's 1 April 2026 import and
is the reproducible comparison anchor, not a claim of exact source identity.
The detailed function and current-owner inventory is in
`plan/baseline/gkx_1_8_2_gx_provenance.md`.

The following GX license notice is reproduced verbatim from
`docs/License.rst` at that comparison revision (SHA-256
`595c4f55da35557154186acdd719756c373cebb8dc388f63e516c0c8b5e4c0e2`):

```text
Copyright (c) 2011-2023 Noah R. Mandell, William D. Dorland, and the GX team.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

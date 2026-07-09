"""Ad performance agent package.

Sets OMP environment variables before any native library (torch, XGBoost,
LightGBM, sklearn) is loaded.  On macOS ARM64 these libraries each bundle
their own libomp.dylib; two copies with conflicting OMP state cause a
segfault during .fit().

KMP_DUPLICATE_LIB_OK  — tolerate multiple libomp copies in the same process.
OMP_NUM_THREADS       — force single-threaded OMP to avoid thread-pool
                        corruption between the two copies.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

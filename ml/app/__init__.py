import os

# faiss-cpu and torch each bundle their own OpenMP runtime (libomp). On macOS, loading both in
# one process and letting torch run multi-threaded CPU ops segfaults. Limiting OpenMP to one
# thread avoids it. This must run before torch/faiss are imported, hence here.
# CLIP runs on MPS/CUDA where available, so this only slows CPU-only fallbacks.
os.environ.setdefault("OMP_NUM_THREADS", "1")

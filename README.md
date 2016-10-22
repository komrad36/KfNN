Fastest CPU implementation of a brute-force
matcher for 128-float descriptors such as SIFT
in 2NN mode, i.e., a match is returned if the best
match between a query vector and a training vector
is more than a certain threshold ratio
better than the second-best match.
AVX2 and all the SSEs are used to accelerate
computation.

Check out the CUDA version, CUDAKfNN, for significantly
more speed.

KfNN supports both raw floats and packed (as uint8_t).
Just set the 'packed' boolean flag accordingly in the
demo. For CPU packed is faster; for CUDA it's slower.

Float descriptors are slow. Check out my K2NN and
CUDAK2NN projects
for much faster binary description matching. Use a
good binary descriptor such as LATCH where possible.

All functionality is contained in the file KfNN.h.
'main.cpp' is simply a sample test harness
with example usage and performance testing.
// Benchmark: range-checked indirect (gather) access vs the alternatives.
//
// Self-contained (no CMSSW headers) -- the SoA layer does not affect the
// compute-loop codegen, since every variant accesses raw column pointers.
//
// The indices are a RANDOM PERMUTATION, so the indirect variants do a genuine
// scattered gather (not the idx==i case, which would secretly be contiguous
// and give misleading numbers).
//
// Built by scram (uses the project -march=x86-64-v3 = AVX2). Run manually:
//   ./SoAIndexedAccessBench [size] [runs]     e.g. taskset -c 4 ./... 50000000 15
//
// Notes:
//   * pin to one core (taskset) on a many-core machine for stable numbers.
//   * we report the MIN over runs (most reproducible) and the mean.

#include <immintrin.h>

#include <experimental/simd>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>

namespace stdx = std::experimental;

// compiler barrier: forces the write to `p` to be treated as observable, so the
// optimizer cannot delete a timed call or hoist it out of the repeat loop.
static inline void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }

static float *alloc_f(size_t n) {
  size_t bytes = ((n * sizeof(float) + 63) / 64) * 64;
  return static_cast<float *>(std::aligned_alloc(64, bytes));
}
static int *alloc_i(size_t n) {
  size_t bytes = ((n * sizeof(int) + 63) / 64) * 64;
  return static_cast<int *>(std::aligned_alloc(64, bytes));
}

// ---------------------------------------------------------------------------
// Variants. Each writes a[] = y[j]*x[j] + z[j] with j = idx[i] (except direct).
// ---------------------------------------------------------------------------

// (0) DIRECT / contiguous reference: j == i. No gather. Upper bound.
static void direct(const float *x, const float *y, const float *z, const int *, float *a, int n) {
  for (int i = 0; i < n; ++i)
    a[i] = y[i] * x[i] + z[i];
}

// (1) indirect, scalar, per-element range check (== the checked SoA view)
static void ind_scalar_check(const float *x, const float *y, const float *z, const int *idx, float *a, int n) {
  for (int i = 0; i < n; ++i) {
    const int j = idx[i];
    if (j < 0 || j >= n)
      throw std::out_of_range("idx");
    a[i] = y[j] * x[j] + z[j];
  }
}

// (2) indirect, scalar, NO check (== the raw-pointer "Sol" version)
static void ind_scalar_nocheck(const float *x, const float *y, const float *z, const int *idx, float *a, int n) {
  for (int i = 0; i < n; ++i) {
    const int j = idx[i];
    a[i] = y[j] * x[j] + z[j];
  }
}

// (3) indirect, std::simd, vectorized range check (portable, no intrinsics)
static void ind_std(const float *x, const float *y, const float *z, const int *idx, float *a, int n) {
  using vfloat = stdx::native_simd<float>;
  using vint = stdx::rebind_simd_t<int, vfloat>;
  constexpr int W = static_cast<int>(vfloat::size());
  int i = 0;
  for (; i + W <= n; i += W) {
    vint vj(idx + i, stdx::element_aligned);
    auto bad = (vj < vint(0)) || (vj >= vint(n));
    if (stdx::any_of(bad))
      throw std::out_of_range("idx");
    vfloat vx([&](auto k) { return x[vj[k]]; });
    vfloat vy([&](auto k) { return y[vj[k]]; });
    vfloat vz([&](auto k) { return z[vj[k]]; });
    vfloat va = vy * vx + vz;
    va.copy_to(a + i, stdx::element_aligned);
  }
  for (; i < n; ++i) {
    const int j = idx[i];
    if (j < 0 || j >= n)
      throw std::out_of_range("idx");
    a[i] = y[j] * x[j] + z[j];
  }
}

// (4) indirect, manual AVX2 hardware gather + vectorized range check
static void ind_avx2(const float *x, const float *y, const float *z, const int *idx, float *a, int n) {
  const __m256i vzero = _mm256_setzero_si256();
  const __m256i vnm1 = _mm256_set1_epi32(n - 1);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i vj = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(idx + i));
    __m256i bad = _mm256_or_si256(_mm256_cmpgt_epi32(vzero, vj), _mm256_cmpgt_epi32(vj, vnm1));
    if (_mm256_movemask_ps(_mm256_castsi256_ps(bad)))
      throw std::out_of_range("idx");
    __m256 vx = _mm256_i32gather_ps(x, vj, 4);
    __m256 vy = _mm256_i32gather_ps(y, vj, 4);
    __m256 vzz = _mm256_i32gather_ps(z, vj, 4);
    _mm256_storeu_ps(a + i, _mm256_fmadd_ps(vy, vx, vzz));
  }
  for (; i < n; ++i) {
    const int j = idx[i];
    if (j < 0 || j >= n)
      throw std::out_of_range("idx");
    a[i] = y[j] * x[j] + z[j];
  }
}

// ---------------------------------------------------------------------------

using clk = std::chrono::steady_clock;

template <class F>
static void run(const char *name, F fn, float *a, int n, int runs) {
  fn();  // warm-up (pages, caches, branch predictor)
  escape(a);
  double best = 1e300, sum = 0;
  for (int r = 0; r < runs; ++r) {
    auto t0 = clk::now();
    fn();
    escape(a);  // prevent the call from being optimized away / hoisted
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best = std::min(best, ms);
    sum += ms;
  }
  double chk = 0;
  for (int i = 0; i < n; ++i)
    chk += a[i];
  std::printf("  %-24s  min %8.2f ms   mean %8.2f ms   %6.3f ns/elem   [chk %.3e]\n",
              name, best, sum / runs, best * 1e6 / n, chk);
}

int main(int argc, char **argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : 50000000;
  int runs = (argc > 2) ? std::atoi(argv[2]) : 15;

  float *x = alloc_f(n), *y = alloc_f(n), *z = alloc_f(n), *a = alloc_f(n);
  int *idx = alloc_i(n);
  if (!x || !y || !z || !a || !idx) {
    std::fprintf(stderr, "allocation failed for n=%d (need ~%.1f GB)\n", n, 5.0 * n * 4 / 1e9);
    return 1;
  }

  for (int i = 0; i < n; ++i) {
    x[i] = float(i) * 1.0f;
    y[i] = float(i) * 2.0f;
    z[i] = float(i) * 3.0f;
    idx[i] = i;
  }
  // random permutation -> genuinely scattered access
  std::mt19937 rng(12345);
  std::shuffle(idx, idx + n, rng);

  std::printf("n = %d elements (%.2f GB total), runs = %d, SIMD width = %zu floats\n\n",
              n, 5.0 * n * 4 / 1e9, runs, stdx::native_simd<float>::size());

  std::printf("INDIRECT (scattered, random-permutation idx):\n");
  run("scalar + check", [&] { ind_scalar_check(x, y, z, idx, a, n); }, a, n, runs);
  run("scalar  no check", [&] { ind_scalar_nocheck(x, y, z, idx, a, n); }, a, n, runs);
  run("std::simd + check", [&] { ind_std(x, y, z, idx, a, n); }, a, n, runs);
  run("AVX2 gather + check", [&] { ind_avx2(x, y, z, idx, a, n); }, a, n, runs);

  std::printf("\nDIRECT (contiguous reference, i.e. j == i):\n");
  run("direct (auto-vec)", [&] { direct(x, y, z, idx, a, n); }, a, n, runs);

  std::free(x);
  std::free(y);
  std::free(z);
  std::free(a);
  std::free(idx);
  return 0;
}

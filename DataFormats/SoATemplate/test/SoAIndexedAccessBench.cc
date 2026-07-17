// Benchmark: a[i] = y[j]*x[j] + z[j].  Matrix = 4 techniques x 2 access patterns.
//   patterns : INDIRECT  j = idx[i]  (random permutation -> scattered gather)
//              DIRECT    j = i        (contiguous)
//   techniques: auto no-check / auto + range check / std::simd + check / AVX2 manual + check
// Self-contained; built by scram with -march=x86-64-v3 (AVX2).
//   ./SoAIndexedAccessBench [size] [runs]     e.g. taskset -c 4 ./... 100000000 20

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

static inline void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }

static float *alloc_f(size_t n) {
  size_t bytes = ((n * sizeof(float) + 63) / 64) * 64;
  return static_cast<float *>(std::aligned_alloc(64, bytes));
}
static int *alloc_i(size_t n) {
  size_t bytes = ((n * sizeof(int) + 63) / 64) * 64;
  return static_cast<int *>(std::aligned_alloc(64, bytes));
}

// ===========================================================================
// INDIRECT : j = idx[i]  (scattered)
// ===========================================================================

static void ind_auto_nocheck(const float *x, const float *y, const float *z, const int *idx, float *a, int n) {
  for (int i = 0; i < n; ++i) {
    const int j = idx[i];
    a[i] = y[j] * x[j] + z[j];
  }
}

static void ind_auto_check(const float *x, const float *y, const float *z, const int *idx, float *a, int n) {
  for (int i = 0; i < n; ++i) {
    const int j = idx[i];
    if (j < 0 || j >= n)
      throw std::out_of_range("idx");
    a[i] = y[j] * x[j] + z[j];
  }
}

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
    (vy * vx + vz).copy_to(a + i, stdx::element_aligned);
  }
  for (; i < n; ++i) {
    const int j = idx[i];
    if (j < 0 || j >= n)
      throw std::out_of_range("idx");
    a[i] = y[j] * x[j] + z[j];
  }
}

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

// ===========================================================================
// DIRECT : j = i  (contiguous). Same four techniques; here the check is on the
// loop index, so it is provably dead -> elided, and every variant still
// vectorizes to packed loads. This is the point of the study.
// ===========================================================================

static void dir_auto_nocheck(const float *x, const float *y, const float *z, const int *, float *a, int n) {
  for (int i = 0; i < n; ++i)
    a[i] = y[i] * x[i] + z[i];
}

static void dir_auto_check(const float *x, const float *y, const float *z, const int *, float *a, int n) {
  for (int i = 0; i < n; ++i) {
    if (i < 0 || i >= n)
      throw std::out_of_range("idx");
    a[i] = y[i] * x[i] + z[i];
  }
}

static void dir_std(const float *x, const float *y, const float *z, const int *, float *a, int n) {
  using vfloat = stdx::native_simd<float>;
  using vint = stdx::rebind_simd_t<int, vfloat>;
  constexpr int W = static_cast<int>(vfloat::size());
  int i = 0;
  for (; i + W <= n; i += W) {
    vint vj([&](auto k) { return i + static_cast<int>(k); });
    auto bad = (vj < vint(0)) || (vj >= vint(n));
    if (stdx::any_of(bad))
      throw std::out_of_range("idx");
    vfloat vx(x + i, stdx::element_aligned);
    vfloat vy(y + i, stdx::element_aligned);
    vfloat vz(z + i, stdx::element_aligned);
    (vy * vx + vz).copy_to(a + i, stdx::element_aligned);
  }
  for (; i < n; ++i) {
    if (i < 0 || i >= n)
      throw std::out_of_range("idx");
    a[i] = y[i] * x[i] + z[i];
  }
}

static void dir_avx2(const float *x, const float *y, const float *z, const int *, float *a, int n) {
  const __m256i iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256i vzero = _mm256_setzero_si256();
  const __m256i vnm1 = _mm256_set1_epi32(n - 1);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i vj = _mm256_add_epi32(_mm256_set1_epi32(i), iota);
    __m256i bad = _mm256_or_si256(_mm256_cmpgt_epi32(vzero, vj), _mm256_cmpgt_epi32(vj, vnm1));
    if (_mm256_movemask_ps(_mm256_castsi256_ps(bad)))
      throw std::out_of_range("idx");
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    __m256 vz = _mm256_loadu_ps(z + i);
    _mm256_storeu_ps(a + i, _mm256_fmadd_ps(vy, vx, vz));
  }
  for (; i < n; ++i) {
    if (i < 0 || i >= n)
      throw std::out_of_range("idx");
    a[i] = y[i] * x[i] + z[i];
  }
}

// ---------------------------------------------------------------------------

using clk = std::chrono::steady_clock;

template <class F>
static void run(const char *name, F fn, float *a, int n, int runs) {
  fn();  // warm-up
  escape(a);
  double best = 1e300, sum = 0;
  for (int r = 0; r < runs; ++r) {
    auto t0 = clk::now();
    fn();
    escape(a);
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best = std::min(best, ms);
    sum += ms;
  }
  double chk = 0;
  for (int i = 0; i < n; ++i)
    chk += a[i];
  std::printf("  %-22s  min %8.2f ms   mean %8.2f ms   %6.3f ns/elem   [chk %.3e]\n",
              name, best, sum / runs, best * 1e6 / n, chk);
}

int main(int argc, char **argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : 100000000;  // ~2 GB across 5 arrays
  int runs = (argc > 2) ? std::atoi(argv[2]) : 20;

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
  std::mt19937 rng(12345);
  std::shuffle(idx, idx + n, rng);

  std::printf("n = %d elements (%.2f GB total), runs = %d, SIMD width = %zu floats\n\n",
              n, 5.0 * n * 4 / 1e9, runs, stdx::native_simd<float>::size());

  std::printf("INDIRECT (scattered, j = idx[i]):\n");
  run("auto no-check", [&] { ind_auto_nocheck(x, y, z, idx, a, n); }, a, n, runs);
  run("auto + check", [&] { ind_auto_check(x, y, z, idx, a, n); }, a, n, runs);
  run("std::simd + check", [&] { ind_std(x, y, z, idx, a, n); }, a, n, runs);
  run("AVX2 manual + check", [&] { ind_avx2(x, y, z, idx, a, n); }, a, n, runs);

  std::printf("\nDIRECT (contiguous, j = i):\n");
  run("auto no-check", [&] { dir_auto_nocheck(x, y, z, idx, a, n); }, a, n, runs);
  run("auto + check", [&] { dir_auto_check(x, y, z, idx, a, n); }, a, n, runs);
  run("std::simd + check", [&] { dir_std(x, y, z, idx, a, n); }, a, n, runs);
  run("AVX2 manual + check", [&] { dir_avx2(x, y, z, idx, a, n); }, a, n, runs);

  std::free(x);
  std::free(y);
  std::free(z);
  std::free(a);
  std::free(idx);
  return 0;
}

#include <experimental/simd>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "SoASimdView.h"

namespace stdx = std::experimental;

GENERATE_SOA_LAYOUT(InputLayoutTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(int, idx))
GENERATE_SOA_LAYOUT(ResultLayoutTemplate, SOA_COLUMN(float, a))

using InputLayout = InputLayoutTemplate<>;
using ResultLayout = ResultLayoutTemplate<>;
using ResultView = ResultLayout::View;
using EnabledView = InputLayout::ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::enabled>;

using SimdInput = SimdInputView<EnabledView>;
using SimdResult = SimdResultView<ResultView>;

using vfloat = SimdInput::vfloat;
using vint = SimdInput::vint;
static constexpr int W = SimdInput::width;

static int failures = 0;
static void report(const char *name, bool ok) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok)
    ++failures;
}

// Owns an InputLayout + ResultLayout of `size`, filled x[i]=i, y[i]=2i, z[i]=3i, idx[i]=i.
struct Fixture {
  std::unique_ptr<std::byte, decltype(std::free) *> inMem, outMem;
  InputLayout inLayout;
  ResultLayout outLayout;
  SimdInput sview;
  SimdResult sresult;

  explicit Fixture(int size)
      : inMem{reinterpret_cast<std::byte *>(std::aligned_alloc(InputLayout::alignment, InputLayout::computeDataSize(size))),
              std::free},
        outMem{reinterpret_cast<std::byte *>(std::aligned_alloc(ResultLayout::alignment, ResultLayout::computeDataSize(size))),
               std::free},
        inLayout{inMem.get(), size},
        outLayout{outMem.get(), size},
        sview{inLayout},
        sresult{outLayout} {
    EnabledView &base = sview;
    for (int i = 0; i < size; ++i)
      base[i] = {float(i), float(i) * 2.f, float(i) * 3.f, i};
  }
};

int main() {
  std::printf("SoASimdView unit tests (W = %d)\n\n", W);

  // 1) writing out of bound throws
  {
    Fixture fx(20);
    bool threw = false;
    try {
      fx.sview[20] = {vfloat(0.f), vfloat(0.f), vfloat(0.f), vint(0)};
    } catch (const std::out_of_range &) {
      threw = true;
    }
    report("1 write out of bound throws", threw);
  }

  // 2) loading out of bound throws
  {
    Fixture fx(20);
    const SimdInput &c = fx.sview;
    bool threw = false;
    try {
      (void)c[20];
    } catch (const std::out_of_range &) {
      threw = true;
    }
    report("2 load out of bound throws", threw);
  }

  // 3) write [i] on a size not multiple of W: tail lanes must not be written
  {
    const int size = 2 * W + 3;      // last chunk at 2W has 3 valid lanes
    Fixture fx(size);
    const int base = 2 * W;
    float *xp = static_cast<EnabledView &>(fx.sview).metadata().addressOf_x();
    for (int k = 0; k < W; ++k)
      xp[base + k] = -999.f;

    fx.sview[base] = {vfloat([](auto k) { return 100.f + int(k); }), vfloat(0.f), vfloat(0.f), vint(0)};

    bool ok = true;
    for (int k = 0; k < 3; ++k)
      ok = ok && (xp[base + k] == 100.f + k);
    for (int k = 3; k < W; ++k)
      ok = ok && (xp[base + k] == -999.f);
    report("3 partial write leaves tail lanes untouched", ok);
  }

  // 4) load [i] in the tail: valid lanes readable, trailing lanes must NOT be
  //    accessible -- reading lane k >= lanesAt(base) has to throw, not hand
  //    back a bogus value the user could feed into a computation.
  {
    const int size = 2 * W + 3;
    Fixture fx(size);
    const int base = 2 * W;
    auto e = static_cast<const SimdInput &>(fx.sview)[base];

    bool ok = true;
    for (int k = 0; k < 3; ++k)
      ok = ok && (e.idx()[k] == base + k);      // valid lanes still readable
    for (int k = 3; k < W; ++k) {
      bool threw = false;
      try {
        (void)e.idx()[k];                        // trailing lane
      } catch (const std::out_of_range &) {
        threw = true;
      }
      ok = ok && threw;                          // access must be refused
    }
    report("4 tail load: trailing lanes not accessible", ok);
  }

  std::printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}

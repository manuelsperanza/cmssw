#ifndef DataFormats_SoATemplate_interface_SoASimdMasked_h
#define DataFormats_SoATemplate_interface_SoASimdMasked_h

#include <experimental/simd>

// Masked SIMD load/store: touch only the first `lanes` elements, so a partial
// final chunk never reads or writes past the valid data (loads zero-fill the
// inactive lanes, stores skip them). A full chunk takes the plain path.
namespace soa_simd {
  namespace stdx = std::experimental;
  using vfloat = stdx::native_simd<float>;
  using vint = stdx::rebind_simd_t<int, vfloat>;
  inline constexpr int width = static_cast<int>(vfloat::size());

  inline vfloat maskedLoad(const float *p, int lanes) {
    vfloat v(0.f);
    if (lanes >= width) {
      v.copy_from(p, stdx::element_aligned);
      return v;
    }
    const vfloat lane([](auto k) { return static_cast<float>(k); });
    stdx::where(lane < vfloat(static_cast<float>(lanes)), v).copy_from(p, stdx::element_aligned);
    return v;
  }

  inline vint maskedLoad(const int *p, int lanes) {
    vint v(0);
    if (lanes >= width) {
      v.copy_from(p, stdx::element_aligned);
      return v;
    }
    const vint lane([](auto k) { return static_cast<int>(k); });
    stdx::where(lane < vint(lanes), v).copy_from(p, stdx::element_aligned);
    return v;
  }

  inline void maskedStore(float *p, const vfloat &v, int lanes) {
    if (lanes >= width) {
      v.copy_to(p, stdx::element_aligned);
      return;
    }
    const vfloat lane([](auto k) { return static_cast<float>(k); });
    stdx::where(lane < vfloat(static_cast<float>(lanes)), v).copy_to(p, stdx::element_aligned);
  }

  inline void maskedStore(int *p, const vint &v, int lanes) {
    if (lanes >= width) {
      v.copy_to(p, stdx::element_aligned);
      return;
    }
    const vint lane([](auto k) { return static_cast<int>(k); });
    stdx::where(lane < vint(lanes), v).copy_to(p, stdx::element_aligned);
  }
}  // namespace soa_simd

#endif  // DataFormats_SoATemplate_interface_SoASimdMasked_h

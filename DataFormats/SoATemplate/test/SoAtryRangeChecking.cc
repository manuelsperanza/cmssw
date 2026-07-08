#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(InputLayoutTemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))

using InputLayout = InputLayoutTemplate<>;

using EnabledView = InputLayout::ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::enabled>;
using DisabledView = InputLayout::ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::disabled>;

int main() {
  constexpr int size = 1024;

  std::unique_ptr<std::byte, decltype(std::free) *> inputMem{
      reinterpret_cast<std::byte *>(aligned_alloc(InputLayout::alignment, InputLayout::computeDataSize(size))),
      std::free};

  InputLayout input{inputMem.get(), size};

  EnabledView enabledView{input};
  DisabledView disabledView{input};

  for (int i = 0; i < size; ++i) {
    enabledView[i] = {float(i), float(i) * 2.f, float(i) * 3.f};
  }

  std::cout << "in-bounds access, enabledView[" << size - 1 << "].x() = " << enabledView[size - 1].x() << '\n';

  std::cout << "\n-- RangeChecking::enabled, accessing index " << size << " (one past the end) --\n";
  try {
    std::cout << "enabledView[" << size << "].x() = " << enabledView[size].x() << '\n';
  } catch (const std::out_of_range &e) {
    std::cout << "caught std::out_of_range: " << e.what() << '\n';
  }

  std::cout << "\n-- RangeChecking::disabled, accessing index " << size << " (one past the end) --\n";
  std::cout << "disabledView[" << size << "].x() = " << disabledView[size].x() << '\n';
  std::cout << "(no exception was thrown: this just read whatever memory happens to sit past the buffer)\n";

  return EXIT_SUCCESS;
}

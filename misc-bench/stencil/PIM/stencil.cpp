// Test: C++ version of the stencil
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#include <type_traits>
#include <queue>
#include <random>
#include <limits>
#include <algorithm>
#include <list>
#include <cstring>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t gridWidth;
  uint64_t gridHeight;
  uint64_t stencilWidth;
  uint64_t stencilHeight;
  const char *configFile;
  const char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./stencil.out [options]"
          "\n"
          "\n    -x    board width (default=2048 elements)"
          "\n    -y    board height (default=2048 elements)"
          "\n    -l    vertical stencil size, must be odd (default=3)"
          "\n    -w    horizontal stencil size, must be odd (default=3)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing a 2d array (default=random)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.gridWidth = 2048;
  p.gridHeight = 2048;
  p.stencilWidth = 3;
  p.stencilHeight = 3;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:x:y:l:w:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'x':
      p.gridWidth = strtoull(optarg, NULL, 0);
      break;
    case 'y':
      p.gridHeight = strtoull(optarg, NULL, 0);
      break;
    case 'l':
      p.stencilHeight = strtoull(optarg, NULL, 0);
      break;
    case 'w':
      p.stencilWidth = strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

//! @brief  Sums the elements to the left and right within a vector according to the horizontal stencil width
//! @tparam  StencilTypeHost  The host datatype for stencil
//! @tparam  StencilTypePIM  The PIM datatype for stencil
//! @param[in]  src  The vector to sum
//! @param[in]  stencilWidth  The horizontal width of the stencil
//! @param[in]  toAssociate  A PIM Object to associate the added data with
//! @return  The sumed PIM row
template <typename StencilTypeHost, PimDataType StencilTypePIM>
std::vector<PimObjId> createShiftedStencilRows(const std::vector<StencilTypeHost> &src, const uint64_t stencilWidth,
                                               const uint64_t numLeft, const PimObjId toAssociate) {
  PimStatus status;

  std::vector<PimObjId> result(stencilWidth);

  for(uint64_t i=0; i<result.size(); ++i) {
    result[i] = pimAllocAssociated(toAssociate, StencilTypePIM);
    assert(result[i] != -1);
  }

  status = pimCopyHostToDevice((void *)src.data(), result[numLeft]);
  assert (status == PIM_OK);

  for(uint64_t i=numLeft; i>0; --i) {
    status = pimCopyObjectToObject(result[i], result[i-1]);
    assert (status == PIM_OK);

    status = pimShiftElementsLeft(result[i-1]);
    assert (status == PIM_OK);
  }

  for(uint64_t i=numLeft+1; i<result.size(); ++i) {
    status = pimCopyObjectToObject(result[i-1], result[i]);
    assert (status == PIM_OK);

    status = pimShiftElementsLeft(result[i]);
    assert (status == PIM_OK);
  }

  return result;
}

//! @brief  Determines the corresponding PIM datatype from a host datatype
//! @tparam  StencilTypeHost  The host datatype
//! @return  The corresponding PIM datatype
template <typename StencilTypeHost>
constexpr PimDataType getPIMTypeFromHostType() {
  if constexpr (std::is_same_v<StencilTypeHost, int8_t>) {
    return PIM_INT8;
  } else if constexpr (std::is_same_v<StencilTypeHost, int16_t>) {
    return PIM_INT16;
  } else if constexpr (std::is_same_v<StencilTypeHost, int32_t>) {
    return PIM_INT32;
  } else if constexpr (std::is_same_v<StencilTypeHost, int64_t>) {
    return PIM_INT64;
  } else if constexpr (std::is_same_v<StencilTypeHost, uint8_t>) {
    return PIM_UINT8;
  } else if constexpr (std::is_same_v<StencilTypeHost, uint16_t>) {
    return PIM_UINT16;
  } else if constexpr (std::is_same_v<StencilTypeHost, uint32_t>) {
    return PIM_UINT32;
  } else if constexpr (std::is_same_v<StencilTypeHost, uint64_t>) {
    return PIM_UINT64;
  } else {
    // This will only trigger if StencilTypeHost is not one of the supported types
    static_assert(!std::is_same_v<StencilTypeHost, StencilTypeHost>, "Error: Unsupported datatype for stencil, aborting");
    return PIM_INT8; // Still need a return, but it will never be reached
  }
}

//! @brief  Computes a stencil pattern over a 2d array
//! @tparam  StencilTypeHost  The host datatype for stencil
//! @param[in]  srcHost  The input stencil grid
//! @param[in]  dstHost  The resultant stencil grid
//! @param[in]  stencilPattern  The stencil pattern to apply
//! @param[in]  numLeft  The number of elements to the left of the output element in the stencil pattern
//! @param[in]  numAbove  The number of elements above the output element in the stencil pattern
template <typename StencilTypeHost>
void stencil(const std::vector<std::vector<StencilTypeHost>> &srcHost, std::vector<std::vector<StencilTypeHost>> &dstHost,
             const std::vector<std::vector<StencilTypeHost>> &stencilPattern, const uint64_t numLeft, const uint64_t numAbove) {
  PimStatus status;
  
  constexpr PimDataType StencilTypePIM = getPIMTypeFromHostType<StencilTypeHost>();
  
  assert(!srcHost.empty());
  assert(!srcHost[0].empty());
  assert(srcHost.size() == dstHost.size());
  assert(srcHost[0].size() == dstHost[0].size());
  assert(!stencilPattern.empty());
  assert(!stencilPattern[0].empty());
  assert(stencilPattern.size() > numAbove);
  assert(stencilPattern[0].size() > numLeft);

  constexpr uint64_t bitsPerElement = 32;
  const uint64_t gridHeight = srcHost.size();
  const uint64_t gridWidth = srcHost[0].size();
  const uint64_t stencilHeight = stencilPattern.size();
  const uint64_t stencilWidth = stencilPattern[0].size();
  const uint64_t numRight = stencilWidth - numLeft - 1;
  const uint64_t numBelow = stencilHeight - numAbove - 1;
  
  // PIM API only supports passing scalar values through uint64_t
  const std::vector<std::vector<uint64_t>> stencilPatternConverted(stencilHeight);
  for(uint64_t y=0; y<stencilHeight; ++y) {
    stencilPatternConverted[y].resize(stencilWidth);
    for(uint64_t x=0; x<stencilWidth; ++x) {
      uint32_t tmp;
      std::memcpy(&tmp, &stencilPattern[y][x], sizeof(float));
      stencilPatternConverted[y][x] = static_cast<uint64_t>(tmp);
    }
  }

  PimObjId resultPim = pimAlloc(PIM_ALLOC_AUTO, gridWidth, StencilTypePIM);
  assert(resultPim != -1);

  PimObjId tempPim = pimAllocAssociated(resultPim, StencilTypePIM);
  assert(tempPim != -1);

  std::list<std::vector<PimObjId>> shiftedRows;

  for(uint64_t i=0; i<stencilHeight-1; ++i) {
    shiftedRows.push_back(createShiftedStencilRows(srcHost[i], stencilWidth, numLeft, resultPim));
  }

  uint64_t nextRowToRemove = 0;
  uint64_t nextRowToAdd = stencilHeight-1;

  for(uint64_t row=numAbove; row<gridHeight-numBelow; ++row) {
    shiftedRows.push_back(createShiftedStencilRows(srcHost[nextRowToAdd], stencilWidth, numLeft, resultPim));
    ++nextRowToAdd;

    uint64_t stencilY = 0;
    for(std::vector<PimObjId> &shiftedRow : shiftedRows) {
      for(uint64_t stencilX = 0; stencilX < stencilWidth; ++stencilX) {
        status = pimMulScalar(shiftedRow[stencilX], tempPim, stencilPatternConverted[stencilY][stencilX]);
        assert (status == PIM_OK);

        status = pimAdd(resultPim, tempPim, resultPim);
        assert (status == PIM_OK);
      }
      ++stencilY;
    }

    status = pimCopyDeviceToHost(resultPim, (void *) dstHost[row].data());
    assert (status == PIM_OK);
    
    for(PimObjId objToFree : shiftedRows.front()) {
      pimFree(objToFree);
    }
    shiftedRows.pop_front();
    ++nextRowToRemove;
  }

  while(!shiftedRows.empty()) {
    for(PimObjId objToFree : shiftedRows.front()) {
      pimFree(objToFree);
    }
    shiftedRows.pop_front();
  }
}

template <typename T>
T getWithDefault(int64_t i, int64_t j, std::vector<std::vector<T>> &x) {
  if(i >= 0 && i < static_cast<int64_t>(x.size()) && j >= 0 && j < static_cast<int64_t>(x[0].size())) {
    return x[i][j];
  }
  return 0;
}

int main(int argc, char* argv[])
{
  using StencilTypeHost = int32_t;
  static_assert(std::is_integral_v<StencilTypeHost>, "Error: Stencil Type must be an integer type");
  struct Params params = getInputParams(argc, argv);

  if(params.stencilHeight % 2 == 0) {
    std::cerr << "Error: Stencil height must be odd, aborting" << std::endl;
    return 1;
  }

  if(params.stencilWidth % 2 == 0) {
    std::cerr << "Error: Stencil width must be odd, aborting" << std::endl;
    return 1;
  }

  std::cout << "Running PIM stencil for board: " << params.gridWidth << "x" << params.gridHeight << std::endl;
  std::vector<std::vector<StencilTypeHost>> x, y;
  if (params.inputFile == nullptr)
  {
    x.resize(params.gridHeight);
    for(size_t i=0; i<x.size(); ++i) {
      x[i].resize(params.gridWidth);
    }

    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<StencilTypeHost> dist(std::numeric_limits<StencilTypeHost>::min(), std::numeric_limits<StencilTypeHost>::max());
        
        #pragma omp for
        for(size_t i=0; i<params.gridHeight; ++i) {
          for(size_t j=0; j<params.gridWidth; ++j) {
            x[i][j] = dist(gen);
          }
        }
    }
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  y.resize(x.size());
  for(size_t i=0; i<y.size(); ++i) {
    y[i].resize(x[0].size());
  }

  stencil(x, y, params.stencilWidth, params.stencilHeight);

  if (params.shouldVerify) 
  {
    bool ok = true;
    const int64_t numElementsPerSide = static_cast<int64_t>(params.stencilWidth >> 1);
    const int64_t numElementsPerTopBot = static_cast<int64_t>(params.stencilHeight >> 1);
    const uint64_t stencilArea = params.stencilHeight * params.stencilWidth;
#pragma omp parallel for
    for(uint64_t i=0; i<y.size(); ++i) {
      for(uint64_t j=0; j<y[0].size(); ++j) {
        StencilTypeHost resCPU = 0;
        for(int64_t offsetY=-numElementsPerTopBot; offsetY<=numElementsPerTopBot; ++offsetY) {
          for(int64_t offsetX=-numElementsPerSide; offsetX<=numElementsPerSide; ++offsetX) {
            resCPU += getWithDefault(static_cast<int64_t>(i) + offsetY, static_cast<int64_t>(j) + offsetX, x);
          }
        }
        if constexpr (std::is_signed_v<StencilTypeHost>) {
          resCPU /= static_cast<int64_t>(stencilArea);
        } else {
          resCPU /= static_cast<uint64_t>(stencilArea);
        }
        if (resCPU != y[i][j])
        {
          #pragma omp critical
          {
            if constexpr (std::is_signed_v<StencilTypeHost>) {
              std::cout << "Wrong answer: " << static_cast<int64_t>(y[i][j]) << " (expected " << static_cast<int64_t>(resCPU) << ")" << std::endl;
            } else {
              std::cout << "Wrong answer: " << static_cast<uint64_t>(y[i][j]) << " (expected " << static_cast<uint64_t>(resCPU) << ")" << std::endl;
            }
            ok = false;
          }
        }
      }
    }
    if(ok) {
      std::cout << "Correct for stencil!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
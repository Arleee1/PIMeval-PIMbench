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
PimObjId sumStencilRow(const std::vector<StencilTypeHost> &src, const uint64_t stencilWidth, const PimObjId toAssociate) {
  const uint64_t numElementsEachSide = stencilWidth >> 1;
  
  PimStatus status;

  PimObjId mid = pimAllocAssociated(toAssociate, StencilTypePIM);
  assert(mid != -1);

  status = pimCopyHostToDevice((void *)src.data(), mid);
  assert (status == PIM_OK);  

  if(numElementsEachSide == 0) {
    return mid;
  }

  PimObjId dst = pimAllocAssociated(toAssociate, StencilTypePIM);
  assert(dst != -1);

  PimObjId other = pimAllocAssociated(toAssociate, StencilTypePIM);
  assert(other != -1);
  
  status = pimCopyObjectToObject(mid, other);
  assert (status == PIM_OK);

  status =  pimShiftElementsRight(other);
  assert (status == PIM_OK);

  status = pimAdd(mid, other, dst);
  assert (status == PIM_OK);

  for(uint64_t shiftIter=1; shiftIter<numElementsEachSide; ++shiftIter) {
    status =  pimShiftElementsRight(other);
    assert (status == PIM_OK);

    status = pimAdd(dst, other, dst);
    assert (status == PIM_OK);
  }

  status = pimCopyObjectToObject(mid, other);
  assert (status == PIM_OK);

  for(uint64_t shiftIter=0; shiftIter<numElementsEachSide; ++shiftIter) {
    status =  pimShiftElementsLeft(other);
    assert (status == PIM_OK);

    status = pimAdd(dst, other, dst);
    assert (status == PIM_OK);
  }

  pimFree(mid);
  pimFree(other);

  return dst;
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

//! @brief  Computes the stencil average for a rectangle around each element
//! @tparam  StencilTypeHost  The host datatype for stencil
//! @param[in]  srcHost  The stencil grid to average
//! @param[in]  dstHost  The averaged grid
//! @param[in]  stencilWidth  The horizontal width of the stencil
//! @param[in]  stencilHeight  The vertical height of the stencil
template <typename StencilTypeHost>
void stencil(const std::vector<std::vector<StencilTypeHost>> &srcHost, std::vector<std::vector<StencilTypeHost>> &dstHost,
             uint64_t stencilWidth, uint64_t stencilHeight) {
  PimStatus status;
  
  constexpr PimDataType StencilTypePIM = getPIMTypeFromHostType<StencilTypeHost>();

  const uint64_t stencilArea = stencilHeight * stencilWidth;
  
  assert(!srcHost.empty());
  assert(!srcHost[0].empty());
  assert(srcHost.size() == dstHost.size());
  assert(srcHost[0].size() == dstHost[0].size());

  size_t height = srcHost.size();
  size_t width = srcHost[0].size();

  // Handle cases when stencil window fully covers grid
  stencilHeight = std::min(stencilHeight, (height<<1)-1);
  stencilWidth = std::min(stencilWidth, (width<<1)-1);

  PimObjId resultPim = pimAlloc(PIM_ALLOC_AUTO, width, StencilTypePIM);
  assert(resultPim != -1);

  // Handle special case
  if(stencilHeight == 1) {
    for(size_t i=0; i<height; ++i) {
      PimObjId summedRow = sumStencilRow<StencilTypeHost, StencilTypePIM>(srcHost[i], stencilWidth, resultPim);

      status = pimDivScalar(summedRow, resultPim, stencilArea);
      assert (status == PIM_OK);

      status = pimCopyDeviceToHost(resultPim, dstHost[i].data());
      assert (status == PIM_OK);

      pimFree(summedRow);
    }

    pimFree(resultPim);
    return;
  }

  PimObjId runningSum = pimAllocAssociated(resultPim, StencilTypePIM);
  assert(runningSum != -1);

  std::queue<PimObjId> pimGrid;

  PimObjId newRow0 = sumStencilRow<StencilTypeHost, StencilTypePIM>(srcHost[0], stencilWidth, resultPim);
  pimGrid.push(newRow0);

  PimObjId newRow1 = sumStencilRow<StencilTypeHost, StencilTypePIM>(srcHost[1], stencilWidth, resultPim);
  pimGrid.push(newRow1);

  status = pimAdd(newRow0, newRow1, runningSum);
  assert (status == PIM_OK);

  const uint64_t numElemsTopBot = stencilHeight >> 1;

  for(size_t i=2; i<=numElemsTopBot; ++i) {
    PimObjId newRow = sumStencilRow<StencilTypeHost, StencilTypePIM>(srcHost[i], stencilWidth, resultPim);
    pimGrid.push(newRow);

    status = pimAdd(runningSum, newRow, runningSum);
    assert (status == PIM_OK);
  }

  uint64_t nextRowToAdd = numElemsTopBot + 1;
  int64_t nextRowToRemove = -static_cast<int64_t>(nextRowToAdd) + 1;

  for(size_t i=0; i<height; ++i) {
    status = pimDivScalar(runningSum, resultPim, stencilArea);
    assert (status == PIM_OK);

    status = pimCopyDeviceToHost(resultPim, dstHost[i].data());
    assert (status == PIM_OK);

    if(nextRowToAdd < height) {
      PimObjId newRow = sumStencilRow<StencilTypeHost, StencilTypePIM>(srcHost[nextRowToAdd], stencilWidth, resultPim);
      status = pimAdd(runningSum, newRow, runningSum);
      assert (status == PIM_OK);
      pimGrid.push(newRow);
    }
    ++nextRowToAdd;

    if(nextRowToRemove >= 0) {
      PimObjId toRemove = pimGrid.front();
      pimGrid.pop();
      status = pimSub(runningSum, toRemove, runningSum);
      assert (status == PIM_OK);
      pimFree(toRemove);
    }
    ++nextRowToRemove;
  }

  pimFree(resultPim);
  pimFree(runningSum);

  while(!pimGrid.empty()) {
    pimFree(pimGrid.front());
    pimGrid.pop();
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
#pragma omp parallel for
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
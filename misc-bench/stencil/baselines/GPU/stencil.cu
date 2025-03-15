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
#include <cmath>
#include <cuda_runtime.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "cuSten.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t gridWidth;
  uint64_t gridHeight;
  uint64_t stencilWidth;
  uint64_t stencilHeight;
  uint64_t numLeft;
  uint64_t numAbove;
  const char *configFile;
  const char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./stencil.out [options]"
          "\n"
          "\n    -x    grid width (default=2048 elements)"
          "\n    -y    grid height (default=2048 elements)"
          "\n    -w    horizontal stencil size (default=3)"
          "\n    -d    vertical stencil size (default=3)"
          "\n    -l    number of elements to the left of the output element for the stencil pattern, must be less than the horizontal stencil size (default=1)"
          "\n    -a    number of elements above the output element for the stencil pattern, must be less than the vertical stencil size (default=1)"
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
  p.numLeft = 1;
  p.numAbove = 1;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:x:y:w:d:l:a:c:i:v:")) >= 0)
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
    case 'w':
      p.stencilWidth = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.stencilHeight = strtoull(optarg, NULL, 0);
      break;
    case 'l':
      p.numLeft = strtoull(optarg, NULL, 0);
      break;
    case 'a':
      p.numAbove = strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't');
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

//! @brief  Computes a stencil pattern over a 2d array
//! @param[in]  srcHost  The input stencil grid
//! @param[in]  dstHost  The resultant stencil grid
//! @param[in]  stencilPattern  The stencil pattern to apply
//! @param[in]  numLeft  The number of elements to the left of the output element in the stencil pattern
//! @param[in]  numAbove  The number of elements above the output element in the stencil pattern
void stencil(const std::vector<std::vector<float>> &srcHost, std::vector<std::vector<float>> &dstHost,
             const std::vector<std::vector<float>> &stencilPattern, const uint64_t numLeft, const uint64_t numAbove) {

  assert(!srcHost.empty());
  assert(!srcHost[0].empty());
  assert(srcHost.size() == dstHost.size());
  assert(srcHost[0].size() == dstHost[0].size());
  assert(!stencilPattern.empty());
  assert(!stencilPattern[0].empty());
  assert(stencilPattern.size() > numAbove);
  assert(stencilPattern[0].size() > numLeft);

  const uint64_t gridHeight = srcHost.size();
  const uint64_t gridWidth = srcHost[0].size();
  const uint64_t stencilHeight = stencilPattern.size();
  const uint64_t stencilWidth = stencilPattern[0].size();
  const uint64_t numBelow = stencilHeight - numAbove - 1;
  const uint64_t numRight = stencilWidth - numLeft - 1;

  constexpr int deviceNumber = 0;
  constexpr int tilesNumber = 4;
  constexpr int blockX = 4;
  constexpr int blockY = 4;

  // cuSten library expects managed memory
  // TODO: Check if this impacts benchmark time
  // TODO: Setup timing
  float* gridInput;
	float* gridOutput;
  const size_t gridSz = gridHeight * gridWidth * sizeof(float);
  cudaMallocManaged(&gridInput, gridSz);
	cudaMallocManaged(&gridOutput, gridSz);

  float* stencilPatternGPU;
  const size_t stencilPatternGPUSz = stencilHeight * stencilWidth * sizeof(float);
  cudaMallocManaged(&stencilPatternGPU, stencilPatternGPUSz);

  for(uint64_t gridY=0; gridY<gridHeight; ++gridY) {
    for(uint64_t gridX=0; gridX<gridWidth; ++gridX) {
      gridInput[gridY * gridWidth + gridX] = srcHost[gridY][gridX];
    }
  }

  for(uint64_t stencilY=0; stencilY<stencilHeight; ++stencilY) {
    for(uint64_t stencilX=0; stencilX<stencilWidth; ++stencilX) {
      stencilPatternGPU[stencilY * stencilWidth + stencilX] = stencilPattern[stencilY][stencilX];
    }
  }


  cuSten_t<float> cuStenHandle;

  cuStenCreate2DXYnp(
    &cuStenHandle,
    deviceNumber,
    tilesNumber,
    gridWidth,
    gridHeight,
    blockX,
    blockY,
    gridOutput,
    gridInput,
    stencilPatternGPU,
    stencilWidth,
    numLeft,
    numRight,
    stencilHeight,
    numAbove,
    numBelow
  );

  cudaDeviceSynchronize();

  cuStenCompute2DXYnp(&cuStenHandle, HOST);

  cudaDeviceSynchronize();

  for(uint64_t gridY=0; gridY<gridHeight; ++gridY) {
    for(uint64_t gridX=0; gridX<gridWidth; ++gridX) {
      dstHost[gridY][gridX] = gridOutput[gridY * gridWidth + gridX];
    }
  }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::cout << "Running GPU stencil for grid: " << params.gridHeight << "x" << params.gridWidth << std::endl;
  std::cout << "Stencil Size: " << params.stencilHeight << "x" << params.stencilWidth << std::endl;
  std::cout << "Num Above: " << params.numAbove << ", Num Left: " << params.numLeft << std::endl;

  std::vector<std::vector<float>> x, y;
  std::vector<std::vector<float>> stencilPattern;
  if (params.inputFile == nullptr)
  {
    // Fill in random grid
    x.resize(params.gridHeight);
    for(size_t i=0; i<x.size(); ++i) {
      x[i].resize(params.gridWidth);
    }

    #pragma omp parallel
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(0.0f, 10000.0f);

      #pragma omp for
      for(size_t i=0; i<params.gridHeight; ++i) {
        for(size_t j=0; j<params.gridWidth; ++j) {
          x[i][j] = dist(gen);
        }
      }
    }

    // Fill in random stencil pattern
    stencilPattern.resize(params.stencilHeight);
    for(size_t i=0; i<stencilPattern.size(); ++i) {
      stencilPattern[i].resize(params.stencilWidth);
    }

    #pragma omp parallel
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);

      #pragma omp for
      for(size_t i=0; i<params.stencilHeight; ++i) {
        for(size_t j=0; j<params.stencilWidth; ++j) {
          stencilPattern[i][j] = dist(gen);
        }
      }
    }
  }
  else
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  y.resize(x.size());
  for(size_t i=0; i<y.size(); ++i) {
    y[i].resize(x[0].size());
  }

  stencil(x, y, stencilPattern, params.numLeft, params.numAbove);

  if (params.shouldVerify)
  {
    bool ok = true;

    // Only compute when stencil is fully in range
    const uint64_t startY = params.numAbove;
    const uint64_t endY = params.gridHeight - (params.stencilHeight - params.numAbove - 1);
    const uint64_t startX = params.numLeft;
    const uint64_t endX = params.gridWidth - (params.stencilWidth - params.numLeft - 1);

    // CPU and GPU results are not exactly the same
    // TODO: Check if this is okay
    constexpr float acceptableDifference = 0.1f;

    #pragma omp parallel for collapse(2)
    for(uint64_t gridY=startY; gridY<endY; ++gridY) {
      for(uint64_t gridX=startX; gridX<endX; ++gridX) {
        float resCPU = 0.0f;
        for(uint64_t stencilY=0; stencilY<params.stencilHeight; ++stencilY) {
          for(uint64_t stencilX=0; stencilX<params.stencilWidth; ++stencilX) {
            resCPU += stencilPattern[stencilY][stencilX] * x[gridY + stencilY - params.numAbove][gridX + stencilX - params.numLeft];
          }
        }
        if (std::abs(resCPU - y[gridY][gridX]) > acceptableDifference)
        {
          #pragma omp critical
          {
            std::cout << std::fixed << std::setprecision(3) << "Wrong answer: " << y[gridY][gridX] << " (expected " << resCPU << ")" << std::endl;
            ok = false;
          }
        }
      }
    }
    if(ok) {
      std::cout << "Correct for stencil!" << std::endl;
    }
  }

  return 0;
}
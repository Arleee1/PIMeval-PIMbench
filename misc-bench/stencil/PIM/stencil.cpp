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
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t width;
  uint64_t height;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./game-of-life.out [options]"
          "\n"
          "\n    -x    board width (default=2048 elements)"
          "\n    -y    board height (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing a game board (default=generates board with random states)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.width = 2048;
  p.height = 2048;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:x:y:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'x':
      p.width = strtoull(optarg, NULL, 0);
      break;
    case 'y':
      p.height = strtoull(optarg, NULL, 0);
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

void game_of_life_row(const std::vector<PimObjId> &pim_board, size_t row_idx, PimObjId tmp_pim_obj, const std::vector<PimObjId>& pim_sums, int old_ind, PimObjId result_obj) {
  size_t mid_idx = 3*row_idx + 1;

  pimAdd(pim_board[mid_idx + 2], pim_board[mid_idx + 3], pim_sums[old_ind]);
  pimAdd(pim_board[mid_idx + 4], pim_sums[old_ind], pim_sums[old_ind]);

  pimAdd(pim_sums[old_ind], pim_sums[(old_ind + 1) % pim_sums.size()], tmp_pim_obj);
  pimAdd(pim_board[mid_idx - 1], tmp_pim_obj, tmp_pim_obj);
  pimAdd(pim_board[mid_idx + 1], tmp_pim_obj, tmp_pim_obj);

  PimStatus status = pimEQScalar(tmp_pim_obj, result_obj, 3);
  assert (status == PIM_OK);

  status = pimEQScalar(tmp_pim_obj, tmp_pim_obj, 2);
  assert (status == PIM_OK);

  status = pimAnd(tmp_pim_obj, pim_board[mid_idx], tmp_pim_obj);
  assert (status == PIM_OK);

  status = pimOr(tmp_pim_obj, result_obj, result_obj);
  assert (status == PIM_OK);
}

// Designed to make it easier to expand to larger stencil areas
struct PIMStencilRow {
  PimObjId left;
  PimObjId mid;
  PimObjId right;

  void addAll(PimObjId dst) {
    PimStatus status;

    status = pimAdd(this->left, this->mid, dst);
    assert (status == PIM_OK);
    status = pimAdd(dst, this->right, dst);
    assert (status == PIM_OK);
  }

  void freeAll() {
    pimFree(left);
    pimFree(mid);
    pimFree(right);
  }
};

//! @brief  Adds a vector to the stencil grid, with copies shifted left and right
//! @tparam  StencilTypeHost  The host datatype for stencil
//! @tparam  StencilTypePIM  The PIM datatype for stencil
//! @param[in]  toAdd  The vector to add to the grid
//! @param[in]  toAssociate  A PIM Object to associate the added data with
//! @return  The added rows
template <typename StencilTypeHost, PimDataType StencilTypePIM>
PIMStencilRow addVectorToGrid(const std::vector<StencilTypeHost> &toAdd, const PimObjId toAssociate) {
  PimStatus status;

  PimObjId left = pimAllocAssociated(mid, StencilTypePIM);
  assert(left != -1);
  PimObjId mid = pimAllocAssociated(toAssociate, StencilTypePIM);
  assert(mid != -1);
  PimObjId right = pimAllocAssociated(mid, StencilTypePIM);
  assert(right != -1);

  status = pimCopyHostToDevice((void *)toAdd.data(), mid);
  assert (status == PIM_OK);
  
  status = pimCopyObjectToObject(mid, left);
  assert (status == PIM_OK);
  status = pimCopyObjectToObject(mid, right);
  assert (status == PIM_OK);


  status = pimShiftElementsRight(left);
  assert (status == PIM_OK);
  status = pimShiftElementsLeft(right);
  assert (status == PIM_OK);

  return {left, mid, right};
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

template <typename StencilTypeHost>
void stencil(const std::vector<std::vector<StencilTypeHost>> &src_host, std::vector<std::vector<StencilTypeHost>> &dst_host) {
  PimStatus status;
  
  constexpr PimDataType StencilTypePIM = getPIMTypeFromHostType<StencilTypeHost>();
  
  assert(!src_host.empty());
  assert(!src_host[0].empty());
  assert(src_host.size() == dst_host.size());
  assert(src_host[0].size() == dst_host[0].size());

  size_t height = src_host.size();
  size_t width = src_host[0].size();

  PimObjId resultPim = pimAlloc(PIM_ALLOC_AUTO, width, PimDataType);
  assert(resultPim != -1);

  // Stores the stencil rows that are currently being used, at most 3 rows for 3x3 stencil
  // TODO: Expand to 5x5 stencil
  constexpr uint64_t numRowsAtOnce = 3; // Must be odd
  std::vector<PIMStencilRow> pimGrid(numRowsAtOnce);
  for(size_t i=numRowsAtOnce/2; i<numRowsAtOnce; ++i) {
    pimGrid[i] = addVectorToGrid<StencilTypeHost, StencilTypePIM>(src_host[i], resultPim);
  }

  PimObjId runningSum = pimAllocAssociated(width, PimDataType);
  assert(runningSum != -1);

  for(size_t i=0; i<height; ++i) {
    status = pimAdd(reuseableSums[numRowsAtOnce/2], reuseableSums[numRowsAtOnce/2 + 1], resultPim);
    assert (status == PIM_OK);

    for(size_t j=numRowsAtOnce/2 + 2; j<numRowsAtOnce; ++j) {
      status = pimAdd(resultPim, reuseableSums[j], resultPim);
      assert (status == PIM_OK);
    }


  }

  pimFree(tmp_pim_obj);

  for(size_t i=pimGrid.size()-1; i>=(pimGrid.size()-6); --i) {
    pimFree(pimGrid[i]);
  }

  pimFree(resultPim);

  for(size_t i=0; i<reuseableSums.size(); ++i) {
    pimFree(reuseableSums[i]);
  }
}

uint8_t get_with_default(size_t i, size_t j, std::vector<std::vector<uint8_t>> &x) {
  if(i >= 0 && i < x.size() && j >= 0 && j < x[0].size()) {
    return x[i][j];
  }
  return 0;
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running PIM game of life for board: " << params.width << "x" << params.height << "\n";
  std::vector<std::vector<uint8_t>> x, y;
  if (params.inputFile == nullptr)
  {
    srand((unsigned)time(NULL));
    x.resize(params.height);
#pragma omp parallel for
    for(size_t i=0; i<params.height; ++i) {
      x[i].resize(params.width);
      for(size_t j=0; j<params.width; ++j) {
        x[i][j] = rand() & 1;
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

  stencil<uint8_t>(x, y);

  if (params.shouldVerify) 
  {
    bool is_correct = true;
#pragma omp parallel for
    for(uint64_t i=0; i<y.size(); ++i) {
      for(uint64_t j=0; j<y[0].size(); ++j) {
        uint8_t sum_cpu = get_with_default(i-1, j-1, x);
        sum_cpu += get_with_default(i-1, j, x);
        sum_cpu += get_with_default(i-1, j+1, x);
        sum_cpu += get_with_default(i, j-1, x);
        sum_cpu += get_with_default(i, j+1, x);
        sum_cpu += get_with_default(i+1, j-1, x);
        sum_cpu += get_with_default(i+1, j, x);
        sum_cpu += get_with_default(i+1, j+1, x);

        uint8_t res_cpu = (sum_cpu == 3) ? 1 : 0;
        sum_cpu = (sum_cpu == 2) ? 1 : 0;
        sum_cpu &= get_with_default(i, j, x);
        res_cpu |= sum_cpu;

        if (res_cpu != y[i][j])
        {
          std::cout << "Wrong answer: " << unsigned(y[i][j]) << " (expected " << unsigned(res_cpu) << ")" << std::endl;
          is_correct = false;
        }
      }
    }
    if(is_correct) {
      std::cout << "Correct for game of life!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
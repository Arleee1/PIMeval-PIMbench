// Test: C++ version of string match
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
#include "libpimeval.h"
#include "string-match-utils.h"
#include "new-pim-ops.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *keysInputFile;
  char *textInputFile;
  char *configFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -k    keys input file, with each key on a seperate line (required, searches in cpp-string-match/dataset directory, note that keys are expected to be sorted by length, with smaller keys first)"
          "\n    -t    text input file to search for keys from (required, searches in cpp-string-match/dataset directory)"
          "\n    -c    dramsim config file"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.keysInputFile = nullptr;
  p.textInputFile = nullptr;
  p.configFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:k:t:c:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'k':
      p.keysInputFile = optarg;
      break;
    case 't':
      p.textInputFile = optarg;
      break;
    case 'c':
      p.configFile = optarg;
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

void print_pim(PimObjId pim_obj, uint64_t len) {
  vector<uint8_t> dst_host;
  dst_host.resize(len, 1);

  PimStatus status = pimCopyDeviceToHost(pim_obj, (void *)dst_host.data());
  assert (status == PIM_OK);

  for (auto val : dst_host) {
    std::cout << unsigned(val) << " ";
  }
  std::cout << std::endl;
}

void print_pim_int(PimObjId pim_obj, uint64_t len) {
  vector<uint32_t> dst_host;
  dst_host.resize(len, 1);

  PimStatus status = pimCopyDeviceToHost(pim_obj, (void *)dst_host.data());
  assert (status == PIM_OK);

  for (auto val : dst_host) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

template <typename T>
void printVec(std::vector<T>& vec) {
  for(T elem : vec) {
    std::cout << elem << ", ";
  }
  std::cout << std::endl;
}

// Generates lookup table for every haystack character match
// Inverts table to slightly optimize later steps
std::vector<PimObjId> create_haystack_matches(const std::string& haystack) {
  PimStatus status;

  std::vector<PimObjId> haystack_matches;
  haystack_matches.reserve(4);
  haystack_matches.push_back(pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT64));
  for(size_t i=1; i<4; ++i) {
    haystack_matches.push_back(pimAllocAssociated(haystack_matches[0], PIM_UINT64));
    assert(haystack_matches.back() != -1);
  }

  PimObjId haystack_pim = pimAllocAssociated(haystack_matches[0], PIM_UINT8);
  assert(haystack_pim != -1);

  status = pimCopyHostToDevice((void *)haystack.c_str(), haystack_pim);
  assert (status == PIM_OK);

  PimObjId pim_scratch = pimAllocAssociated(haystack_matches[0], PIM_UINT8);
  assert(pim_scratch != -1);

  for(size_t i=0; i<256; ++i) {
    size_t matches_ind = i/64;
    size_t matches_offset = i%64;

    status = pimEQScalar1BitResult(pim_scratch, haystack_pim, haystack_matches[matches_ind], matches_offset, i);
    assert(status == PIM_OK);
  }

  pimFree(haystack_pim);
  pimFree(pim_scratch);

  for(size_t i=0; i<4; ++i) {
    status = pimXnorScalar(haystack_matches[i], haystack_matches[i], 0);
    assert(status == PIM_OK);
  }

  return haystack_matches;
}

// Creates new table to be refilled after shifting on later iterations
std::vector<PimObjId> create_haystack_matches_copy(const std::vector<PimObjId>& haystack_matches) {
  std::vector<PimObjId> haystack_matches_copy;
  haystack_matches_copy.reserve(4);
  for(size_t i=0; i<4; ++i) {
    haystack_matches_copy.push_back(pimAllocAssociated(haystack_matches[0], PIM_UINT64));
    assert(haystack_matches_copy.back() != -1);
  }

  return haystack_matches_copy;
}

// Populates table from backup
void copy_to_haystack_matches_copy(const std::vector<PimObjId>& haystack_matches, std::vector<PimObjId>& haystack_matches_copy) {
  PimStatus status;
  
  for(size_t i=0; i<4; ++i) {
    status = pimCopyDeviceToDevice(haystack_matches[i], haystack_matches_copy[i]);
    assert(status == PIM_OK);
  }
}

void string_match(std::vector<std::string>& needles, std::string& haystack, std::vector<int>& matches, uint64_t num_rows) {
  
  PimStatus status;
  // TODO update types when pim type conversion operation is available, currently everything uses PIM_UINT32, however this is unecessary

  // If vertical, each pim object takes 32 rows, 1 row if horizontal
  // Two rows used by the haystack and intermediate

  // 256 rows for haystack lookup table, 32 for 32 bit temp variable (only on vertical architectures), 32 bit final result variable (only on vertical architectures) 
  constexpr uint64_t necessary_rows_one_iter = 256+32+32;
  // Everything in necessary_rows_one_iter, plus extra 256 bits to store a backup of the lookup table, used to copy from for later iterations after shifting
  constexpr uint64_t necessary_rows_multiple_iter = necessary_rows_one_iter + 256;
  uint64_t max_needles_per_iteration_one_iteration = num_rows - necessary_rows_one_iter;
  uint64_t max_needles_per_iteration_multiple_iterations = num_rows - necessary_rows_multiple_iter;
  uint64_t max_needles_per_iteration;
  uint64_t num_iterations;
  if(needles.size() > max_needles_per_iteration_one_iteration) {
    max_needles_per_iteration = max_needles_per_iteration_multiple_iterations;
    num_iterations = (needles.size() + max_needles_per_iteration - 1) / max_needles_per_iteration;
  } else {
    max_needles_per_iteration = max_needles_per_iteration_one_iteration;
    num_iterations = 1;
  }

  std::vector<PimObjId> haystack_matches = create_haystack_matches(haystack);
  std::vector<PimObjId> haystack_matches_backup;
  if(num_iterations > 1) {
    haystack_matches_backup = create_haystack_matches_copy(haystack_matches);
    copy_to_haystack_matches_copy(haystack_matches, haystack_matches_backup);
  }

  uint64_t needles_done = 0;

  uint64_t num_needles = needles.size();
  
  // Temporary variable for intermediate calculations
  PimObjId intermediate_pim = pimAllocAssociated(haystack_matches[0], PIM_UINT32);
  assert(intermediate_pim != -1);

  PimObjId final_result_pim = pimAllocAssociated(haystack_matches[0], PIM_UINT32);
  assert(final_result_pim != -1);

  status = pimBroadcastUInt(final_result_pim, 0);
  assert (status == PIM_OK);

  // Array of PIM_UINT8 objects, should represent 8x 1bit pim objects
  size_t num_needle_matches_objs = max_needles_per_iteration/8;
  std::vector<PimObjId> pim_individual_needle_matches;
  pim_individual_needle_matches.reserve(num_needle_matches_objs);
  for(size_t i=0; i<num_needle_matches_objs; ++i) {
    pim_individual_needle_matches.push_back(pimAllocAssociated(haystack_matches[0], PIM_UINT8));
    assert(pim_individual_needle_matches.back() != -1);
  }

  for(uint64_t iter=0; iter<num_iterations; ++iter) {

    uint64_t needles_this_iteration = min(max_needles_per_iteration, num_needles - needles_done);

    // Algorithm Start
    uint64_t needles_finished_this_iter = 0;

    for(uint64_t char_idx=0; needles_finished_this_iter < needles_this_iteration; ++char_idx) {
      
      for(uint64_t needle_idx=0; needle_idx < needles_this_iteration; ++needle_idx) {
        
        uint64_t host_needle_idx = needle_idx + needles_done;
        uint64_t pim_needle_idx = needle_idx;

        if(char_idx >= needles[host_needle_idx].size()) {
          continue;
        }

        uint64_t curr_char = (uint64_t) needles[host_needle_idx][char_idx];
        // Would be replaced with indexing into 1bit pim array
        size_t haystack_matches_ind = curr_char/64;
        size_t haystack_matches_offset = curr_char%64;

        // Would be replaced with indexing into 1bit pim array
        size_t needle_matches_ind = pim_needle_idx/8;
        size_t needle_matches_offset = pim_needle_idx%8;

        if(char_idx == 0) {
          // status = pimEQScalar(haystack_pim, pim_individual_needle_matches[needle_idx_pim], (uint64_t) needles[current_needle_idx][char_idx]);
          status = pimCopyDeviceToDevice1Bit(haystack_matches[haystack_matches_ind], haystack_matches_offset, pim_individual_needle_matches[needle_matches_ind], needle_matches_offset);
          assert (status == PIM_OK);
        } else {
          // status = pimEQScalar(haystack_pim, intermediate_pim, (uint64_t) needles[current_needle_idx][char_idx]);
          // assert (status == PIM_OK);

          // status = pimAnd(pim_individual_needle_matches[needle_idx_pim], intermediate_pim, pim_individual_needle_matches[needle_idx_pim]);
          // assert (status == PIM_OK);
          status = pimOr1bit(pim_individual_needle_matches[needle_matches_ind], needle_matches_offset, haystack_matches[haystack_matches_ind], haystack_matches_offset, pim_individual_needle_matches[needle_matches_ind], needle_matches_offset);
          assert (status == PIM_OK);
        }

        if(char_idx + 1 == needles[host_needle_idx].size()) {
          ++needles_finished_this_iter;
        }
      }

      if(needles_finished_this_iter < needles_this_iteration) {
        for(size_t i=0; i<4; ++i) {
          // TODO: Shifting left sets right elements to 0/false
          // Currently, false represents a match, so keys that are cutoff are assumed to match
          // Fix by representing true as a match
          status = pimShiftElementsLeft(haystack_matches[i]);
          assert (status == PIM_OK);
        }
      }
    }

    // for(uint64_t needle_idx = 0; needle_idx < num_needles; ++needle_idx) {
    //   pimMulScalar(pim_individual_needle_matches[needle_idx], pim_individual_needle_matches[needle_idx], 1 + needle_idx);
    // }

    for(uint64_t needle_idx = 0; needle_idx < needles_this_iteration; ++needle_idx) {
      uint64_t host_needle_idx = needle_idx + needles_done;
      uint64_t pim_needle_idx = needle_idx;

      // Would be replaced with indexing into 1bit pim array
      size_t needle_matches_ind = pim_needle_idx/8;
      size_t needle_matches_offset = pim_needle_idx%8;

      status = pimCast1BitTo32Bit(pim_individual_needle_matches[needle_matches_ind], needle_matches_offset, intermediate_pim);
      assert (status == PIM_OK);

      status = pimSubScalar(intermediate_pim, intermediate_pim, 1);
      assert (status == PIM_OK);

      status = pimAndScalar(intermediate_pim, intermediate_pim, 1 + host_needle_idx);
      assert (status == PIM_OK);

      status = pimMax(final_result_pim, intermediate_pim, final_result_pim);
      assert (status == PIM_OK);

      // status = pimXorScalar(pim_individual_needle_matches[needle_idx_pim], pim_individual_needle_matches[needle_idx_pim], 1);
      // assert (status == PIM_OK);

      // status = pimSubScalar(pim_individual_needle_matches[needle_idx_pim], pim_individual_needle_matches[needle_idx_pim], 1);
      // assert (status == PIM_OK);

      // status = pimAndScalar(pim_individual_needle_matches[needle_idx_pim], pim_individual_needle_matches[needle_idx_pim], 1 + current_needle_idx);
      // assert (status == PIM_OK);
    }

    // for(uint64_t needle_idx = 1; needle_idx < needles_this_iteration + first_avail_pim_needle_result; ++needle_idx) {
    //   status = pimMax(pim_individual_needle_matches[0], pim_individual_needle_matches[needle_idx], pim_individual_needle_matches[0]);
    //   assert (status == PIM_OK);
    // }

    needles_done += needles_this_iteration;

    if(iter+1 != num_iterations) {
      copy_to_haystack_matches_copy(haystack_matches_backup, haystack_matches);
    }
  }

  status = pimCopyDeviceToHost(final_result_pim, (void *)matches.data());
  assert (status == PIM_OK);
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  
  if(params.keysInputFile == nullptr) {
    std::cout << "Please provide a keys input file" << std::endl;
    return 1;
  }
  if(params.textInputFile == nullptr) {
    std::cout << "Please provide a text input file" << std::endl;
    return 1;
  }
  
  std::cout << "Running PIM string match for \"" << params.keysInputFile << "\" as the keys file, and \"" << params.textInputFile << "\" as the text input file\n";
  
  std::string haystack;
  std::vector<std::string> needles;
  std::vector<int> matches;

  const std::string DATASET_FOLDER_PREFIX = "./../dataset/";

  haystack = get_text_from_file(DATASET_FOLDER_PREFIX, params.textInputFile);
  if(haystack.size() == 0) {
    std::cout << "There was an error opening the text file" << std::endl;
    return 1;
  }

  needles = get_needles_from_file(DATASET_FOLDER_PREFIX, params.keysInputFile);
  if(needles.size() == 0) {
    std::cout << "There was an error opening the keys file" << std::endl;
    return 1;
  }
  
  // if (!createDevice(params.configFile))
  // {
  //   return 1;
  // }
  PimStatus status;

  unsigned numRanks = 2;
  unsigned numBankPerRank = 128; // 8 chips * 16 banks
  unsigned numSubarrayPerBank = 32;
  unsigned numRows = 1024;
  unsigned numCols = 8192;

  status = pimCreateDevice(PIM_DEVICE_BITSIMD_V, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);

  PimDeviceProperties deviceProp;
  status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  matches.resize(haystack.size(), 0);
  
  string_match(needles, haystack, matches, deviceProp.numRowPerSubarray);

  // std::cout << "matches: ";
  // printVec(matches);

  if (params.shouldVerify) 
  {
    std::vector<int> matches_cpu;
    
    matches_cpu.resize(haystack.size());

    string_match_cpu(needles, haystack, matches_cpu);

    // verify result
    bool is_correct = true;
    #pragma omp parallel for
    for (unsigned i = 0; i < matches.size(); ++i)
    {
      if (matches[i] != matches_cpu[i])
      {
        std::cout << "Wrong answer: " << unsigned(matches[i]) << " (expected " << unsigned(matches_cpu[i]) << "), for position " << i << std::endl;
        is_correct = false;
      }
    }
    if(is_correct) {
      std::cout << "Correct for string match!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}

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

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t stringLength;
  uint64_t keyLength;
  uint64_t numKeys;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -s    string size (default=2048 elements)"
          "\n    -k    key size (default = 20 elements)"
          "\n    -n    number of keys (default = 4 keys)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing string and keys (default=generates strings with random characters)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.stringLength = 2048;
  p.keyLength = 20;
  p.numKeys = 4;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:s:k:n:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 's':
      p.stringLength = strtoull(optarg, NULL, 0);
      break;
    case 'k':
      p.keyLength = strtoull(optarg, NULL, 0);
      break;
    case 'n':
      p.numKeys = strtoull(optarg, NULL, 0);
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
  vector<int> dst_host;
  dst_host.resize(len, 1);

  PimStatus status = pimCopyDeviceToHost(pim_obj, (void *)dst_host.data());
  assert (status == PIM_OK);

  for (auto val : dst_host) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

void string_match(vector<string>& needles, string& haystack, vector<vector<uint8_t>>& matches) {

  uint64_t num_needles = needles.size();

  PimObjId haystack_pim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT8);
  assert(haystack_pim != -1);

  // intermediate_pim, and pim_results[i] can be replaced with booleans/1 bit ints
  PimObjId intermediate_pim = pimAllocAssociated(haystack_pim, PIM_UINT8);
  assert(intermediate_pim != -1);

  vector<PimObjId> pim_results;
  for(uint64_t i=0; i<num_needles; ++i) {
    pim_results.push_back(pimAllocAssociated(haystack_pim, PIM_UINT8));
    assert(pim_results.back() != -1);
  }

  PimStatus status = pimCopyHostToDevice((void *)haystack.c_str(), haystack_pim);
  assert (status == PIM_OK);

  // Algorithm Start
  uint64_t needles_finished = 0;

  for(uint64_t char_idx=0; needles_finished < num_needles; ++char_idx) {
    for(uint64_t needle_idx=0; needle_idx < num_needles; ++needle_idx) {

      if(char_idx >= needles[needle_idx].size()) {
        continue;
      }

      if(char_idx == 0) {
        status = pimEQScalar(haystack_pim, pim_results[needle_idx], (uint64_t) needles[needle_idx][char_idx]);
        assert (status == PIM_OK);
      } else {
        status = pimEQScalar(haystack_pim, intermediate_pim, (uint64_t) needles[needle_idx][char_idx]);
        assert (status == PIM_OK);

        status = pimAnd(pim_results[needle_idx], intermediate_pim, pim_results[needle_idx]);
        assert (status == PIM_OK);
      }

      if(char_idx + 1 == needles[needle_idx].size()) {
        ++needles_finished;
      }
    }

    if(needles_finished < num_needles) {
      pimShiftElementsLeft(haystack_pim);
    }
  }

  for(uint64_t needle_idx = 0; needle_idx < num_needles; ++needle_idx) {
    status = pimCopyDeviceToHost(pim_results[needle_idx], (void *)matches[needle_idx].data());
    assert (status == PIM_OK);
  }
}

void string_match_cpu(vector<string>& needles, string& haystack, vector<vector<uint8_t>>& matches) {

  for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
    size_t pos = haystack.find(needles[needle_idx], 0);

    while (pos != string::npos) {
        matches[needle_idx][pos] = 1;
        pos = haystack.find(needles[needle_idx], pos + 1);
    }
  }
}

void getString(string& str, uint64_t len) {
  str.resize(len);
#pragma omp parallel for
  for(uint64_t i=0; i<len; ++i) {
    str[i] = 'a' + (rand()%26);
  }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running PIM string match for string size: " << params.stringLength << ", key size: " << params.keyLength << ", number of keys: " << params.numKeys << "\n";
  string haystack;
  vector<string> needles;
  vector<vector<uint8_t>> matches;
  if (params.inputFile == nullptr)
  {
    // getString(haystack, params.stringLength);
    // for(uint64_t i=0; i < params.numKeys; ++i) {
    //   needles.push_back("");
    //   getString(needles.back(), params.keyLength);
    // }
    
    haystack = "dabcd";
    needles = {"abc", "d"};
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

  matches.resize(needles.size());

  for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
    matches[needle_idx].resize(haystack.size());
  }

  //TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  string_match(needles, haystack, matches);

  if (params.shouldVerify) 
  {
    vector<vector<uint8_t>> matches_cpu;
    
    matches_cpu.resize(needles.size());

    for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
      matches_cpu[needle_idx].resize(haystack.size());
    }

    string_match_cpu(needles, haystack, matches_cpu);

    // verify result
    bool is_correct = true;
    #pragma omp parallel for
    for (unsigned i = 0; i < matches.size(); ++i)
    {
      for (unsigned j = 0; j < haystack.size(); ++j)
      {
        if (matches[i][j] != matches_cpu[i][j])
        {
          std::cout << "Wrong answer: " << unsigned(matches[i][j]) << " (expected " << unsigned(matches_cpu[i][j]) << "), for needle: " << i << " at position " << j << std::endl;
          is_correct = false;
        }
      }
    }
    if(is_correct) {
      std::cout << "Correct for string match!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
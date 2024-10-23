/* File:     string-match.cu
 * Purpose:  Implement string matching on a GPU
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <vector>

#include "utilBaselines.h"

constexpr uint8_t CHAR_OFFSET = 5;

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  int64_t stringLength;
  uint64_t keyLength;
  uint64_t numKeys;
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
          "\n    -i    input file containing string and keys (default=generates strings with random characters)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.stringLength = 2048;
  p.keyLength = 20;
  p.numKeys = 4;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:s:k:n:i:v:")) >= 0)
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

/**
 * @brief gpu string match kernel
 * @param haystack full haystack, with words seperated by and ending in '\n'
 * @param haystack_len length of the haystack, including all '\n' delimiters
 * @param needles array of all of the needles adjacent to each other, separated and terminated by '\0'
 * @param num_needles number of needles
 * @param needle_indexes prefix array of needle lengths
 * @param matches result, matches[i] > 0 iff there is a match for the ith needle
 */

// x -> haystack index
// y -> needles index
// Must set matches array to 0s ahead of time
__global__ void string_match(char* haystack, size_t haystack_len, char* needles, uint64_t num_needles, uint64_t* needle_indexes, unsigned long long int* matches, unsigned long long int needles_len) {

  size_t haystack_idx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t needle_idx = blockIdx.y*blockDim.y + threadIdx.y;
  size_t old_needle_idx = needle_idx;
  
  if(haystack_idx == 0 && needle_idx == 0) {
    printf("gpu haystack:");
    for(uint64_t i=0; i<haystack_len; ++i) {
      printf("%c", haystack[i]);
    }
    printf("\n");
    printf("gpu needles:");
    for(uint64_t i=0; i<needles_len; ++i) {
      printf("%c", needles[i]);
    }
    printf("\n");
    printf("gpu needle indexes:");
    for(uint64_t i=0; i<num_needles; ++i) {
      printf("%d ", needle_indexes[i]);
    }
    printf("\n");
  }
  bool should_match = (needle_idx == 1) && (haystack_idx == 0);
  if(should_match) {
    printf("should match reached!\n");
  }
  if (haystack_idx < haystack_len && needle_idx < num_needles && haystack[haystack_idx] == '\n') {
    needle_idx = needle_indexes[needle_idx];
    if(should_match) {
      printf("needle char index: %d\n", needle_idx);
    }
    ++haystack_idx;
    if(should_match) {
      printf("needle char index: %d\n", needle_idx);
      printf("before loop - haystack idx: %d, needle_idx: %d\n", (int)haystack_idx, (int)needle_idx);
    }
    int i;
    for (i = 0; needles[needle_idx + i] != '\n' && i+haystack_idx < haystack_len; ++i) {
      if (haystack[haystack_idx + i] != needles[needle_idx + i]) {
          break;
      }
    }

    if(should_match) {
      printf("after loop - needle_idx: %d, i: %d, haystack_idx: %d\n", (int)needle_idx, (int)i, (int)haystack_idx);
      printf("after loop - needle char: %d, haystack char: %d\n", (int)needles[needle_idx + i], (int)haystack[i+haystack_idx]);
      printf("after loop - newline num: %d\n", (int)'\n');
    }

    if(needles[needle_idx + i] == '\n' && i+haystack_idx < haystack_len && (haystack[i+haystack_idx] == '\n' || haystack[i+haystack_idx] == '\r')) {
      printf("adding to matches for key: %d\n", (int) old_needle_idx);
      atomicAdd(matches + old_needle_idx, (unsigned long long int)1);
    }
  }
}

void string_match_cpu(string& needle, string& haystack, vector<uint8_t>& matches) {
  size_t pos = haystack.find(needle, 0);

  if (pos == string::npos) {
    return;
  }

  while (pos != string::npos) {
      matches[pos] = 1;
      pos = haystack.find(needle, pos + 1);
  }
}

void getString(string& str, uint64_t len) {
  str.resize(len);
#pragma omp parallel for
  for(uint64_t i=0; i<len; ++i) {
    str[i] = 'a' + (rand()%26);
  }
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params params = input_params(argc, argv);
  std::cout << "Running GPU string match for string size: " << params.stringLength << ", key size: " << params.keyLength << ", number of keys: " << params.numKeys << "\n";
  string haystack;
  vector<string> needles;
  vector<unsigned long long int> matches;

  if (params.inputFile == nullptr)
  {
    // getString(haystack, params.stringLength);
    // for(uint64_t i=0; i < params.numKeys; ++i) {
    //   needles.push_back("");
    //   getString(needles.back(), params.keyLength);
    // }

    haystack = "abc\ndef\nghi\neldkslkdfj\nhelloworld";
    needles = {"helloworld", "abc", "lmp", "helloworld", "eld", "slk", "eldkslkdfj"};
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  // Ensure that haystack starts and ends in '\n'
  haystack = '\n' + haystack + '\n';

  // Setup length variables
  uint64_t haystack_len = haystack.size();
  uint64_t num_needles = needles.size();

  // Setup needle indexes for gpu to use
  vector<uint64_t> needle_indexes;
  needle_indexes.reserve(num_needles);
  uint64_t curr_index = 0;
  for(string& needle : needles) {
    needle_indexes.push_back(curr_index);
    // Add 1 for the delimiter between needles
    curr_index += needle.size() + 1;
  }

  //Resize result
  matches.resize(num_needles);

  // Setup concataned needles
  string needles_concat;
  needles_concat.reserve(curr_index);
  for(string& needle : needles) {
    needles_concat += needle + '\n';
  }

  // char* haystack, size_t haystack_len, char* needles, uint64_t num_needles, uint64_t* needle_indexes

  char* gpu_haystack;
  char* gpu_needles;
  uint64_t* gpu_needle_indexes;
  unsigned long long int* gpu_matches;

  size_t haystack_sz = sizeof(char)*haystack_len;
  size_t needle_sz = sizeof(char)*curr_index;
  size_t needle_indexes_sz = sizeof(uint64_t)*num_needles;
  size_t matches_sz = sizeof(unsigned long long int)*num_needles;

  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&gpu_haystack, haystack_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMemcpy(gpu_haystack, haystack.c_str(), haystack_sz, cudaMemcpyHostToDevice);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMalloc((void**)&gpu_needles, needle_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMemcpy(gpu_needles, needles_concat.c_str(), needle_sz, cudaMemcpyHostToDevice);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  // Needle indexes
  cuda_error = cudaMalloc((void**)&gpu_needle_indexes, needle_indexes_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMemcpy(gpu_needle_indexes, needle_indexes.data(), needle_indexes_sz, cudaMemcpyHostToDevice);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMalloc((void**)&gpu_matches, matches_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMemset(gpu_matches, 0, matches_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  dim3 dimBlock(16, 16);
  dim3 dimGrid((haystack_len + dimBlock.x - 1) / dimBlock.x, (num_needles + dimBlock.y - 1) / dimBlock.y);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  cudaEventRecord(start, 0);

  string_match<<<dimGrid, dimBlock>>>(gpu_haystack, haystack_len, gpu_needles, num_needles, gpu_needle_indexes, gpu_matches, curr_index);
  
  cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
  {
      std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
      exit(1);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  printf("Execution time of string match = %f ms\n", timeElapsed);

  cuda_error = cudaMemcpy(matches.data(), gpu_matches, matches_sz, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
      exit(1);
  }
  cudaFree(gpu_haystack);
  cudaFree(gpu_needles);
  cudaFree(gpu_needle_indexes);
  cudaFree(gpu_matches);

  for(uint64_t i=0; i<matches.size(); ++i) {
    cout << i;
    if(matches[i]) {
      cout << " is a match\n";
    } else {
      cout << " is not a match\n";
    }
  }

  // if (params.shouldVerify) 
  // {
  //   vector<uint8_t> matches_cpu;
  //   matches_cpu.resize(haystack.size());
  //   string_match_cpu(needle, haystack, matches_cpu);

  //   // verify result
  //   bool is_correct = true;
  //   #pragma omp parallel for
  //   for (unsigned i = 0; i < matches.size(); ++i)
  //   {
  //     if (matches[i] != matches_cpu[i])
  //     {
  //       std::cout << "Wrong answer: " << unsigned(matches[i]) << " (expected " << unsigned(matches_cpu[i]) << "), at index: " << i << std::endl;
  //       is_correct = false;
  //     }
  //   }
  //   if(is_correct) {
  //     std::cout << "Correct for string match!" << std::endl;
  //   }
  // }

  return 0;
}

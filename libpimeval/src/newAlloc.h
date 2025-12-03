#include <cstdint>
#include <vector>
#include <cstdlib>
#include "libpimeval.h"

// Question: what should happen if you specify a granularity finer than the architecture supports?
// e.g., specify subarray when architecture is bank-level PIM, or how to specify subarray when BitSIMD aggregates two subarrays per core
// For now, we just ignore the finer granularity
// TODO: define behavior more clearly
struct PimAllocLocation {
  int64_t rank = -1;
  int64_t bank = -1;
  int64_t subarray = -1;
};

typedef int PimObjBlock; // Could change definition later
  
// In the original alloc API, there is no way to specify subarrays or banks, only number of elements
// Would probably need to specify a range of subarrays etc.
// Could modify existing API, which would be tricky for backward compatibility, or could add a new API
// I propose adding a new API
// Old:
// PimObjId pimAlloc(PimAllocEnum allocType, uint64_t numElements, PimDataType dataType);

// New, with location specification
// Allocates all of a range of subarrays/banks/ranks
PimObjBlock pimAllocBlock(PimAllocEnum allocType, PimDataType dataType, PimAllocLocation& startLocation, PimAllocLocation& endLocation);

// For PIM copy, specify mapping of host to PIM locations
// @param src: flattened 2D array
// @param destBlock. The PimObjBlock allocated by pimAllocBlock
// @param srcWidth: width of the source array in host memory
// @param srcHeight: height of the source array in host memory
// @param tileWidth: width of each tile to be copied to each subarray/bank (assumed uniform, can be cutoff at edges)
// @param tileHeight: height of each tile to be copied to each subarray/bank (assumed uniform, can be cutoff at edges)
// @param blockMapping: 2D vector specifying the PimAllocLocation for each subarray/bank
//                      blockMapping[i][j] gives the PimAllocLocation for the (i,j)th tile in the block
// @return: 3D vector of PimObjId
//          The first two dimensions correspond to the location of the tiles
//          The third dimension corresponds to multiple PimObjIds, as the tiles are 2D memory locations, however each PimObjId is a 1D array
//          Necessary to split up large block into multiple PimObjIds to operate on

std::vector<std::vector<std::vector<PimObjId>>> pimCopyHostToBlock(void* src, PimObjBlock destBlock, uint64_t srcWidth, uint64_t srcHeight,
    uint64_t tileWidth, uint64_t tileHeight, std::vector<std::vector<PimAllocLocation>>& blockMapping);


void exampleUsage() {
    // Example usage
    PimAllocLocation startLoc;
    startLoc.rank = 0;
    startLoc.bank = 0;
    startLoc.subarray = 0;
    
    PimAllocLocation endLoc;
    endLoc.rank = 0;
    endLoc.bank = 3; // Allocate banks 0 to 3
    endLoc.subarray = -1; // All subarrays
    
    PimObjBlock block = pimAllocBlock(PIM_ALLOC_AUTO, PIM_INT32, startLoc, endLoc); // Reserves banks 0-3 in rank 0
    
    const uint64_t srcWidth = 1024;
    const uint64_t srcHeight = 1024;
    const uint64_t tileWidth = 256;
    const uint64_t tileHeight = 256;
    
    // Define block mapping
    std::vector<std::vector<PimAllocLocation>> blockMapping(4, std::vector<PimAllocLocation>(4));
    for (uint64_t i = 0; i < 4; ++i) {
        for (uint64_t j = 0; j < 4; ++j) {
        PimAllocLocation loc;
        loc.rank = 0;
        loc.bank = i; // Map to bank i
        loc.subarray = j; // Map to subarray j
        blockMapping[i][j] = loc;
        }
    }


    int* src = (int*) std::malloc(srcWidth * srcHeight * sizeof(int)); // Example source data
    
    std::vector<std::vector<std::vector<PimObjId>>> pimObjs = pimCopyHostToBlock(src, block, srcWidth, srcHeight, tileWidth, tileHeight, blockMapping);
    
    // Now pimObjs contains the PimObjIds for each tile in the block
    // Each pimObjs[i][j] is a vector of PimObjIds corresponding to the (i,j)th tile
    std::vector<PimObjId> firstTileObjs = pimObjs[0][0];
    // Each tile is represented by multiple PimObjIds, as each PimObjId is a 1D array, so require multiple to represent 2D tile
    std::free(src);
}
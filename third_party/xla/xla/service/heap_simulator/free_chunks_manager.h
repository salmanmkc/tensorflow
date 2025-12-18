/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_HEAP_SIMULATOR_FREE_CHUNKS_MANAGER_H_
#define XLA_SERVICE_HEAP_SIMULATOR_FREE_CHUNKS_MANAGER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"

namespace xla {

// Represents a chunk of memory, which can be in different states: allocated,
// free, or a candidate for allocation.
class MemoryChunk {
 public:
  MemoryChunk(int64_t offset, int64_t end, int64_t aligned_chunk_offset,
              int64_t id)
      : offset_(offset),
        end_(end),
        aligned_chunk_offset_(aligned_chunk_offset),
        id_(id) {}
  MemoryChunk(int64_t offset, int64_t end) : offset_(offset), end_(end) {}

  // Returns the usable size of this chunk for aligned allocations.
  int64_t size() const { return end_ - aligned_chunk_offset_; }

  bool operator<(const MemoryChunk& other) const {
    return offset_ < other.offset_;
  }

  bool operator==(const MemoryChunk& other) const {
    return offset_ == other.offset_ && end_ == other.end_;
  }

  int64_t offset() const { return offset_; }
  int64_t end() const { return end_; }
  int64_t aligned_chunk_offset() const { return aligned_chunk_offset_; }
  int64_t id() const { return id_; }

 private:
  int64_t offset_ = -1;  // Inclusive offset of this memory chunk.
  int64_t end_ = -1;     // Exclusive end of this memory chunk.
  // The smallest address >= offset_ that satisfies alignment requirements.
  // If offset_ is aligned, aligned_chunk_offset_ == offset_, otherwise
  // aligned_chunk_offset_ > offset_.
  // When an allocation is placed in a free chunk, it must be placed at an
  // aligned offset. The smallest aligned offset in [offset_, end_) is
  // aligned_chunk_offset_, therefore a free chunk [offset_, end_) can contain
  // an allocation of size `S` only if `S <= end_ - aligned_chunk_offset_`.
  int64_t aligned_chunk_offset_ = -1;
  // A unique ID assigned to free chunks, used internally by FreeChunksManager
  // to track removed chunks for lazy removal.
  int64_t id_ = -1;
};

bool FreeChunkOffsetLessThan(const MemoryChunk& a, const MemoryChunk& b);
bool FreeChunkSizeLessThan(const MemoryChunk& a, const MemoryChunk& b);

using FreeChunksByOffset =
    absl::btree_set<MemoryChunk, decltype(&FreeChunkOffsetLessThan)>;
using FreeChunksBySize =
    absl::btree_set<MemoryChunk, decltype(&FreeChunkSizeLessThan)>;

// Manages the free space created by the given free chunks.
// Maintains the chunks:
// - sorted by offset: to allow for efficient insertion of a chunk.
//   This may split a free chunk into two chunks (see Allocate()).
// - sorted by size: to allow for efficient querying of free chunks of size
//   at least the given size.
class FreeChunksManager {
 public:
  explicit FreeChunksManager(
      absl::AnyInvocable<int64_t(int64_t)> chunk_alignment);

  // Allocates the given interval [offset, end), removing it from free chunks.
  // If [offset, end) is carved out of a larger free chunk, the remainder
  // portion(s) remain free.
  // It's an error to allocate an interval that is not contained in a single
  // free chunk.
  void Allocate(int64_t offset, int64_t end);

  // Deallocates the given interval [offset, end), adding it to free chunks.
  // The manager maintains disjointness of free chunks: if [offset, end)
  // overlaps with or is adjacent to any existing free chunks, they are merged
  // to form a single larger free chunk. If all or parts of the given interval
  // are already free, they are merged with adjacent free chunks.
  void Deallocate(int64_t offset, int64_t end);

  // Returns a free chunk that is large enough to fit the given size, or
  // std::nullopt if no such chunk exists.
  std::optional<MemoryChunk> FindJustLargeEnough(int64_t size);

  // (Slow -- for testing/debugging) returns all the free chunks in a vector,
  // sorted by chunk offset.
  std::vector<MemoryChunk> GetFreeChunks();

 private:
  // Creates and adds a new free chunk representing interval [offset, end) to
  // free_chunks_by_offset_ and free_chunks_by_size_.
  void AddNewFreeChunk(int64_t offset, int64_t end);

  // Removes free chunk pointed by 'it' from free_chunks_by_offset_, and marks
  // it as removed in to_be_removed_ for lazy removal from
  // free_chunks_by_size_.
  void RemoveFreeChunk(FreeChunksByOffset::iterator it);

  // Removes any existing free chunks that overlap with or are adjacent to
  // [offset, end), merges them into a single chunk, and adds that chunk.
  void InvalidateAndMerge(int64_t offset, int64_t end);

  absl::AnyInvocable<int64_t(int64_t)> chunk_alignment_;

  // Free chunks sorted by offset. The endpoint is used as a tie breaker
  // and is only used when querying (using lower/upper bound).
  FreeChunksByOffset free_chunks_by_offset_;

  // Free chunks sorted by size. This set may contain free chunks that have
  // been removed, so they must be discarded and avoid returning them.
  FreeChunksBySize free_chunks_by_size_;

  // Keeps track of the IDs of the free chunks that have been removed
  // (see also free_chunks_by_size_).
  // This is an optimization to avoid expensive removals from
  // free_chunks_by_size_ when a chunk is removed. Removing a chunk from
  // free_chunks_by_offset_ gives us an iterator, but removing from
  // free_chunks_by_size_ requires constructing a MemoryChunk and doing a lookup
  // by value O(logN). Instead of paying that cost on every removal, we mark
  // chunks as removed here in O(1) and lazily remove them from
  // free_chunks_by_size_ in FindJustLargeEnough when we iterate past them.
  absl::flat_hash_set<int64_t> to_be_removed_;

  // The ID of the next inserted free chunk.
  int64_t next_free_chunk_id_ = 1;
};

}  // namespace xla

#endif  // XLA_SERVICE_HEAP_SIMULATOR_FREE_CHUNKS_MANAGER_H_

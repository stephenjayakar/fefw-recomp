// SPDX-License-Identifier: GPL-3.0-or-later
// Per-thread call counters for the guest's hot paths (HLE thunks, SVCs). A shared std::atomic counter bumped by every guest thread on every
// call is a cache line that moves between cores on each call (the game thread spent 8.9 % of its busy samples on such a fetch_add). Here each
// thread counts into its own block (only that thread writes it: a relaxed load and store, no locked instruction, no line transfer); a read
// sums every block ever made, so counts of threads that have exited stay in. Blocks are never freed (N counters x 8 bytes per thread).
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace nxrt {

// Tag: a type unique to each counter set (the thread's block pointer is a thread_local of the instantiation).
template <size_t N, class Tag>
class TlCounts {
public:
    static constexpr size_t kSize = N;
    // i < N. The calling thread's block is made on its first count.
    void add(size_t i) {
        Block* b = mine();
        b->n[i].store(b->n[i].load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }
    uint64_t sum(size_t i) {
        std::lock_guard<std::mutex> lk(mu_);
        uint64_t t = 0;
        for (Block* b : blocks_) t += b->n[i].load(std::memory_order_relaxed);
        return t;
    }
    uint64_t total() {
        std::lock_guard<std::mutex> lk(mu_);
        uint64_t t = 0;
        for (Block* b : blocks_)
            for (auto& v : b->n) t += v.load(std::memory_order_relaxed);
        return t;
    }
    // Zeroes every block (tests, counter resets; a concurrent add may survive).
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        for (Block* b : blocks_)
            for (auto& v : b->n) v.store(0, std::memory_order_relaxed);
    }

private:
    struct Block { std::atomic<uint64_t> n[N] = {}; };
    Block* mine() {
        thread_local Block* t_block = nullptr;     // one per thread per instantiation: each TlCounts<N, Tag> must be a single static
        if (!t_block) {
            t_block = new Block();
            std::lock_guard<std::mutex> lk(mu_);
            blocks_.push_back(t_block);
        }
        return t_block;
    }
    std::mutex mu_;
    std::vector<Block*> blocks_;
};

}  // namespace nxrt

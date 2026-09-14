// Thin native adapter for the pinned FlashAttention-3 forward implementation.
// No ATen/Python runtime or K/V payload conversion is required.
#include "wrapper.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cuda_check.h"

// Upstream's standalone checks exit the process. Translate native failures
// into our C ABI error result instead, without modifying the provider source.
#undef CHECK_CUDA
#undef CHECK_CUDA_KERNEL_LAUNCH
#undef CHECK_CUTLASS
#define CHECK_CUDA(call) do { \
    cudaError_t result = (call); \
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); \
} while (0)
#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())
#define CHECK_CUTLASS(call) do { \
    cutlass::Status result = (call); \
    if (result != cutlass::Status::kSuccess) \
        throw std::runtime_error(cutlass::cutlassGetStatusString(result)); \
} while (0)

#include "flash_fwd_launch_template.h"
#include "flash_prepare_scheduler.cu"

#if !defined(ORBITKV_HEAD_DIM) || !defined(ORBITKV_BF16) || !defined(ORBITKV_LOCAL)
#error "FlashAttention dtype, head dimension and visibility must be instantiated explicitly"
#endif

namespace orbitkv_flashattention {
thread_local std::string last_error;
constexpr int HEAD_DIM = ORBITKV_HEAD_DIM;
constexpr bool LOCAL = ORBITKV_LOCAL != 0;
using Element = std::conditional_t<ORBITKV_BF16 != 0, cutlass::bfloat16_t, cutlass::half_t>;

// Four warps copy each request's compact page row. This is metadata conversion,
// not an attention implementation; K/V remain in their original allocations.
constexpr unsigned METADATA_THREADS = 4 * 32;
__global__ void convert_page_metadata(
    const int32_t* indices, const int32_t* indptr, const int32_t* last,
    int32_t* table, int32_t* lengths, int width, int page_size, int cache_pages) {
    int request = blockIdx.x;
    int begin = indptr[request];
    int end = indptr[request + 1];
    assert(begin >= 0 && end > begin && end <= width);
    assert(last[request] > 0 && last[request] <= page_size);
    if (threadIdx.x == 0) lengths[request] = (end - begin - 1) * page_size + last[request];
    for (int slot = threadIdx.x; slot < width; slot += blockDim.x) {
        int page = slot < end - begin ? indices[begin + slot] : 0;
        assert(page >= 0 && page < cache_pages);
        table[static_cast<int64_t>(request) * width + slot] = page;
    }
}

void launch(const OrbitKVFlashAttentionLaunch& args, cudaStream_t stream) {
    convert_page_metadata<<<args.requests, METADATA_THREADS, 0, stream>>>(
        args.page_indices, args.page_indptr, args.last_page_len,
        args.page_table, args.kv_lengths, args.context_pages, args.page_size, args.cache_pages);
    CHECK_CUDA_KERNEL_LAUNCH();

    Flash_fwd_params params{};
    params.q_ptr = const_cast<void*>(args.query);
    params.k_ptr = const_cast<void*>(args.key);
    params.v_ptr = const_cast<void*>(args.value);
    params.o_ptr = args.output;
    params.q_row_stride = static_cast<int64_t>(args.query_heads) * HEAD_DIM;
    params.q_head_stride = HEAD_DIM;
    params.k_row_stride = params.v_row_stride = static_cast<int64_t>(args.kv_heads) * HEAD_DIM;
    params.k_head_stride = params.v_head_stride = HEAD_DIM;
    params.k_batch_stride = params.v_batch_stride = static_cast<int64_t>(args.page_size) * args.kv_heads * HEAD_DIM;
    params.v_dim_stride = 1;
    // The logical result is [heads, query_tokens, dimension]. FA3 accepts these
    // output strides directly, avoiding an extra output transpose.
    params.o_row_stride = HEAD_DIM;
    params.o_head_stride = static_cast<int64_t>(args.query_tokens) * HEAD_DIM;
    params.cu_seqlens_q = const_cast<int32_t*>(args.query_indptr);
    params.seqused_k = args.kv_lengths;
    params.softmax_lse_ptr = args.lse;
    params.b = args.requests;
    params.h = args.query_heads;
    params.h_k = args.kv_heads;
    params.total_q = params.seqlen_q = args.query_tokens;
    params.seqlen_k = args.context_pages * args.page_size;
    params.d = params.dv = params.d_rounded = params.dv_rounded = HEAD_DIM;
    params.scale_softmax = args.scale;
    params.p_dropout = params.rp_dropout = 1.f;
    params.p_dropout_in_uint8_t = 255;
    params.is_bf16 = ORBITKV_BF16 != 0;
    params.is_causal = !LOCAL;
    params.is_local = LOCAL;
    params.window_size_left = LOCAL ? args.window_left : params.seqlen_k - 1;
    params.window_size_right = 0;
    params.page_table = args.page_table;
    params.page_table_batch_stride = args.context_pages;
    params.page_size = args.page_size;
    params.num_pages = args.cache_pages;
    params.pagedkv_tma = false;
    params.arch = 90;
    params.num_sm = args.num_sm;
    params.num_splits = 1;
    params.pack_gqa = true;
    params.num_splits_dynamic_ptr = args.split_counts;
    params.num_m_blocks_ptr = args.query_tiles;
    params.varlen_sort_batches = !LOCAL;
    params.varlen_batch_idx_ptr = LOCAL ? nullptr : args.batch_order;
    params.head_swizzle = true;
    params.num_nheads_in_l2_ptr = args.head_swizzle;
    params.tile_count_semaphore = args.tile_counter;
    params.prepare_varlen_pdl = false;

    // Upstream SM90, paged non-TMA, packed GQA, non-split forward algorithm.
    // Its own tile policy and scheduler choose the actual thread/GMMA layout.
    run_flash_fwd<90, HEAD_DIM, HEAD_DIM, 1, Element, Element,
        !LOCAL, LOCAL, false, true, true, false, false, true, false, false>(params, stream);
}
}  // namespace orbitkv_flashattention

extern "C" int orbitkv_flashattention_run(const OrbitKVFlashAttentionLaunch* args, void* stream) noexcept {
    try {
        orbitkv_flashattention::last_error.clear();
        if (!args) throw std::invalid_argument("missing FlashAttention launch arguments");
        orbitkv_flashattention::launch(*args, static_cast<cudaStream_t>(stream));
        return 0;
    } catch (const std::exception& error) {
        orbitkv_flashattention::last_error = error.what();
    } catch (...) {
        orbitkv_flashattention::last_error = "unknown native FlashAttention failure";
    }
    return -1;
}

extern "C" const char* orbitkv_flashattention_last_error() noexcept {
    return orbitkv_flashattention::last_error.c_str();
}

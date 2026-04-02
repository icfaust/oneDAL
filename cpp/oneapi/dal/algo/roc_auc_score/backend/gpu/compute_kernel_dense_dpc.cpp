/*******************************************************************************
* Copyright contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "oneapi/dal/algo/roc_auc_score/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/backend/primitives/atomic.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/backend/primitives/placement.hpp"
#include "oneapi/dal/backend/primitives/sort.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/detail/profiler.hpp"


namespace oneapi::dal::roc_auc_score::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

tepmlate <typename Float>
double compute_roc_auc_score(sycl::queue& queue,
                        pr::ndarray<Float, 1>& y0, // mutability is required for sorting
                        pr::ndarray<Float, 1>& y1,
                        const dal::backend::event_vector& deps = {}) {

                    double result = 0;
                    // Steps:
                    // 1. sort the y1 data using key value pairs (sort primitive)
                    // 2. diff the sort data based on y1 using a gather approach (neq comparison to next element array)
                    // 3. add up y0 instances per y1 rank (custom segmented sum reduction kernel) true and false positive scores via cumsum (placement primitive)
                    // 4. integrate results for roc_auc_score (custom sum reduction kernel)
                    
                    // STEP 1
                    auto row_count = y1.get_dimension(0);

                    // sort y1 data using key value pairs (sort primitive) to get y0 and y1 in y1 ascending order
                    // this complicates/inverts some of the logic in calculating roc_auc_score but improves perf
                    // do not use dpl version due to hardware limitations
                    pr::radix_sort_indices_inplace<Float, Float>(queue, y1, y0, deps).wait_and_throw();

                    // STEP 2
                    // done as integer to prevent rank incrementing problems at higher float values     
                    auto diff = pr::ndarray<std::uint32_t, 1>::ones(queue, {row_count}, sycl::usm::alloc::device);

                    auto base = y0.get_slice(0, row_count - 1);
                    auto offset = y0.get_slice(1, row_count);
                    auto diff_offset = diff.get_slice(1, row_count - 1);

                    const auto kernel_neq = [=](const Float a, const Float b) -> std::uint32_t {
                        return reinterpret_cast<std::uint32_t>(a != b);
                    };

                    element_wise(queue, kernel_neq, base, offset, diff_offset).wait_and_throw();

                    // does this need to be done as integers to prevent large number spacing issues (probably)
                    cumulative_sum_1d(queue, diff).wait_and_throw();

                    // STEP 3
                    // create necessary arrays for doing the integration (tps and fps)
                    // requires first allocating 2 arrays size of the number of ranks and then
                    // a custom kernel for finding the total number of elements in that rank
                    const std::uint32_t* diff_ptr = diff.get_data();
                    auto total_ranks = diff_ptr[row_count - 1];
                    auto ps = pr::ndarray<Float, 2>::empty(queue, { total_ranks, 2 }, sycl::usm::alloc::device);

                    // custom kernel to add by group using atomic add, instead of doing it twice, create an interleaved
                    // tps, fps 2d data array to leverage hardware efficiencies (vector loading) and to reduce atomics
                    // pressure.
                    // first extract necessary pointers from the data
                    const Float* ps_ptr = ps.get_data();
                    const Float* y0_ptr = y0.get_data();
                    
                    // look in array and find rank location in the 2d array, and then select tps or fps based on y0 value
                    queue.parallel_for(sycl::range<1>(total_ranks), [=](sycl::id<1> i) {
                        bk::atomic_global_add(ps_ptr + 2*diff_ptr[i] + y0_ptr[i], Float(1))                     
                    }).wait_and_throw();

                    auto tps = pr::ndarray<Float, 1>::empty(queue, { total_ranks}, sycl::usm::alloc::device);
                    auto fps = pr::ndarray<Float, 1>::empty(queue, { total_ranks}, sycl::usm::alloc::device);
                    const Float* tps_ptr = tps.get_data();
                    const Float* fps_ptr = fps.get_data();

                    // extract fps and tps into separate arrays (keep clean to convince compiler to memcpy)
                    auto de_interleave_event = queue.parallel_for(sycl::range<1>(total_ranks), [=](sycl::id<1> i) {
                        fps_ptr[i] = ps_ptr[2*i];
                        tps_ptr[i] = ps_ptr[2*i + 1];                  
                    });

                    // cumulative sum the fps and tps (possibly do this as double?)
                    auto fps_event = cumulative_sum_1d(queue, fps, {de_interleave_event});
                    auto tps_event = cumulative_sum_1d(queue, tps, {de_interleave_event});

                    queue.wait_and_throw(); // do this instead to catch both at once

                    const Float tot_pos = tps_ptr[total_ranks - 1];
                    const Float tot_neg = fps_ptr[total_ranks - 1];

                    // STEP 4
      
                    // cannot use onedal reduction primitive without performance loss (i.e. two steps instead of one)
                    // note result is a double to maintain precision.
                    queue.parallel_for(sycl::range<1>(total_ranks-1), sycl::reduction(result, sycl::plus<>()), [=](sycl::id<1> i, auto &sum){
                        sum += (tps_ptr[i + 1] + tps_ptr[i]) * (fps_ptr[i + 1] - fps_ptr[i]);
                    }).wait_and_throw();
                    // something is wrong with my math
                    result += tpr_ptr[0] * fpr_ptr[0];

                    return double(1) - (double(.5) * result) / (tot_pos * tot_neg); // due to the nature of how the original sort was done
                }


template <typename Float>
struct compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        auto& queue = ctx.get_queue();
        const auto y0_table = input.get_y0();
        const auto y1_table = input.get_y1();

        auto y0 = pr::table2ndarray_1d<Float>(queue, y0_table, sycl::usm::alloc::device);
        auto y1 = pr::table2ndarray_1d<Float>(queue, y1_table, sycl::usm::alloc::device);

        return result_t{}.set_score(compute_roc_auc_score(queue, y0, y1);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const context_gpu& ctx,
                    const descriptor_t& desc,
                    const table& y0_table,
                    const table& y1_table,
                    double& res) {
        auto& queue = ctx.get_queue();

        auto y0 = pr::table2ndarray_1d<Float>(queue, y0_table, sycl::usm::alloc::device);
        auto y1 = pr::table2ndarray_1d<Float>(queue, y1_table, sycl::usm::alloc::device);

        res = compute_roc_auc_score(queue, y0, y1);
    }
#endif
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::roc_auc_score::backend

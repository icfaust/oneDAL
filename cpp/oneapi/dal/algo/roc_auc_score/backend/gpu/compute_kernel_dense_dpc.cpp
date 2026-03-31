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
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/detail/profiler.hpp"


namespace oneapi::dal::roc_auc_score::backend {

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

tepmlate <typename Float>
double compute_roc_auc_score(sycl::queue& queue,
                        pr::ndarray<Float, 1>& y0, // mutability is required for sorting
                        pr::ndarray<Float, 1>& y1,
                        const dal::backend::event_vector& deps = {}) {

                    // Steps:
                    // 1. sort the y1 data using key value pairs (sort primitive)
                    // 2. diff the sort data based on y1 using a gather approach (neq comparison to next element array)
                    // 3. add up instances per group of y0 (custom segmented sum kernel) true and false positive scores via cumsum (placement primitive)
                    // 4. integrate results for roc_auc_score (sum reduction primitive)
                    
                    // STEP 1
                    auto row_count = y1.get_dimension(0);

                    // sort y1 data using key value pairs (sort primitive) to get y0 and y1 in y1 ascending order
                    // this complicates/inverts some of the logic in calculating roc_auc_score but improves perf
                    // do not use dpl version due to hardware limitations
                    auto sort_event = pr::radix_sort_indices_inplace<Float, Float>(queue, y1, y0, deps);

                    // STEP 2
                    // done as integer to prevent rank spacing problems     
                    auto diff = pr::ndarray<std::uint32_t, 1>::ones(queue, { row_count }, sycl::usm::alloc::device);

                    auto base = dal::make_ndview(y0.get_data(), row_count - 1);
                    auto offset = dal::make_ndview(y0.get_data() + 1, row_count - 1);
                    auto diff_offset = dal::make_ndview(diff.get_data() + 1, row_count - 1);

                    const auto kernel_neq = [=](const Float a, const Float b) -> std::uint32_t {
                        return reinterpret_cast<std::uint32_t>(a != b);
                    };

                    auto diff_event = element_wise(queue, kernel_neq, base, offset, diff_offset, {sort_event});

                    // does this need to be done as integers to prevent large number spacing issues (probably)
                    rank_gen_event = cumulative_sum_1d(queue, diff, {diff_event});

                    // STEP 3
                    // create necessary arrays for doing the integration (tps and fps)
                    // requires first allocating 2 arrays size of the number of ranks and then
                    // a custom kernel for finding the total number of elements in that rank
                    const std::uint32_t* data_ptr = diff.get_data();
                    auto total_ranks = data_ptr[row_count - 1]
                    auto fps = pr::ndarray<Float, 1>::empty(queue, { total_ranks }, sycl::usm::alloc::device);
                    auto tps = pr::ndarray<Float, 1>::empty(queue, { total_ranks }, sycl::usm::alloc::device);

                    // custom kernel to add by group using atomic add

                }

template <typename Float>
struct compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        throw unimplemented(dal::detail::error_messages::method_not_implemented());
    }

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const context_gpu& ctx,
                    const descriptor_t& desc,
                    const table& y0,
                    const table& y1,
                    double& res) {
        throw unimplemented(dal::detail::error_messages::method_not_implemented());
    }
#endif
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::roc_auc_score::backend

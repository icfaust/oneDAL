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
                        const pr::ndview<Float, 1>& y0,
                        const pr::ndview<Float, 1>& y1,
                        const dal::backend::event_vector& deps = {}) {

                    // Steps:
                    // 1. sort the y1 data using key value pairs (sort primitive)
                    // 2. diff the sort data based on y1 using a gather approach (neq comparison to next element array)
                    // 3. add up instances per group of y0 (custom segmented sum kernel) true and false positive scores via cumsum (placement primitive)
                    // 4. integrate results for roc_auc_score (sum reduction primitive)
                    
                    // STEP 1
                    auto row_count = y1.get_dimension(0);

                    // create index array
                    auto ind = pr::ndarray<std::uint32_t, 1>::empty(queue, { row_count }, sycl::usm::alloc::device);
                    ind.arange(queue, deps).wait_and_throw();
                    
                    // sort y1 data using key value pairs (sort primitive)
                    // do not use dpl version due to hardware limitations
                    pr::radix_sort_indices_inplace<Float, std::uint_32t>(queue, y1, ind, deps).wait_and_throw();

                    // get y0 sorted using ind for use in true and false positive calculations
                    auto y0_sorted = pr::ndarray<Float, 1>::empty(queue, { row_count }, sycl::usm::alloc::device);

                    // important differentiation from daal approach (scatter) do it via gather for GPU
                    queue.submit([&](sycl::handler& cgh) {
                        const auto range = dal::backend::make_range_1d(row_count);
                        cgh.depends_on(deps);
                        cgh.parallel_for(range, [=](sycl::id<1> idx) {
                            y0_sorted[idx] = y0[ind[idx]];
                        });
                    }).wait_and_throw();

                    // STEP 2


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

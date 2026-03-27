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

#include <daal/include/data_management/data/internal/roc_auc_score.h>

#include "oneapi/dal/algo/roc_auc_score/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::roc_auc_score::backend {

using dal::backend::context_cpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace interop = dal::backend::interop;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& y0,
                                 const table& y1) {
    const auto daal_y0 = interop::convert_to_daal_table<Float>(y0);
    const auto daal_y1 = interop::convert_to_daal_table<Float>(y1);

    double score = daal::data_management::internal::rocAucScore<Float>(*daal_y0.get(), *daal_y1.get());
    return result_t().set_score(score);
}

template <typename Float>
static result_t compute(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_y0(), input.get_y1());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::dense, task::compute> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const context_cpu& ctx,
                    const descriptor_t& desc,
                    const table& y0,
                    const table& y1,
                    double& res) const {
        throw unimplemented(dal::detail::error_messages::method_not_implemented());
    }
#endif
};

template struct compute_kernel_cpu<float, method::dense, task::compute>;
template struct compute_kernel_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::roc_auc_score::backend

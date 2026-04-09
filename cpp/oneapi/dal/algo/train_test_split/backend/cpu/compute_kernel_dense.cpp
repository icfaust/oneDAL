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

#include <cmath>
#include <random>

#include <daal/include/data_management/data/internal/train_test_split.h>
#include <daal/include/data_management/data/homogen_numeric_table.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>

#include "oneapi/dal/algo/train_test_split/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/rng/host_engine.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::train_test_split::backend {

using dal::backend::context_cpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace interop = dal::backend::interop;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data) {
    auto daal_data = interop::convert_to_daal_table<Float>(data);

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t col_count = data.get_column_count();

    double train_s = desc.get_train_size();
    double test_s = desc.get_test_size();
    
    if (train_s < 0.0 && test_s < 0.0) {
        train_s = 0.75;
        test_s = 0.25;
    } else if (train_s < 0.0) {
        train_s = 1.0 - test_s;
    } else if (test_s < 0.0) {
        test_s = 1.0 - train_s;
    }

    std::int64_t train_count = std::floor(row_count * train_s);
    if (train_count == 0 && row_count > 0 && train_s > 0.0) train_count = 1;
    std::int64_t test_count = row_count - train_count;
    if (test_count == 0 && row_count > 0 && test_s > 0.0) {
        test_count = 1;
        train_count = row_count - 1;
    }

    if (row_count == 0) {
        return result_t().set_train_data(homogen_table::empty()).set_test_data(homogen_table::empty());
    }

    // Allocate DAAL index table for the entire dataset
    auto daal_indices = daal::data_management::HomogenNumericTable<int>::create(1, row_count, daal::data_management::NumericTable::doAllocate);

    if (desc.get_shuffle()) {
        const int MT19937_NUMBERS = 624;
        
        auto daal_rng_state = daal::data_management::HomogenNumericTable<int>::create(1, MT19937_NUMBERS, daal::data_management::NumericTable::doAllocate);
        int* rng_state_ptr = daal_rng_state->getArray();

        auto engine_type = dal::backend::primitives::convert_engine_method(desc.get_engine_type());
        dal::backend::primitives::host_engine eng(desc.get_seed(), engine_type);
        
        dal::backend::primitives::uniform<int>(MT19937_NUMBERS, rng_state_ptr, eng, 0, std::numeric_limits<int>::max());

        daal::data_management::internal::generateShuffledIndices<int>(daal_indices, daal_rng_state);
    } else {
        int* idx_ptr = daal_indices->getArray();
        for (std::int64_t i = 0; i < row_count; ++i) {
            idx_ptr[i] = static_cast<int>(i);
        }
    }

    auto daal_train_idx = daal::data_management::HomogenNumericTable<int>::create(1, train_count, daal::data_management::NumericTable::doAllocate);
    auto daal_test_idx = daal::data_management::HomogenNumericTable<int>::create(1, test_count, daal::data_management::NumericTable::doAllocate);

    int* full_idx = daal_indices->getArray();
    int* train_idx = daal_train_idx->getArray();
    int* test_idx = daal_test_idx->getArray();

    for (std::int64_t i = 0; i < train_count; ++i) {
        train_idx[i] = full_idx[i];
    }
    for (std::int64_t i = 0; i < test_count; ++i) {
        test_idx[i] = full_idx[train_count + i];
    }

    auto daal_train_table = daal::data_management::HomogenNumericTable<Float>::create(col_count, train_count, daal::data_management::NumericTable::doAllocate);
    auto daal_test_table = daal::data_management::HomogenNumericTable<Float>::create(col_count, test_count, daal::data_management::NumericTable::doAllocate);

    daal::data_management::internal::trainTestSplit<int>(daal_data, daal_train_table, daal_test_table, daal_train_idx, daal_test_idx);

    auto train_data_oneapi = interop::convert_from_daal_table<Float>(daal_train_table);
    auto test_data_oneapi = interop::convert_from_daal_table<Float>(daal_test_table);

    return result_t().set_train_data(train_data_oneapi).set_test_data(test_data_oneapi);
}

template <typename Float>
static result_t compute(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::dense, task::compute> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_cpu<float, method::dense, task::compute>;
template struct compute_kernel_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::train_test_split::backend

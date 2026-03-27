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

#include "oneapi/dal/algo/roc_auc_score/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::roc_auc_score {

template <typename Task>
class detail::v1::compute_input_impl : public base {
public:
    compute_input_impl(const table& y0, const table& y1) : y0(y0), y1(y1) {}
    table y0;
    table y1;
};

template <typename Task>
class detail::v1::compute_result_impl : public base {
public:
    double score = 0.0;
};

using detail::v1::compute_input_impl;
using detail::v1::compute_result_impl;

namespace v1 {

template <typename Task>
compute_input<Task>::compute_input(const table& y0, const table& y1) : impl_(new compute_input_impl<Task>(y0, y1)) {}

template <typename Task>
const table& compute_input<Task>::get_y0() const {
    return impl_->y0;
}

template <typename Task>
const table& compute_input<Task>::get_y1() const {
    return impl_->y1;
}

template <typename Task>
void compute_input<Task>::set_y0_impl(const table& value) {
    impl_->y0 = value;
}

template <typename Task>
void compute_input<Task>::set_y1_impl(const table& value) {
    impl_->y1 = value;
}

template class ONEDAL_EXPORT compute_input<task::compute>;

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
double compute_result<Task>::get_score() const {
    return impl_->score;
}

template <typename Task>
void compute_result<Task>::set_score_impl(double value) {
    impl_->score = value;
}

template class ONEDAL_EXPORT compute_result<task::compute>;

} // namespace v1
} // namespace oneapi::dal::roc_auc_score

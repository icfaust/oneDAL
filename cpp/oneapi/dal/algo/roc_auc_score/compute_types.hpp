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

#pragma once

#include "oneapi/dal/algo/roc_auc_score/common.hpp"

namespace oneapi::dal::roc_auc_score {

namespace detail {
namespace v1 {
template <typename Task>
class compute_input_impl;

template <typename Task>
class compute_result_impl;
} // namespace v1

using v1::compute_input_impl;
using v1::compute_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::compute`.
template <typename Task = task::by_default>
class compute_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`.
    compute_input(const table& truedata, const table& testdata);

    /// @remark default = table{}
    const table& get_truedata() const;
    const table& get_testdata() const;

    auto& set_truedata(const table& truedata) {
        set_truedata_impl(truedata);
        return *this;
    }

    auto& set_testdata(const table& testdata) {
        set_testdata_impl(testdata);
        return *this;
    }

protected:
    void set_truedata_impl(const table& truedata);
    void set_testdata_impl(const table& testdata);


private:
    dal::detail::pimpl<detail::compute_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::compute`.
template <typename Task = task::by_default>
class compute_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    compute_result();

    /// A double with the result Receiver Operating Characteristic Area Under the Curve.
    double get_score() const;

    auto& set_finite(const double& value) {
        set_score_impl(value);
        return *this;
    }

protected:
    void set_score_impl(const double&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;

} // namespace oneapi::dal::roc_auc_score

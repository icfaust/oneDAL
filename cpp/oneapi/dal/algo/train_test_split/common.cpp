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

#include "oneapi/dal/algo/train_test_split/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::train_test_split::detail {
namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    double train_size = -1.0;
    double test_size = -1.0;
    bool shuffle = true;
    engine_type engine = engine_type::mt19937;
    std::uint64_t seed = 777;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
double descriptor_base<Task>::get_train_size() const {
    return impl_->train_size;
}

template <typename Task>
void descriptor_base<Task>::set_train_size(double value) {
    if (value < 0.0 || value > 1.0) {
        // We defer constraint checking to the ops level, but we can set it here.
    }
    impl_->train_size = value;
}

template <typename Task>
double descriptor_base<Task>::get_test_size() const {
    return impl_->test_size;
}

template <typename Task>
void descriptor_base<Task>::set_test_size(double value) {
    impl_->test_size = value;
}

template <typename Task>
bool descriptor_base<Task>::get_shuffle() const {
    return impl_->shuffle;
}

template <typename Task>
void descriptor_base<Task>::set_shuffle(bool value) {
    impl_->shuffle = value;
}

template <typename Task>
engine_type descriptor_base<Task>::get_engine_type() const {
    return impl_->engine;
}

template <typename Task>
void descriptor_base<Task>::set_engine_type(engine_type value) {
    impl_->engine = value;
}

template <typename Task>
std::uint64_t descriptor_base<Task>::get_seed() const {
    return impl_->seed;
}

template <typename Task>
void descriptor_base<Task>::set_seed(std::uint64_t value) {
    impl_->seed = value;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace v1
} // namespace oneapi::dal::train_test_split::detail

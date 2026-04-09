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

#include "oneapi/dal/algo/train_test_split/compute.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::train_test_split::test {

namespace te = dal::test::engine;

template <typename TestType>
class train_test_split_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    void check_split(const te::dataframe& x_data,
                     double train_sz,
                     double test_sz,
                     bool shuffle,
                     const te::table_id& x_data_table_id) {
        const table x = x_data.get_table(this->get_policy(), x_data_table_id);
        const std::int64_t row_count = x.get_row_count();

        INFO("create descriptor");
        const auto split_desc =
            train_test_split::descriptor<Float, Method>{}
                .set_train_size(train_sz)
                .set_test_size(test_sz)
                .set_shuffle(shuffle);

        INFO("run compute");
        const auto compute_result = this->compute(split_desc, x);
        const table train_res = compute_result.get_train_data();
        const table test_res = compute_result.get_test_data();

        double explicit_train = train_sz;
        double explicit_test = test_sz;

        if (explicit_train < 0.0 && explicit_test < 0.0) {
            explicit_train = 0.75;
            explicit_test = 0.25;
        } else if (explicit_train < 0.0) {
            explicit_train = 1.0 - explicit_test;
        } else if (explicit_test < 0.0) {
            explicit_test = 1.0 - explicit_train;
        }

        std::int64_t expected_train_count = std::floor(row_count * explicit_train);
        if (expected_train_count == 0 && row_count > 0 && explicit_train > 0.0) expected_train_count = 1;
        std::int64_t expected_test_count = row_count - expected_train_count;
        if (expected_test_count == 0 && row_count > 0 && explicit_test > 0.0) {
            expected_test_count = 1;
            expected_train_count = row_count - 1;
        }

        REQUIRE(train_res.get_row_count() == expected_train_count);
        REQUIRE(test_res.get_row_count() == expected_test_count);
        REQUIRE(train_res.get_column_count() == x.get_column_count());
        REQUIRE(test_res.get_column_count() == x.get_column_count());
    }
};

using split_types = COMBINE_TYPES((float, double), (train_test_split::method::dense));

TEMPLATE_LIST_TEST_M(train_test_split_batch_test,
                     "train test split basic sizes",
                     "[train_test_split][integration][batch]",
                     split_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 5, 2 }.fill_normal(0, 1, 7777));

    const double train_sz = GENERATE(-1.0, 0.8, 0.5);
    const double test_sz = GENERATE(-1.0, 0.1);
    const bool shuffle = GENERATE(0, 1);

    if (train_sz == 0.8 && test_sz == 0.1) return; // Ignore invalid sum for this test
    if (train_sz == 0.5 && test_sz == 0.1) return;

    this->check_split(x_data, train_sz, test_sz, shuffle, this->get_homogen_table_id());
}

} // namespace oneapi::dal::train_test_split::test

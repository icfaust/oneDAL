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

#include <limits>
#include <cmath>

#include "oneapi/dal/algo/roc_auc_score/compute.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::roc_auc_score::test {

namespace te = dal::test::engine;

template <typename TestType>
class roc_auc_score_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    void check_roc_auc(const te::dataframe& y0_data,
                       const te::dataframe& y1_data) {
        const table y0 = y0_data.get_table(this->get_policy(), this->get_homogen_table_id());
        const table y1 = y1_data.get_table(this->get_policy(), this->get_homogen_table_id());

        INFO("create descriptor");
        const auto roc_desc = roc_auc_score::descriptor<Float, Method>{};

        INFO("run compute");
        const double score = dal::compute(this->get_policy(), roc_desc, y0, y1).get_score();
        REQUIRE(score >= 0.0);
    }
};

using roc_auc_types = COMBINE_TYPES((float, double), (roc_auc_score::method::dense));

TEMPLATE_LIST_TEST_M(roc_auc_score_batch_test,
                     "roc_auc_score typical",
                     "[roc_auc_score][integration][batch]",
                     roc_auc_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe y0_data = GENERATE_DATAFRAME(te::dataframe_builder{ 50, 1 }.fill_normal(0, 1, 7777));
    const te::dataframe y1_data = GENERATE_DATAFRAME(te::dataframe_builder{ 50, 1 }.fill_normal(0, 1, 7778));

    this->check_roc_auc(y0_data, y1_data);
}

} // namespace oneapi::dal::roc_auc_score::test

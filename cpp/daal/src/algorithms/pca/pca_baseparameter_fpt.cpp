/* file: pca_baseparameter_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Implementation of PCA algorithm interface.
//--
*/
#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{
/** Constructs PCA parameters */
template <typename algorithmFPType, Method method>
BaseParameter<algorithmFPType, method>::BaseParameter() {};

template DAAL_EXPORT BaseParameter<DAAL_FPTYPE, correlationDense>::BaseParameter();
template DAAL_EXPORT BaseParameter<DAAL_FPTYPE, svdDense>::BaseParameter();

} // namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal

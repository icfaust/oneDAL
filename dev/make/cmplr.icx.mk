#===============================================================================
# Copyright 2012-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#++
#  Intel compiler defenitions for makefile
#--

PLATs.icx = lnx32e

CMPLRDIRSUFF.icx = 

CORE.SERV.COMPILER.icx = generic

-Zl.icx =
-DEBC.icx = -g
-Qopt = -qopt-

COMPILER.lnx.icx = icpx $(if $(IA_is_ia32),-m32,-m64) -qopenmp-simd -stdlib=libstdc++ -fgnu-runtime -fwrapv \
                    -Werror -Wreturn-type

link.dynamic.lnx.icx = icpx $(if $(IA_is_ia32),-m32,-m64)

pedantic.opts.lnx.icx = -pedantic \
                        -Wall \
                        -Wextra \
                        -Wno-unused-parameter

daaldep.lnx32e.rt.icx = -static-intel
daaldep.lnx32.rt.icx  = -static-intel

p4_OPT.icx   = -march=nocona
mc_OPT.icx   = -march=core2
mc3_OPT.icx  = -march=nehalem
avx_OPT.icx  = -march=sandybridge
avx2_OPT.icx = -march=haswell
skx_OPT.icx  = -march=skx
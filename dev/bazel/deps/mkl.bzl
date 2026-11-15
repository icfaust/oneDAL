#===============================================================================
# Copyright 2020 Intel Corporation
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

load("@onedal//dev/bazel:repos.bzl", "repos")

# Building and linking needs the unversioned `.so` symlinks next to the
# versioned files (`libmkl_blas_sycl.so.6`). The PyPI packages ship only the
# latter, so the SYCL domain libraries are taken from Anaconda, which ships
# both. Headers, for which the distinction does not matter, stay on PyPI.
_VERSION = "2026.2.0"

# The dynamic core package is pinned separately from the rest, which are kept in
# lockstep on `_VERSION`.
_CORE_VERSION = "2026.2.0"

# Windows CPU/MSVC builds use the latest classic MKL PyPI wheels; Windows
# DPC++/SYCL dependencies are out of scope.
_WIN_VERSION = "2026.2.0"

mkl_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "MKLROOT",
    archives = [
        # Core static MKL runtime libraries (libmkl_intel_lp64.a, libmkl_core.a,
        # libmkl_sequential.a, ...), linked by the host build. The DPC build uses
        # dynamic classic MKL with the per-ISA dispatch kernels it `dlopen`s; see
        # `mkl_classic_dynamic` in `dev/bazel/deps/mkl.tpl.BUILD`.
        repos.archive(
            url = "https://anaconda.org/conda-forge/mkl-static/{v}/download/linux-64/mkl-static-{v}-ha770c72_881.conda".format(v = _VERSION),
            sha256 = "f16d373010a39409a0dac12a1587422ff06003a23edda53c125b53b2a5297ea9",
            # The .conda packages already unpack into the required layout.
            strip_prefix = "",
        ),
        # Headers of the classic C/C++ interface (`mkl.h` and friends). Does not
        # follow the oneAPI directory structure.
        repos.archive(
            url = "https://files.pythonhosted.org/packages/79/76/92504b3f0e1caa52c5dd1422a5f0ddac111b6339ccd2a3a87a657900553c/mkl_include-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "889cc72057a1671fee1e9a8beb2fc4e0d222ba78c08b6d69584dae03fbec4822",
            strip_prefix = "mkl_include-{}.data/data".format(_VERSION),
        ),
        # Headers of the DPC++/SYCL interface (`onemkl/*.hpp`).
        repos.archive(
            url = "https://files.pythonhosted.org/packages/29/85/56513df991f9ba59a1c5e9046a279371f764ccae78fd8144d45d844b8c35/onemkl_sycl_include-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "97bebbac7c41cd6df0c9c7e2cc45e143582190687d80995d2f0934bd061e457a",
            strip_prefix = "onemkl_sycl_include-{}.data/data".format(_VERSION),
        ),
        # SYCL BLAS domain: libmkl_sycl_blas.so.6 (GEMM, AXPY, ...).
        repos.archive(
            url = "https://anaconda.org/conda-forge/onemkl-sycl-blas/{v}/download/linux-64/onemkl-sycl-blas-{v}-h0165e16_881.conda".format(v = _VERSION),
            sha256 = "b787d56500d39d4ab823536715d9bd75dee79581994122f86302f8cd3889defa",
            strip_prefix = "",
        ),
        # SYCL LAPACK domain: libmkl_sycl_lapack.so.6 (LU, QR, eigenvalues).
        repos.archive(
            url = "https://anaconda.org/conda-forge/onemkl-sycl-lapack/{v}/download/linux-64/onemkl-sycl-lapack-{v}-h0165e16_881.conda".format(v = _VERSION),
            sha256 = "eff8eb57d26e00ea70e21f8961e35ee28f9c79c2c9761ef6194c2b33617d39ce",
            strip_prefix = "",
        ),
        # SYCL sparse domain: libmkl_sycl_sparse.so.6.
        repos.archive(
            url = "https://anaconda.org/conda-forge/onemkl-sycl-sparse/{v}/download/linux-64/onemkl-sycl-sparse-{v}-h0165e16_881.conda".format(v = _VERSION),
            sha256 = "057efdd9c4c1ba8324a12fa64d6ab16861b567633390013dd733d76f28b01216",
            strip_prefix = "",
        ),
        # SYCL RNG domain: libmkl_sycl_rng.so.6.
        repos.archive(
            url = "https://anaconda.org/conda-forge/onemkl-sycl-rng/{v}/download/linux-64/onemkl-sycl-rng-{v}-h0165e16_881.conda".format(v = _VERSION),
            sha256 = "b671bfc98816eb9d722643d8c01413cd48e32166bcf9ae55006032b45aace77f",
            strip_prefix = "",
        ),
        # Dynamic core: libmkl_core.so.3, required for SYCL execution.
        repos.archive(
            url = "https://anaconda.org/conda-forge/mkl/{v}/download/linux-64/mkl-{v}-hd2095e1_881.conda".format(v = _CORE_VERSION),
            sha256 = "172814c644b9146c9f746bba9b9215ecb4afd003de8b8cb730ad3dc93ae0aad7",
            strip_prefix = "",
        ),
    ],
    win_archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/16/18/31a0738a9170fe2fa732ab3f2a66715e459987f9e0fb925b48f7e3d113e8/mkl_static-{}-py2.py3-none-win_amd64.whl".format(_WIN_VERSION),
            sha256 = "df7274c5b5ccee7f9544f9744bb80b8e4b9801becd522c7972534053da36418f",
            strip_prefix = "mkl_static-{}.data/data/Library".format(_WIN_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/31/bc/0e8380d982d353bad1f01d4828830563e53d309eeb38d5be3d6f519d83cb/mkl-{}-py2.py3-none-win_amd64.whl".format(_WIN_VERSION),
            sha256 = "5f2c187f0dda47efa19c1922cb3c8815924b8e5ed1613d28a3a69ec2327d4045",
            strip_prefix = "mkl-{}.data/data/Library".format(_WIN_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/d0/29/c91e211e3effac2ed2e55d7e5f725bd11389fb3b2efa3957a7fc9ac9b87f/mkl_devel-{}-py2.py3-none-win_amd64.whl".format(_WIN_VERSION),
            sha256 = "06d0d20dd119b1960fc4224adc9d09b5f0dd7295e45fd27ecef9e1b409758f55",
            strip_prefix = "mkl_devel-{}.data/data/Library".format(_WIN_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/2a/bb/e62ab5c1adebdc3e2950c18f7475b9619c8f9ffd441ce65a86daa9667d8d/mkl_include-{}-py2.py3-none-win_amd64.whl".format(_WIN_VERSION),
            sha256 = "e07acc96c043de9e8c72ddfd624455ce58f077d6bd95db7854ab2bc4f7c323f5",
            strip_prefix = "mkl_include-{}.data/data/Library".format(_WIN_VERSION),
        ),
    ],
    includes = [
        "include",
    ],
    libs = [
        "lib/libmkl_core.a",
        "lib/libmkl_intel_ilp64.a",
        "lib/libmkl_tbb_thread.a",
        "lib/libmkl_core.so*",
        "lib/libmkl_intel_lp64.so*",
        "lib/libmkl_gnu_thread.so*",
        "lib/libmkl_sycl_blas.so*",
        "lib/libmkl_sycl_lapack.so*",
        "lib/libmkl_sycl_sparse.so*",
        "lib/libmkl_sycl_rng.so*",

    ],
    # VML kernels are `dlopen`-ed like the CPU dispatch kernels below and the
    # pinned package ships them (`libmkl_vml_def`, `_avx2`, `_avx512`, `_mc3`,
    # `_cmpt`), so they are symlinked when present. They stay merely optional
    # rather than collectively required because the failure measured below is the
    # dispatch kernels': no oneDAL target was observed to need a VML kernel, and
    # requiring a file no measurement implicates would be a guess.
    optional_libs = [
        "lib/libmkl_vml_*.so*",
    ],
    # CPU dispatch kernels. `libmkl_core.so.3` holds only the dispatcher: the
    # actual kernels live in per-ISA shared objects that it `dlopen`s by SONAME
    # at the first classic-MKL call. Without any of them in the repository every
    # target that reaches classic MKL at run time dies with
    #   INTEL oneMKL ERROR: .../libmkl_avx512.so.3: cannot open shared object
    #     file
    #   Intel oneMKL FATAL ERROR: Cannot load libmkl_avx512.so.3 or
    #     libmkl_def.so.3.
    # Which ISA families a given release ships changes between versions (the
    # pinned 2025 package has no `libmkl_avx.so*`/`libmkl_mc.so*`), so no single
    # family can be required -- that would break host-only builds on a perfectly
    # good package. But *none* of them is not a layout to build against, so the
    # set is collectively required: present families are symlinked, and a package
    # with the classic libraries and no kernel fails here, naming the package,
    # rather than at the first `gemm` call, naming a file the user never asked
    # for. They end up in the same `cc_library` as `libmkl_core.so.3`
    # (`mkl_classic_dynamic` in mkl.tpl.BUILD), which is where the dispatcher
    # looks; see the comment there.
    #
    # Windows needs no counterpart: `win_bins` globs `bin/*.dll`, which is a
    # required pattern and covers `mkl_def.2.dll` and the ISA kernels with it.
    required_any_libs = [
        "lib/libmkl_def.so*",
        "lib/libmkl_avx*.so*",
        "lib/libmkl_mc*.so*",
    ],
    required_any_libs_description = "a classic MKL CPU dispatch kernel",
    build_template = "@onedal//dev/bazel/deps:mkl.tpl.BUILD",
    win_includes = [
        "include",
    ],
    win_libs = [
        "lib/mkl_core.lib",
        "lib/mkl_intel_ilp64.lib",
        # Globbed so the debug-CRT threading layer (`mkl_tbb_threadd.lib`,
        # see dev/make/deps.mkl.mk:51) is symlinked too when the MKL layout
        # ships it. `mkl_win.tpl.BUILD` names both variants explicitly in its
        # select(), so a debug-runtime build with the file absent fails with a
        # plain missing-input error instead of silently linking the release one.
        "lib/mkl_tbb_thread*.lib",
        "lib/mkl_core_dll.lib",
        "lib/mkl_intel_lp64_dll.lib",
        "lib/mkl_intel_thread_dll.lib",
        "lib/mkl_sycl_blas_dll.lib",
        "lib/mkl_sycl_lapack_dll.lib",
        "lib/mkl_sycl_rng_dll.lib",
        "lib/mkl_sycl_sparse_dll.lib",
    ],
    win_bins = [
        "bin/*.dll",
    ],
    win_build_template = "@onedal//dev/bazel/deps:mkl_win.tpl.BUILD",
)

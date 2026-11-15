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

_VERSION = "2023.2.0"

tbb_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "TBBROOT",
    archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/59/fd/8ca476c7b2513b1e64e08d3f5404c8ed19b34d0213000b5e23b3a91676c2/tbb-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "7d80615b6e970f11671fa05b6eaa9483ac864fbfe525ac280271a78b37870b7c",
            strip_prefix = "tbb-{}.data/data".format(_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/1b/8e/b1c0f027f6907c2e4963d7ba853c87ca90781a14389cec8375a0d8a6d55c/tbb_devel-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "83d004c948614a3d36c10e19e10f9620eeee927cc68cee29442b94ec609e31d0",
            strip_prefix = "tbb_devel-{}.data/data".format(_VERSION),
        ),
    ],
    win_archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/d0/41/2aeac799ab57e81898c8cc24f503fe301e12f19758bb1fe34d522764d3bb/tbb-{}-py3-none-win_amd64.whl".format(_VERSION),
            sha256 = "b39121cbdaff350fc61a448766cbb26807c523491d66e364838c9feeb3c6a0eb",
            strip_prefix = "tbb-{}.data/data/Library".format(_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/91/3e/f6aa800f22976484abe75469d45cb3756d1f786de4b299859c6c587cd792/tbb_devel-{}-py3-none-win_amd64.whl".format(_VERSION),
            sha256 = "1655eb481004cadd2dbfdd98e98f43513a73a7f45c42076e6fc0f2abd424a459",
            strip_prefix = "tbb_devel-{}.data/data/Library".format(_VERSION),
        ),
    ],
    includes = [
        "include",
    ],
    libs = [
        "lib/libtbb.so",
        "lib/libtbb.so.12",
        "lib/libtbbmalloc.so",
        "lib/libtbbmalloc.so.2",
    ],
    build_template = "@onedal//dev/bazel/deps:tbb.tpl.BUILD",
    win_includes = [
        "include",
    ],
    # Globs rather than exact names so the `_debug` variants required by an
    # `-MDd` build (`tbb12_debug.lib`, see makefile:330) are picked up when the
    # TBB layout ships them, without making the default release build fail on
    # installations that do not. `tbb_win.tpl.BUILD` names the debug files
    # explicitly in its `select()`, so a debug-runtime build that is missing
    # them fails with a plain missing-input error instead of silently
    # dropping the dependency.
    win_libs = [
        "lib/tbb12*.lib",
        "lib/tbbmalloc*.lib",
    ],
    win_bins = [
        "bin/tbb12*.dll",
        "bin/tbbmalloc*.dll",
    ],
    win_build_template = "@onedal//dev/bazel/deps:tbb_win.tpl.BUILD",
)

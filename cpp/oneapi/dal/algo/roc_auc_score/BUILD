package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "roc_auc_score",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
        "@onedal//cpp/oneapi/dal/backend/primitives:sort",
    ],
    extra_deps = [
        "@onedal//cpp/daal:data_management",
    ]
)

dal_test_suite(
    name = "cpu_tests",
    private = True,
    compile_as = [ "c++" ],
    srcs = glob([
        "backend/cpu/test/*.cpp",
    ]),
    dal_deps = [
        ":roc_auc_score",
    ],
)

dal_test_suite(
    name = "gpu_tests",
    private = True,
    compile_as = [ "dpc++" ],
    srcs = glob([
        "backend/gpu/test/*.cpp",
    ]),
    dal_deps = [
        ":roc_auc_score",
    ],
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":roc_auc_score",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
        ":gpu_tests",
        ":interface_tests",
    ],
)

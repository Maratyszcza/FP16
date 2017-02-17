#!/usr/bin/env python


import confu
parser = confu.standard_parser("FP16 configuration script")


def main(args):
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["fp16.h"])

    with build.options(source_dir="test", extra_include_dirs="test", deps=[build.deps.googletest, build.deps.psimd]):
        fp16_tables = build.cxx("tables.cc")
        build.unittest("ieee-to-fp32-bits",
            [build.cxx("ieee-to-fp32-bits.cc"), fp16_tables])
        build.unittest("ieee-to-fp32-value",
            [build.cxx("ieee-to-fp32-value.cc"), fp16_tables])
        build.unittest("ieee-from-fp32-value",
            [build.cxx("ieee-from-fp32-value.cc"), fp16_tables])

        build.unittest("alt-to-fp32-bits",
            [build.cxx("alt-to-fp32-bits.cc"), fp16_tables])
        build.unittest("alt-to-fp32-value",
            [build.cxx("alt-to-fp32-value.cc"), fp16_tables])
        build.unittest("alt-from-fp32-value",
            [build.cxx("alt-from-fp32-value.cc"), fp16_tables])

        build.unittest("ieee-to-fp32-psimd", build.cxx("ieee-to-fp32-psimd.cc"))
        build.unittest("alt-to-fp32-psimd", build.cxx("alt-to-fp32-psimd.cc"))

        build.unittest("bitcasts", build.cxx("bitcasts.cc"))

    with build.options(source_dir="bench", deps=[build.deps.googlebenchmark, build.deps.psimd]):
        build.benchmark("ieee-bench", build.cxx("ieee.cc"))
        build.benchmark("alt-bench", build.cxx("alt.cc"))

    return build


if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()

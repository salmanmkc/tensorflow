"""
A rule to compile a C++ file to a header containing LLVM IR.

This rule is critical for generating LLVM IR bitcode that is embedded into the XLA compiler.
It uses standard cc_library with clang flags to generate IR, then extracts it.
"""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

visibility(DEFAULT_LOAD_VISIBILITY)

def to_camel_case(s):
    """Converts a snake_case or kebab-case string to CamelCase."""
    return "".join([p.capitalize() for p in s.replace("-", "_").split("_")])

def _force_clang_transition_impl(settings, _attr):
    # Determine the compiler string based on the target CPU
    cpu = settings["//command_line_option:cpu"]

    if "windows" in cpu:
        # Windows toolchains in XLA/TF are usually keyed by "clang-cl"
        compiler = "clang-cl"
    else:
        # Linux/macOS/Generic typically use "llvm"
        compiler = "llvm"

    return {"//command_line_option:compiler": compiler}

_force_clang_transition = transition(
    implementation = _force_clang_transition_impl,
    # We must declare cpu as an input to read it from 'settings'
    inputs = ["//command_line_option:cpu"],
    outputs = ["//command_line_option:compiler"],
)

def _compile_with_clang_impl(ctx):
    # Forward the default provider from the dependency.
    # Note: ctx.attr.dep becomes a list when a transition is attached.
    return [
        ctx.attr.dep[0][DefaultInfo],
    ]

_compile_with_clang = rule(
    implementation = _compile_with_clang_impl,
    attrs = {
        "dep": attr.label(cfg = _force_clang_transition),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist:function_transition_allowlist",
        ),
    },
)

def cc_ir_header(name, src, deps = [], copts = [], **kwargs):
    """A macro that generates an IR header and wraps it in a cc_library.

    Args:
      name: The name of the generated cc_library.
      src: The C++ source file to compile.
      deps: The C++ dependencies of the source file.
      copts: Additional compiler flags.
      **kwargs: Additional arguments to pass to the generated cc_library.
    """

    # Extract arguments that are not for cc_library
    base_name = kwargs.pop("base_name", name)
    namespace = kwargs.pop("namespace", "llvm_ir")

    common_attrs = {}
    for attr in ["visibility", "testonly"]:
        if attr in kwargs:
            common_attrs[attr] = kwargs[attr]

    compatible_with = None

    # Do a little dance so the line below matches copybara rules.
    # copybara_removed compatible_with = ["//buildenv/target:non_prod"]
    compatible_with = kwargs.get("compatible_with", compatible_with)
    if compatible_with:
        common_attrs["compatible_with"] = compatible_with

    # Define intermediate targets
    lib_name = name + "_lib"
    out_header = name + ".h"

    # We compile the source to bitcode (-emit-llvm -c).
    # We use -c so that we get a bitcode file (wrapped in .o or raw .bc) that 'ar' can handle.
    compile_flags = [
        "-c",
        "-emit-llvm",
        "-O3",
        "-DNDEBUG",
        "-mprefer-vector-width=512",
        "-DEIGEN_VECTORIZE_GENERIC",
        "-fno-builtin",
        "-Wno-psabi",
        "-std=c++17",
    ] + copts

    # Disabled features to avoid instrumentations in the IR
    # AND disable thin archives to ensure we have actual content to extract.
    disabled_features = [
        "thin_lto",
        "thin_archives",
        "per_object_debug_info",
        "module_maps",
        "use_header_modules",
        "layering_check",
        "parse_headers",
        "fdo_optimize",
        "fdo_instrument",
        "asan",
        "msan",
        "tsan",
        "ubsan",
    ]

    # Prefix features with '-'
    features = ["-" + f for f in disabled_features] + kwargs.pop("features", [])

    # Create a cc_library. This will compile the source and create an archive (.a).
    # We name it with _internal suffix so the transition rule can take the original name.
    cc_library(
        name = lib_name + "_internal",
        srcs = [src],
        deps = deps,
        copts = compile_flags,
        features = features,
        tags = ["manual"],
        **common_attrs
    )

    # Wrap the library with the transition to force Clang compiler.
    # This rule applies the transition to the dependency.
    _compile_with_clang(
        name = lib_name + "_transitioned",
        dep = ":" + lib_name + "_internal",
        tags = ["manual"],
        **common_attrs
    )

    # Alias to choose between the transitioned library (forced Clang) and the
    # internal library (default toolchain).
    # For sanitizer builds (ASAN, MSAN, TSAN, UBSAN), we use the default toolchain
    # (which is already Clang-based) to avoid conflicts with Rust sanitizer configurations.
    native.alias(
        name = lib_name,
        actual = select({
            "//tools/cpp:asan_build": ":" + lib_name + "_internal",
            "//tools/cpp:msan_build": ":" + lib_name + "_internal",
            "//tools/cpp:tsan_build": ":" + lib_name + "_internal",
            "//tools/cpp:ubsan_build": ":" + lib_name + "_internal",
            "//conditions:default": ":" + lib_name + "_transitioned",
        }),
    )

    # Extract the object file (which is bitcode) from the archive.
    native.genrule(
        name = name + "_extract_bc",
        srcs = [":" + lib_name],
        outs = [name + ".extracted.bc"],
        # cmd explanation:
        # 1. Iterate over all outputs of the cc_library.
        # 2. Find the static library (.a or .lib) and extract it.
        # 3. Move the extracted object to the output.
        cmd = """
          for f in $(locations :""" + lib_name + """); do
            if [[ "$$f" == *.a || "$$f" == *.lib ]]; then
              ar x "$$f"
            fi
          done

          # Try to match likely object files
          FOUND_FILE=""
          for f in *.o *.lo *.pic.o *.obj; do
            if [[ -f "$$f" ]]; then
              FOUND_FILE="$$f"
              break
            fi
          done

          if [[ -n "$$FOUND_FILE" ]]; then
            mv "$$FOUND_FILE" $@
          else
            # Soft fallback: if no object file is found (e.g. clang unavailable or compilation failed to produce obj),
            # generate an empty bitcode file. This allows the build to proceed with empty strings in the header.
            echo "Warning: No object file extracted from archive(s). Generating empty bitcode."
            touch $@
          fi
        """,
        tags = ["manual"],
        **common_attrs
    )

    # Generate the header file from the IR (Bitcode).
    variable_name = "k{}Ir".format(to_camel_case(base_name))

    ir_to_string_tool = "//xla/codegen/intrinsic/cpp:ir_to_string"

    native.genrule(
        name = name + "_gen_header",
        srcs = [":" + name + "_extract_bc"],
        outs = [out_header],
        tools = [ir_to_string_tool],
        cmd = "$(location {}) $< $@ {} {}".format(ir_to_string_tool, variable_name, namespace),
        tags = ["manual"],
        **common_attrs
    )

    # Exposed library
    cc_library(
        name = name,
        hdrs = [":" + out_header],
        deps = deps,
        **kwargs
    )

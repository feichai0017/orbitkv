# DeepGEMM patch stack

## 0001-torch-cxx20.patch

Applies to SGLang DeepGEMM `v0.1.5.post3` /
`fa3a5ca07d768dd0f9089f70a445208b166c48d1`.

The release wheel scripts and fallback module builder hard-code C++17 even
though this revision's top-level CMake already selects C++20. PyTorch 2.13+
headers require C++20 (`std::strong_ordering`, concepts, and string-view
`starts_with`/`ends_with`), so the SGLang wheel build fails before linking.

The Aletheia source-workspace builder checks and applies this patch only for the pinned
revision, builds the wheel, then reverses it in a `finally` block. The submodule
therefore remains clean and at the exact upstream commit. Remove this patch once
the pinned SGLang DeepGEMM release uses C++20 in all three packaging paths.

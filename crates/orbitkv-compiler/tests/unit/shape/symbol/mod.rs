use crate::prelude::*;

/// The whole reason `to_kernel_with_index` exists. Callers used to string-
/// replace `const_z` in generated source, which is prefix-unsafe once dim
/// names can exceed one char: `const_z` is a prefix of `const_zip`, so a
/// dim named `zip` would be rewritten mid-name to `iip`.
///
/// Substituting on the term means only the reserved index moves and every
/// dim is emitted untouched. `zip` is not spellable while `Symbol` is a
/// char, so the prefix case itself gets pinned when the newtype lands.
#[test]
fn index_substitution_only_touches_the_reserved_index() {
    let e = expr('z') + expr('a');
    assert_eq!(e.to_kernel_with_index("i"), "(i+const_a)");
    assert!(!e.to_kernel_with_index("i").contains("const_z"));
}

#[test]
fn to_kernel_defaults_to_const_z_for_the_index() {
    assert_eq!((expr('z') + expr('a')).to_kernel(), "(const_z+const_a)");
}

/// Generated kernels declare a local `long long const_z` for the thread
/// index, so a dim may never be named `z` — it would `#define` over that
/// local.
#[test]
fn reserved_index_is_z_and_is_not_a_dyn_var() {
    assert_eq!(Symbol::reserved_index().to_string(), "z");
    assert_eq!(kernel_const_name(&Symbol::reserved_index()), "const_z");
    assert!(expr('z').dyn_vars().is_empty());
    assert_eq!(expr('a').dyn_vars(), vec![sym("a")]);
}

#[cfg(test)]
mod no_const_prefix_surgery {
    use std::path::Path;

    fn scan(dir: &Path, needle: &str, hits: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if name != "target" && name != ".git" {
                    scan(&path, needle, hits);
                }
            } else if name.ends_with(".rs")
                && let Ok(src) = std::fs::read_to_string(&path)
                && src.contains(needle)
            {
                hits.push(path.display().to_string());
            }
        }
    }

    /// Rewriting generated kernel source by string-replacing a `const_`-prefixed
    /// name is prefix-unsafe: `const_z` sits inside `const_zip`, so the surgery
    /// lands in the middle of an unrelated dim's name and silently miscompiles.
    ///
    /// `Expression::to_kernel_with_index` / `to_kernel_with` substitute on the
    /// term instead, before any string exists. Nothing stops someone reaching
    /// for the old pattern again, so fail the build if they do.
    #[test]
    fn no_source_file_string_replaces_a_const_prefix() {
        // Assembled so this file does not match its own needles.
        //
        // Two forms, because the original bug had the second one: Metal built
        // the name with `format!("const_{symbol}")` and then replaced *that*,
        // so a needle matching only a string literal would have missed it.
        let literal = format!("{}(\"const_", "replace");
        let built = format!("{}!(\"const_", "format");
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));

        let mut hits = Vec::new();
        {
            let dir = root
                .parent()
                .expect("compiler belongs to the workspace crates directory")
                .to_path_buf();
            assert!(dir.is_dir(), "expected to scan {}", dir.display());
            scan(&dir, &literal, &mut hits);
            // Building the name is only a problem when it feeds a rewrite, so
            // require a rewrite call in the same file rather than flagging
            // every construction site — kernel_const_name is a legitimate one.
            // Needles are assembled rather than written literally, or this
            // file matches itself.
            let mut built_hits = Vec::new();
            scan(&dir, &built, &mut built_hits);
            // Assembled, or this file matches its own filter: it contains
            // kernel_const_name's legitimate `format!("const_…")`.
            let rewrite = format!(".{}(", "replace");
            let rewrite_n = format!(".{}(", "replacen");
            hits.extend(built_hits.into_iter().filter(|f| {
                std::fs::read_to_string(f)
                    .map(|src| src.contains(&rewrite) || src.contains(&rewrite_n))
                    .unwrap_or(false)
            }));
        }
        hits.sort();
        hits.dedup();

        // A lint that cannot tell "clean" from "never looked" is worse than no
        // lint: scan() swallows read_dir errors and the needles are assembled
        // at runtime, so prove the scanner finds something it should.
        let mut control = Vec::new();
        scan(&root.join("src"), "fn is_well_formed", &mut control);
        assert!(
            !control.is_empty(),
            "scanner found nothing at all — the lint is not actually looking"
        );

        assert!(
            hits.is_empty(),
            "these files rewrite generated source with {literal}…) or {built}…), which is \
         prefix-unsafe — a dim named `zip` emits `const_zip` and the replace \
         lands mid-name. Use Expression::to_kernel_with_index / to_kernel_with \
         instead, which substitute on the term:\n  {}",
            hits.join("\n  ")
        );
    }
}

/// Symbol #26 used to land on 'z', the reserved loop index, which
/// `dyn_vars()` filters out — so the dim vanished from the buffer planner
/// and every search candidate died evaluating its size.
#[test]
fn reserved_index_cannot_be_minted_as_a_dim() {
    assert_eq!(Symbol::try_new_dim("z"), Err(InvalidSymbolName::Reserved));
    assert!(Symbol::try_new("z").is_ok(), "still nameable as the index");
}

#[test]
#[should_panic(expected = "reserved runtime loop index")]
fn set_dim_rejects_the_reserved_index() {
    let mut cx = Graph::new();
    cx.set_dim('z', 4);
}

/// `set_dim` is not the only way a dim gets named — a shape can name one
/// directly, and that path had no guard. `cx.tensor(('z', 4))` built fine,
/// reported no dyn_vars (so the buffer planner never saw the dim), and
/// emitted `const_z`, which in generated CUDA is the thread index.
#[test]
#[should_panic(expected = "reserved runtime loop index")]
fn a_shape_cannot_be_sized_by_the_loop_index() {
    let mut cx = Graph::default();
    let _ = cx.tensor(('z', 4));
}

/// ...but a *stride* legitimately uses it, which is what makes the guard
/// specific to dimension sizes rather than to expressions generally.
#[test]
fn strides_may_use_the_loop_index() {
    let cx_shape = ShapeTracker::new((4, 8));
    assert!(
        cx_shape.strides.iter().any(|s| s.uses_reserved_index()),
        "strides are indexed by the loop var"
    );
    assert!(!cx_shape.dims.iter().any(|d| d.uses_reserved_index()));
}

/// Equality is by name, so a char literal and a string name a single dim.
/// The `.egg` rewrite rules match `(MVar "s")` literally, so this is what
/// keeps hand-written models hitting them.
#[test]
fn char_and_str_spellings_name_the_same_dim() {
    assert_eq!(Symbol::from('s'), sym("s"));
    assert_eq!(expr('s'), expr(sym("s")));
    assert_eq!(expr('s').to_egglog(), "(MVar \"s\")");
}

/// Ordering drives `dyn_dims[]` slot assignment, and the host uploads in
/// that same order. Sorting by name keeps it a function of the graph — for
/// single-char names, byte-identical to the char ordering this replaced.
#[test]
fn ordering_is_by_name_and_matches_the_old_char_order() {
    let mut got: Vec<Symbol> = ['s', 'a', 'Z', 'A', 'c'].map(Symbol::from).to_vec();
    got.sort();
    let mut want = ['s', 'a', 'Z', 'A', 'c'];
    want.sort();
    assert_eq!(
        got,
        want.iter().copied().map(Symbol::from).collect::<Vec<_>>()
    );

    // Independent of the order the names were first interned in.
    let _ = sym("zz9");
    let _ = sym("aa9");
    let mut a = [sym("zz9"), sym("aa9")];
    a.sort();
    assert_eq!(a, [sym("aa9"), sym("zz9")]);
}

/// `From<char>` used to hand back any ASCII byte straight from the name
/// static ASCII table this branch later deleted, skipping validation
/// entirely — so `expr('%')` produced
/// `const_%`, not a C identifier, and `Symbol::from('"')` produced
/// `(MVar """)`, not an egglog literal. The module claims the alphabet
/// holds *by construction* so codegen need not re-check; this is what
/// makes that true.
#[test]
fn char_conversion_is_validated_too() {
    for c in ['a', 'z', 'A', 'Z'] {
        assert_eq!(Symbol::from(c).to_string(), c.to_string());
    }
    for c in ['%', '"', '-', '7', ' ', '\\', '_'] {
        assert!(
            std::panic::catch_unwind(|| Symbol::from(c)).is_err(),
            "Symbol::from({c:?}) must not mint an unusable name"
        );
    }
}

/// The seam this whole change exists to fix, joined end to end.
///
/// `simplify()` round-trips through egglog: `to_egglog` encodes the dim as
/// `(MVar "s77")`, egglog runs, and extraction decodes it back. Extraction
/// used to take `name.chars().next()`, so `s77` came back as `s` — a
/// *different, possibly live* dim — and the wrong name then missed in
/// `dyn_map`, where the CUDA backend turns a miss into a dimension of 0. Wrong
/// numerics, no diagnostic.
///
/// The decoder is unit-tested in egglog_utils, but nothing ran a
/// multi-char name through the real encode/egglog/decode path until this.
#[test]
fn multichar_dim_survives_a_simplify_round_trip() {
    let s = expr(sym("s77"));

    // More than one term, so this takes the egglog path rather than
    // Expression::simplify's single-term shortcut.
    let simplified = ((s * 2) / 2).simplify();

    assert_eq!(
        simplified.dyn_vars(),
        vec![sym("s77")],
        "name must survive egglog; truncation would yield s"
    );

    let mut env = DynMap::default();
    env.insert(sym("s77"), 12);
    assert_eq!(
        simplified.exec(&env),
        Some(12),
        "a truncated name misses in dyn_map and resolves to nothing"
    );
}

/// A dim is emitted as `const_<name>`, so the alphabet has to keep that a
/// valid C identifier. Rejected, never sanitized: a sanitizer maps `a.b`
/// and `a-b` both to `a_b`, silently collapsing two dims onto one #define.
#[test]
fn malformed_names_are_rejected_not_mangled() {
    for bad in ["s-77", "_x", "a__b", "7s", "a b", "a\"b", ""] {
        assert!(
            Symbol::try_new(bad).is_err(),
            "{bad:?} should not be a dim name"
        );
    }
    assert_eq!(kernel_const_name(&sym("s77")), "const_s77");
}

/// The prefix hazard that `to_kernel_with_index` exists to prevent, now
/// that a dim named `zip` is actually spellable.
#[test]
fn index_substitution_does_not_clobber_a_z_prefixed_dim() {
    let e = expr('z') + expr(sym("zip"));
    let out = e.to_kernel_with_index("i");
    assert_eq!(out, "(i+const_zip)");
}

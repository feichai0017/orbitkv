use super::*;
use crate::shape::sym;
use proptest::prelude::*;

#[test]
fn test_empty_product_is_one() {
    // The empty product (e.g. for a rank-0 tensor's shape) must be the
    // multiplicative identity, 1 — not 0. the CUDA backend and other kernel
    // emitters use `shape.iter().product()` to compute `numel`, and a
    // rank-0 tensor has 1 element. Returning 0 here would yield a CUDA
    // launch with grid=(0, 1, 1) and crash at runtime.
    let empty: Vec<Expression> = vec![];
    assert_eq!(
        empty.into_iter().product::<Expression>(),
        Expression::from(1)
    );
}

#[test]
fn test_empty_sum_is_zero() {
    // Sanity check the additive identity stays 0 (it always was).
    let empty: Vec<Expression> = vec![];
    assert_eq!(empty.into_iter().sum::<Expression>(), Expression::from(0));
}

#[test]
fn test_basic_simplifications() {
    let x = expr('x');
    let a = expr('a');
    // Identity operations simplify away
    assert_eq!(((a * 1) + 0) / 1 + (1 - 1), a);
    // Evaluation after simplification
    let n = (x + (256 - (x % 256))).simplify();
    assert_eq!(
        n.exec(&[(sym("x"), 767)].into_iter().collect()).unwrap(),
        768
    );
}

#[test]
fn test_merge_dim_simplifications() {
    assert!((((expr('z') / 3) * 3) + (expr('z') % 3)).simplify().len() == 1);
}

#[test]
fn test_const_remainder_div_mod_simplifications() {
    let z = expr('z');

    assert_eq!((expr(5) / 6).simplify(), expr(0));
    assert_eq!((expr(5) % 6).simplify(), expr(5));
    assert_eq!(((z * 6 + 5) / 6).simplify(), z);
    assert_eq!(((z * 6 + 5) % 6).simplify(), expr(5));
    assert_eq!(
        (((z * 6 + 5) / 6) * 6 + ((z * 6 + 5) % 6)).simplify(),
        z * 6 + 5
    );
}

#[test]
fn test_interval_simplifications() {
    let s = expr('s');
    let intervals = [(sym("s"), DimInterval::new(0, 127))].into_iter().collect();

    assert_eq!((s % 128).simplify_with_intervals(&intervals), s);
    assert_eq!((s / 128).simplify_with_intervals(&intervals), expr(0));
    assert_eq!(s.lt(128).simplify_with_intervals(&intervals), expr(1));
    assert_eq!(s.gte(128).simplify_with_intervals(&intervals), expr(0));
    assert_eq!(s.min(128).simplify_with_intervals(&intervals), s);
}

#[test]
fn test_add_num_does_not_fold_into_nested_num() {
    // Regression: adding an integer to `(a*b) + rest` must not fold the
    // integer into the `a` of the multiplication. `(11*16) + 15` is 191,
    // not `11*(16+15) = 341`. The Add fast-path used to fold into the
    // leading Num token regardless of whether it was a top-level operand.
    let empty = FxHashMap::default();
    // (11*16) + 15 must be 191, not 11*(16+15) = 341.
    assert_eq!((expr(11) * 16 + 15).exec(&empty), Some(191));
    // A symbolic chain `(s*384 + (11*16)) + 15` must keep the constant 191.
    let s = expr('s');
    let chain = (s * 384 + expr(11) * 16) + 15;
    assert_eq!(chain.substitute('s', 2).exec(&empty), Some(768 + 191));
}

#[test]
fn test_singleton_interval_substitutes_dynamic_var() {
    let s = expr('s');
    let intervals = [(sym("s"), DimInterval::new(1, 1))].into_iter().collect();

    assert_eq!((s + 127).simplify_with_intervals(&intervals), expr(128));
    assert_eq!((s.lt(2)).simplify_with_intervals(&intervals), expr(1));
}

#[test]
fn test_interval_simplification_requires_proof() {
    let s = expr('s');
    let intervals = [(sym("s"), DimInterval::new(0, 256))].into_iter().collect();

    assert_ne!((s % 128).simplify_with_intervals(&intervals), s);
    assert_ne!(s.lt(128).simplify_with_intervals(&intervals), expr(1));
}

#[test]
fn test_lt_mod_shortcut_requires_literal_bound() {
    let z = expr('z');
    let range = expr(651) / 4; // 162
    let upper = expr(1) + (expr(643) / 4); // 161, but not a literal expression
    let mask = (z % range).lt(upper);

    assert_eq!(mask.exec_single_var_checked(160), Some(1));
    assert_eq!(mask.exec_single_var_checked(161), Some(0));
}

#[test]
fn test_substitution() {
    let x = expr('x');
    let new = (x - 255).substitute('x', x / 2).simplify();
    assert_eq!(new.len(), 5);
}

#[test]
fn test_group_terms() {
    let s = expr('s');
    let expr = (s * ((s - 4) + 1)) + (((s + 1) * ((s - 4) + 1)) - (s * ((s - 4) + 1)));
    assert_eq!(expr.simplify().len(), 7);
}

#[test]
fn test_egglog_equality() {
    let a = expr('a');
    let b = expr('b');
    assert!((a + (b - a)).egglog_equal(b));
    assert!(!(a + 1).egglog_equal(a + 2));
}

#[test]
fn test_simplify() {
    let (z, w, h, s) = (expr('z'), expr('w'), expr('h'), expr('s'));
    // Nested divisions combine: ((((w + 3) / 2) + 2) / 2) -> (w + 7) / 4
    assert_eq!(((((w + 3) / 2) + 2) / 2).simplify(), (w + 7) / 4);
    // Complex division simplification
    let o = (z
        / ((-5 + (((((-5 + ((((((w + 153) / 2) / 2) / 2) / 2) / 2)) * 4) + 9) / 2) / 2))
            * (-5 + (((9 + (4 * (-5 + ((((((153 + h) / 2) / 2) / 2) / 2) / 2)))) / 2) / 2))))
        % 64;
    assert!(o.simplify().len() <= 27);
    // // Mul-div simplification
    // let x = z % (((((153 + h) / 8) + -31) * ((((w + 153) / 8) + -31) / 16)) * 64);
    // assert!(x.simplify().len() < 15);
    // Like-term combining: 1+s+8+s+12+s+1+s+3+s+8+s+3+s+11+s+15+s+8+s+19 -> 10*s + 89
    let x: Expression =
        (((((((((((((((((((1 + s) + 8) + s) + 12) + s) + 1) + s) + 3) + s) + 8) + s) + 3)
            + s)
            + 11)
            + s)
            + 15)
            + s)
            + 8)
            + s)
            + 19;
    assert!(x.simplify().egglog_equal((s * 10) + 89));
}

#[test]
fn test_no_explode() {
    // This expression previously caused e-graph explosion with naive associativity rules
    let x: Expression = 1 + ((8 / expr(32)) + 27);
    x.simplify();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn test_simplify_preserves_eval(x_val in 0usize..100, y_val in 0usize..100, z_val in 0usize..100) {
        let (x, y, z) = (expr('x'), expr('y'), expr('z'));
        // Simplification preserves evaluation
        let expr = ((x + 3) * 2) - (x * 2) + (y % 5);
        let env = [(sym("x"), x_val), (sym("y"), y_val)].into_iter().collect();
        assert_eq!(expr.exec(&env).unwrap(), expr.simplify().exec(&env).unwrap());
        // Substitution + simplification preserves evaluation
        let expr = (x + y) * (y - x);
        let substituted = expr.substitute('x', z + 1).substitute('y', z - 1);
        let env = [(sym("z"), z_val)].into_iter().collect();
        assert_eq!(substituted.exec(&env).unwrap(), substituted.simplify().exec(&env).unwrap());
    }
}

#[test]
fn test_hash_consing() {
    // Creating identical expressions should return the same underlying storage
    // Use a unique variable name to avoid interference from other tests.
    // This used to need a private-use-area char, because a dim was a char
    // and every readable one was potentially in use by another test.
    let unique_var = sym("hashConsingProbe");

    // Create expression with unique var + 42
    let x1 = expr(unique_var) + 42;

    // Create the same expression again - should reuse storage
    let x2 = expr(unique_var) + 42;

    // The expressions should be equal
    assert_eq!(x1, x2);

    // They should share the same GenerationalBox (same id)
    // This is the key test for hash consing - identical terms = same box
    assert_eq!(
        x1.terms.id(),
        x2.terms.id(),
        "Hash consing failed: identical expressions should share storage"
    );

    // Different expression should create new entry
    let unique_var2 = sym("hashConsingProbe2");
    let y = expr(unique_var2) + 43;
    assert_ne!(
        x1.terms.id(),
        y.terms.id(),
        "Different expressions should have different storage"
    );
}

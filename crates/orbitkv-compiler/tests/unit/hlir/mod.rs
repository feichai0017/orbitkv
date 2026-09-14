use crate::egglog_utils::run_egglog;
use crate::op::IntoEgglogOp;

use super::*;

#[test]
fn loop_inputs_with_the_same_stream_values_are_unified() {
    let ops = <HLIROps as IntoEgglogOp>::into_vec();
    let program = r#"
        (let x0 (Input 0 "x0" (Int)))
        (let x1 (Input 1 "x1" (Int)))
        (let values (ICons x0 (ICons x1 (INil))))
        (let a (Op (LoopInput 7 3 (Int)) values))
        (let b (Op (LoopInput 7 9 (Int)) values))
        (let out_a (Output a 0 false))
        (let out_b (Output b 1 false))
        (let root (OutputJoin out_a out_b))
    "#;
    let egraph = run_egglog(program, "root", &ops, false).unwrap();
    let unified = egraph
        .eclasses
        .values()
        .filter(|(sort, nodes)| {
            sort == "IR"
                && nodes
                    .iter()
                    .filter(|node| {
                        let (label, children) = &egraph.enodes[*node];
                        label == "Op"
                            && egraph.eclasses[&children[0]]
                                .1
                                .iter()
                                .any(|kind| egraph.enodes[kind].0 == "LoopInput")
                    })
                    .count()
                    == 2
        })
        .count();

    assert_eq!(unified, 1);
}

fn assert_f64_unary(op: &dyn ReferenceOp, input: &[f64], expected_fn: fn(f64) -> f64) {
    let input_data = ReferenceData::F64(input.to_vec());
    let actual = op.execute(vec![&input_data], &FxHashMap::default());
    let ReferenceData::F64(actual) = actual else {
        panic!("F64 unary input must produce an F64 reference buffer")
    };
    let expected: Vec<f64> = input.iter().copied().map(expected_fn).collect();

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn reference_unary_ops_execute_f64_natively() {
    let input = [0.25, 0.5, 1.0, 2.0, 4.0];
    let shape = vec![input.len().into()];
    let strides = vec!['z'.into()];

    assert_f64_unary(
        &Log2 {
            shape: shape.clone(),
            strides: strides.clone(),
            ..Default::default()
        },
        &input,
        f64::log2,
    );
    assert_f64_unary(
        &Exp2 {
            shape: shape.clone(),
            strides: strides.clone(),
            ..Default::default()
        },
        &input,
        f64::exp2,
    );
    assert_f64_unary(
        &Sin {
            shape: shape.clone(),
            strides: strides.clone(),
            ..Default::default()
        },
        &input,
        f64::sin,
    );
    assert_f64_unary(
        &Recip {
            shape: shape.clone(),
            strides: strides.clone(),
            ..Default::default()
        },
        &input,
        f64::recip,
    );
    assert_f64_unary(
        &Sqrt {
            shape,
            strides,
            ..Default::default()
        },
        &input,
        f64::sqrt,
    );
}

#[test]
fn reference_narrow_integer_casts_preserve_native_widths() {
    let source = ReferenceData::Int(vec![-32_769, -129, -128, -1, 0, 127, 128, 255, 256]);
    let dyn_map = FxHashMap::default();

    let i8_data = Cast(9.into(), DType::I8).execute(vec![&source], &dyn_map);
    assert!(matches!(
        i8_data,
        ReferenceData::I8(ref values)
            if values == &[-1, 127, -128, -1, 0, 127, -128, -1, 0]
    ));

    let u8_data = Cast(9.into(), DType::U8).execute(vec![&source], &dyn_map);
    assert!(matches!(
        u8_data,
        ReferenceData::U8(ref values)
            if values == &[255, 127, 128, 255, 0, 127, 128, 255, 0]
    ));

    let i16_data = Cast(9.into(), DType::I16).execute(vec![&source], &dyn_map);
    assert!(matches!(
        i16_data,
        ReferenceData::I16(ref values)
            if values == &[32767, -129, -128, -1, 0, 127, 128, 255, 256]
    ));
}

#[test]
fn reference_narrow_integer_add_wraps_in_declared_dtype() {
    let op = Add {
        shape: vec![2.into()],
        a_strides: vec!['z'.into()],
        b_strides: vec!['z'.into()],
        ..Default::default()
    };
    let dyn_map = FxHashMap::default();

    let i8_lhs = ReferenceData::I8(vec![127, -128]);
    let i8_rhs = ReferenceData::I8(vec![1, -1]);
    assert!(matches!(
        op.execute(vec![&i8_lhs, &i8_rhs], &dyn_map),
        ReferenceData::I8(values) if values == [-128, 127]
    ));

    let u8_lhs = ReferenceData::U8(vec![255, 0]);
    let u8_rhs = ReferenceData::U8(vec![1, 255]);
    assert!(matches!(
        op.execute(vec![&u8_lhs, &u8_rhs], &dyn_map),
        ReferenceData::U8(values) if values == [0, 255]
    ));

    let i16_lhs = ReferenceData::I16(vec![32_767, -32_768]);
    let i16_rhs = ReferenceData::I16(vec![1, -1]);
    assert!(matches!(
        op.execute(vec![&i16_lhs, &i16_rhs], &dyn_map),
        ReferenceData::I16(values) if values == [-32_768, 32_767]
    ));
}

fn round_tripped(v: f32) -> f32 {
    let s = Constant(v).to_egglog(&[]);
    let inner = &s["(Op (Constant ".len()..s.len() - ") (INil))".len()];
    // The egglog Constant sort stores f64: text -> f64 -> f32 is the
    // path a constant takes through the e-graph and back.
    inner
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("unparseable constant text {inner:?}")) as f32
}

/// f32 -> serialized text -> f64 (egglog) -> f32 must be the identity.
/// `{:.6}` zeroed sub-5e-7 constants (gelu's sign epsilon -> NaN at
/// x==0) and shifted transcendental coefficients (LUM-631).
#[test]
fn constant_to_egglog_round_trips_exactly() {
    let adversarial = [
        0.0f32,
        -0.0,
        1e-10,
        -1e-10,
        f32::EPSILON,
        f32::MIN_POSITIVE,
        1e-45,       // smallest subnormal
        1.595_769_2, // tanh-gelu outer coeff (frontend 1.5957691216 as f32)
        0.044715,
        std::f32::consts::LOG2_E,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        1e38,
        -1e-38,
        0.1,
        1.0 / 3.0,
    ];
    for &v in &adversarial {
        assert_eq!(round_tripped(v).to_bits(), v.to_bits(), "constant {v:?}");
    }
}

#[test]
fn f64_constant_to_egglog_round_trips_exactly() {
    let adversarial = [
        0.0f64,
        -0.0,
        1.000_000_000_000_000_2,
        f64::EPSILON,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        std::f64::consts::PI,
        1e300,
        -1e-300,
    ];

    for value in adversarial {
        let serialized = ConstantF64(value).to_egglog(&[]);
        let prefix = "(Op (ConstantF64 ";
        let suffix = ") (INil))";
        let inner = &serialized[prefix.len()..serialized.len() - suffix.len()];
        let round_tripped = inner
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("unparseable F64 constant text {inner:?}"));
        assert_eq!(
            round_tripped.to_bits(),
            value.to_bits(),
            "F64 constant changed across egglog serialization: {value:?}"
        );
    }
}

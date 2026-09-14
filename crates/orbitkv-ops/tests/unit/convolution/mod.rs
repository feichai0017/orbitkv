use super::ConvND;
use candle_core::{Device, Tensor};

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (idx, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4 {
            panic!("values differ at {idx}: {lhs} vs {rhs} (diff {diff})");
        }
    }
}

fn candle_conv1d_output(
    conv: &ConvND,
    input: &[f32],
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> candle_core::Result<Vec<f32>> {
    let device = Device::Cpu;
    let input = Tensor::from_vec(input.to_vec(), (1, conv.ch_in, width), &device)?;
    let weight = Tensor::from_vec(
        weight.to_vec(),
        (conv.ch_out, conv.ch_in, conv.kernel[0]),
        &device,
    )?;
    let bias = match bias {
        Some(b) => Some(Tensor::from_vec(b.to_vec(), conv.ch_out, &device)?),
        None => None,
    };

    let output = input.conv1d(
        &weight,
        conv.padding[0],
        conv.stride[0],
        conv.dilation[0],
        1,
    )?;
    let output = match bias {
        Some(bias) => {
            let bias = bias.reshape((1, conv.ch_out, 1))?;
            output.broadcast_add(&bias)?
        }
        None => output,
    };
    output.flatten_all()?.to_vec1::<f32>()
}

fn candle_conv2d_output(
    conv: &ConvND,
    input: &[f32],
    height: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> candle_core::Result<Vec<f32>> {
    let device = Device::Cpu;
    let input = Tensor::from_vec(input.to_vec(), (1, conv.ch_in, height, width), &device)?;
    let weight = Tensor::from_vec(
        weight.to_vec(),
        (conv.ch_out, conv.ch_in, conv.kernel[0], conv.kernel[1]),
        &device,
    )?;
    let bias = match bias {
        Some(b) => Some(Tensor::from_vec(b.to_vec(), conv.ch_out, &device)?),
        None => None,
    };

    assert_eq!(
        conv.padding[0], conv.padding[1],
        "Candle conv2d only supports equal padding"
    );
    assert_eq!(
        conv.stride[0], conv.stride[1],
        "Candle conv2d only supports equal stride"
    );
    assert_eq!(
        conv.dilation[0], conv.dilation[1],
        "Candle conv2d only supports equal dilation"
    );

    let output = input.conv2d(
        &weight,
        conv.padding[0],
        conv.stride[0],
        conv.dilation[0],
        1,
    )?;
    let output = match bias {
        Some(bias) => {
            let bias = bias.reshape((1, conv.ch_out, 1, 1))?;
            output.broadcast_add(&bias)?
        }
        None => output,
    };
    output.flatten_all()?.to_vec1::<f32>()
}

#[test]
fn conv1d_values_match_expected_window_sums() -> candle_core::Result<()> {
    let mut cx = orbitkv_compiler::graph::Graph::new();
    let conv = ConvND::new(1, 1, vec![3], vec![1], vec![1], vec![1], true, &mut cx);

    let input = [1., 2., 3., 4., 5.];
    let weight = [1., 1., 1.];
    let bias = [0.5];

    let out = candle_conv1d_output(&conv, &input, input.len(), &weight, Some(&bias))?;

    assert_close(&out, &[3.5, 6.5, 9.5, 12.5, 9.5]);
    Ok(())
}

#[test]
fn conv2d_values_accumulate_across_channels() -> candle_core::Result<()> {
    let mut cx = orbitkv_compiler::graph::Graph::new();
    let conv = ConvND::new(
        2,
        1,
        vec![2, 2],
        vec![1, 1],
        vec![1, 1],
        vec![0, 0],
        true,
        &mut cx,
    );

    let input = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., // channel 0
        9., 8., 7., 6., 5., 4., 3., 2., 1., // channel 1
    ];
    let weight = [1., 1., 1., 1., 2., 2., 2., 2.];
    let bias = [0.25];

    let out = candle_conv2d_output(&conv, &input, 3, 3, &weight, Some(&bias))?;

    assert_close(&out, &[68.25, 64.25, 56.25, 52.25]);
    Ok(())
}

#[test]
fn conv1d_shapes_follow_stride_and_padding() {
    let mut cx = orbitkv_compiler::graph::Graph::new();
    let conv = ConvND::new(1, 1, vec![3], vec![2], vec![1], vec![1], false, &mut cx);

    // expected length: floor((padded_len - dilation*(k-1) -1)/stride +1)
    // padded_len = 7 + 2 = 9
    // effective kernel = 3
    // => (9 -3)/2 +1 = 4
    let inferred = conv.infer_output_shape(&[2, 1, 7]);
    assert_eq!(inferred, vec![2, 1, 4]);
}

#[test]
fn conv2d_shapes_follow_stride_and_padding() {
    let mut cx = orbitkv_compiler::graph::Graph::new();
    let conv = ConvND::new(
        3,
        2,
        vec![2, 3],
        vec![1, 2],
        vec![1, 1],
        vec![0, 1],
        true,
        &mut cx,
    );

    // height: (5 - dilation*(2-1) -1 + 0 +0)/1 +1 = 4
    // width: (6 - dilation*(3-1) -1 + 1 +1)/2 +1 = 3
    let inferred = conv.infer_output_shape(&[1, 3, 5, 6]);
    assert_eq!(inferred, vec![1, 2, 4, 3]);
}

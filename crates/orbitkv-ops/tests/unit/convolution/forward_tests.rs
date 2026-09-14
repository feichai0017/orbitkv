use super::ConvND;
use orbitkv_compiler::prelude::*;

/// ConvND forward vs a naive host-side convolution, on the reference
/// runtime. Covers multi-channel input/output and a batch dimension.
#[test]
fn convnd_2d_matches_naive() {
    let (b, ci, co, h, w, k) = (2usize, 2usize, 3usize, 4usize, 4usize, 2usize);
    let (oh, ow) = (h - k + 1, w - k + 1);
    let x_data: Vec<f32> = (0..b * ci * h * w)
        .map(|i| (i as f32 * 0.13).sin())
        .collect();
    let w_data: Vec<f32> = (0..co * ci * k * k)
        .map(|i| (i as f32 * 0.29).cos())
        .collect();

    let mut cx = Graph::new();
    let x = cx.tensor((b, ci, h, w));
    let conv = ConvND::new(
        ci,
        co,
        vec![k, k],
        vec![1, 1],
        vec![1, 1],
        vec![0, 0],
        false,
        &mut cx,
    );
    let out = conv.forward(x).output();
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(
        ReferenceRuntime::default(),
        CompileOptions::default().search_graph_limit(1),
    );
    rt.set_data(x.id, x_data.clone());
    rt.set_data(conv.weight.id, w_data.clone());
    rt.execute(&cx.dyn_map);
    let got = rt.get_f32(out.id).clone();

    // Naive conv; weight layout is (co, ci * k * k), ci-major.
    let mut want = vec![0.0f32; b * co * oh * ow];
    for bi in 0..b {
        for o in 0..co {
            for y in 0..oh {
                for xx in 0..ow {
                    let mut acc = 0.0;
                    for c in 0..ci {
                        for dy in 0..k {
                            for dx in 0..k {
                                let xv = x_data[((bi * ci + c) * h + y + dy) * w + xx + dx];
                                let wv = w_data[(o * ci + c) * k * k + dy * k + dx];
                                acc += xv * wv;
                            }
                        }
                    }
                    want[((bi * co + o) * oh + y) * ow + xx] = acc;
                }
            }
        }
    }
    for (i, (g, e)) in got.iter().zip(&want).enumerate() {
        assert!((g - e).abs() < 1e-4, "element {i}: got {g}, want {e}");
    }
}

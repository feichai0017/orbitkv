//! Deterministic host fixtures shared by H20 gates and matched benchmarks.

use half::bf16;

pub fn deterministic_bf16(len: usize, salt: u64) -> Vec<bf16> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let signed = (state % 2001) as i32 - 1000;
        values.push(bf16::from_f32(signed as f32 / 2048.0));
    }
    values
}

/// Counts references in a closed test fixture's physical page table.
pub fn page_refcounts(max_num_pages: usize, page_indices: &[i32]) -> Vec<i32> {
    let mut counts = vec![0_i32; max_num_pages];
    for &page in page_indices {
        let page = usize::try_from(page).expect("fixture page indices are nonnegative");
        assert!(page < max_num_pages, "fixture page index is in range");
        counts[page] += 1;
    }
    counts
}

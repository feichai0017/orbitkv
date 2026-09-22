use super::*;

#[test]
fn test_numa_node_display() {
    assert_eq!(format!("{}", NumaNode(0)), "NUMA0");
    assert_eq!(format!("{}", NumaNode(7)), "NUMA7");
    assert_eq!(format!("{}", NumaNode::UNKNOWN), "UNKNOWN");
}

#[test]
fn test_pin_unknown_node_fails() {
    let result = pin_thread_to_numa_node(NumaNode::UNKNOWN);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unknown"));
}

#[test]
fn test_parse_cpulist_range() {
    let cpus = parse_cpulist("0-3").unwrap();
    assert_eq!(cpus, vec![0, 1, 2, 3]);
}

#[test]
fn test_parse_cpulist_list() {
    let cpus = parse_cpulist("0,4,8").unwrap();
    assert_eq!(cpus, vec![0, 4, 8]);
}

#[test]
fn test_parse_cpulist_mixed() {
    let cpus = parse_cpulist("0-2,8,16-17").unwrap();
    assert_eq!(cpus, vec![0, 1, 2, 8, 16, 17]);
}

#[test]
fn test_parse_cpulist_hyperthreading() {
    let cpus = parse_cpulist("0-15,32-47").unwrap();
    assert_eq!(cpus.len(), 32);
    assert_eq!(cpus[0], 0);
    assert_eq!(cpus[15], 15);
    assert_eq!(cpus[16], 32);
    assert_eq!(cpus[31], 47);
}

#[test]
fn test_parse_cpulist_empty() {
    let cpus = parse_cpulist("").unwrap();
    assert!(cpus.is_empty());
}

#[test]
fn test_parse_cpulist_single_cpu() {
    let cpus = parse_cpulist("5").unwrap();
    assert_eq!(cpus, vec![5]);
}

#[test]
fn test_gpu_numa_nodes_are_valid_sorted_unique() {
    let topology = NumaTopology {
        gpu_numa_map: HashMap::from([
            (0, NumaNode(3)),
            (1, NumaNode(3)),
            (2, NumaNode::UNKNOWN),
            (3, NumaNode(0)),
            (4, NumaNode(5)),
        ]),
        numa_nodes: vec![
            NumaNode(0),
            NumaNode(1),
            NumaNode(2),
            NumaNode(3),
            NumaNode(4),
            NumaNode(5),
        ],
    };

    assert_eq!(
        topology.gpu_numa_nodes(),
        vec![NumaNode(0), NumaNode(3), NumaNode(5)]
    );
}

#[test]
fn closest_cpu_numa_node_uses_first_reported_id() {
    assert_eq!(
        parse_closest_cpu_numa_node("NUMA ID of closest CPU: 3\n"),
        NumaNode(3)
    );
    assert_eq!(
        parse_closest_cpu_numa_node("NUMA IDs of closest CPU: 1,18-33\n"),
        NumaNode(1)
    );
}

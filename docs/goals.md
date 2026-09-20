# Goals

1. **A Data Path Purpose-Built for LLM Inference**

   Focus exclusively on the typical data flows in large model inference: data movement between GPU and CPU, between compute nodes, and high-throughput transport of KV cache blocks. We only solve this specific class of problems—high-bandwidth, predictable, structured data paths.

2. **Mooncake-Backed, High-Performance Remote Movement**

   Reuse the pinned upstream Mooncake Transfer Engine for RDMA, GPUDirect, TCP
   fallback, topology discovery, connection management, and completion. OrbitKV
   optimizes cache semantics and transfer plans instead of maintaining a second
   verbs stack.

3. **Developer-Friendly Abstractions**

   Provide clear, minimal transport semantics and channel models: easy to understand, simple to integrate, and predictable in behavior. Avoid hidden policies that cause mysterious performance jitter, allowing users to make confident performance assumptions.

4. **Built-In Observability and Tunability**

   Export key metrics and debugging information from day one (throughput, latency distribution, resource utilization, error signals, etc.), giving cluster operators data to guide topology and parameter tuning—rather than black-box trial-and-error.

5. **Embeddable in Existing Inference Systems**

   Serve as an optional "transport backend" that can plug into existing inference/dispatch/scheduling components—without requiring a full rewrite of the upper layers—ensuring the PoC can be validated quickly in real production stacks.

# Non-Goals

1. **Not a General-Purpose RPC or Service Framework**

   No request routing, load balancing, IDL, or serialization format wars—these concerns belong to upper layers or other projects.

2. **Not a Universal Network Virtualization Layer**

   No attempt to automatically adapt to all network environments, cloud providers, or dynamic topologies; the initial focus is deep optimization for known, controlled, performance-sensitive clusters.

3. **Not a Full-Featured Communication Middleware**

   Does not cover collectives, group communication semantics, or a comprehensive flow control ecosystem—only focused on high-value point-to-point (or few-node) bulk transfer scenarios.

4. **Not a Transport Protocol Lab**

   Do not duplicate Mooncake transports inside OrbitKV. Portability follows the
   capabilities of the pinned Mooncake runtime; OrbitKV's contribution is the
   cache authority, lifecycle, and physical plan above it.

5. **Not a Security or Compliance Component**

   No built-in complex authentication, encryption, or multi-tenant isolation; default assumption is deployment in controlled environments, with security handled by infrastructure.

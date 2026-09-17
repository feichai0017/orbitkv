这份整理后的文档调整了整体文风（从“草稿/自述笔记”转为一份**清晰、严谨且富有启发性的技术背景与设计调研指南**），去除了死板的实施路线图，转而把设计难点提炼为**开放探索课题**，同时完整保留了核心概念、类型与状态边界的设计哲学，以及相关的参考体系。

---

# Model-Agnostic GPU Executable Runtime: 背景与设计探索

## 1. 问题与核心动机

当前主流 LLM 推理框架（Inference Framework）的痛点在于：**框架对模型内部结构的感知过深**。

每当需要支持一个新模型或新架构时，通常需要侵入式地修改框架：
```text
新增模型架构代码 + 算子实现 + Kernel Glue Code + 模型特化分支适配
```
这会导致推理引擎代码量剧增且维护成本极高。

我们希望将新模型的支持收敛为一种 **数据驱动（Data-driven）的执行机制**：

```text
模型拓扑声明 (Launch/Executable IR)
               +
       一组编译好的 Binary (如 .cubin)
               ↓
          Thin Runtime
               ↓
            执行推理
```

* **目标**：从“在框架内编写模型实现 + 拼装 Kernel”，转变为“声明可执行程序拓扑（Executable Program） + 传入编译好的二进制 Kernel”。
* **核心边界**：模型 Architecture、Kernel 实现 DSL（CUDA/Triton/CuTe/CUTLASS 等）与 Serving Runtime 彻底解耦。

---

## 2. 核心抽象与设计哲学

### 2.1 Kernel 是 Opaque Executable
Runtime 无需关心 Kernel 是如何编写和编译的，也不感知具体算子语义（GEMM、RMSNorm、FlashAttention）。
* Kernel 在 IR 中被视为具有 **Typed Signature（类型化签名）** 的不透明二进制入口。
* 优化、融合（Fusion）或量化（如 BF16 替换为 FP8）本质上只是更新了 `.cubin` 和对应的 Launch Graph，不需要在 Serving 引擎内部重新实现特化逻辑。

### 2.2 Forward 是一个 Typed Pure Function
不要将 KV Cache 等状态视为算子内部隐蔽的“副作用”，更好的建模方式是将 Forward 抽象为一个纯函数：

$$\text{Forward}(\text{Inputs}, \text{State}) \to (\text{Outputs}, \text{NewState})$$

* 模型内部的计算图依然是纯粹的数据流。
* Runtime 真正需要理解的是 **Forward Function 的类型化边界**。

### 2.3 Serving State 只存在于边界
LLM Serving 不可能做到 $100\%$ 不感知模型语义，其根源在于 **Serving State（KV Cache、Sliding Window State、SSM/Mamba State）的管理**（生命周期、分配、Paged/Prefix 缓存、跨 Step 持久化等）。

* **Runtime 的职责**：不理解具体模型计算，只负责调度 Kernel 与管理 Serving State。
* **边界隔离**：`State<DenseKV>`、`State<Mamba>` 等高级状态类型**仅存在于 Program 边界**；进入内部计算图后，被解构为常规的 `buffer`、`pointer` 和 `scalar offset`。

---

## 3. Launch IR 的概念原型

在 Compiler（或导出工具）与 Runtime 之间，我们需要一个比 ONNX/StableHLO 更低层、专为 GPU Launch 设计的 **Launch / Executable IR**。

### 3.1 为什么需要类型系统？
如果 IR 内部仅使用无类型的原始指针（Raw Pointer），类型错误（例如将 `buffer<fp8>` 误接至 `buffer<bf16>`）只能在运行时暴露甚至直接引发 GPU Crash。

IR 需要极小但完备的静态类型系统（用于 Artifact 加载时的 Verify 阶段）：
* **Buffer Types**：`buffer<bf16>`, `buffer<fp8>`, `buffer<i32>` ...
* **Scalar Types**：`i32`, `i64`, `f32` ...
* **State Types**：`state<DenseKV>`, `state<SWA>` ...（仅限函数接口）

### 3.2 最小 Primitive 集合（供参考的概念模型）
一个极简的 Launch IR 概念结构示意：

```text
module @qwen_decode {
    // 1. Opaque Kernel 声明
    kernel @gemm_k0
        binary = "kernels.cubin"
        symbol = "gemm_kernel"
        : (buffer<bf16>, buffer<bf16>, buffer<bf16>, i32) -> ()

    // 2. 类型化 Forward 拓扑
    func @forward(
        %input: buffer<bf16>,
        %weight: buffer<bf16>,
        %state: state<DenseKV>,
        %tokens: i32
    ) -> (buffer<bf16>, state<DenseKV>) {

        // SSA 依赖驱动的节点调度与内存计算
        %grid_x = ceil_div(%tokens, 128)
        %tmp0 = alloc ...

        dispatch @gemm_k0 launch_grid(%grid_x, 1, 1)(
            %input, %weight, %tmp0, %tokens
        )

        return %tmp0, %state
    }
}
```

---

## 4. 相关领域与已有方案参考

在构思整体架构时，可重点对比和参考以下系统：

| 系统 / 技术 | 其设计理念与启发 | 我们与它的差异 / 我们的取舍 |
| :--- | :--- | :--- |
| **IREE HAL** | 具有极佳的 `Executable`、`EntryPoint`、`Buffer`、`Binding`、`Dispatch` 抽象，架构最为契合。 | IREE 考虑多后端通用性；我们可以更专注在 CUDA 体系与 LLM 特有 Serving 场景做极薄抽象。 |
| **AOTInductor** | 验证了 `PyTorch -> Compile -> Generated C++ / Cubin` 方案的可行性。 | 可以作为 Launch IR 的 Upstream Producer（模型前端产生源）之一。 |
| **TensorRT Engine** | `Model -> Builder -> Serialized Engine -> Thin Runtime` 的工业界标杆。 | 其内部 Executable Representation 为闭源黑盒；我们期望探索开放、可验证、可组合的 IR。 |
| **MLIR / xDSL** | 提供了成熟的 SSA、Dialect、Type Verifier、Pass 系统。 | 是构建 Launch IR AST、Type Checker 和 Serialization 的潜在基础设施。 |
| **ONNX / StableHLO** | 专注于高级 Tensor/Operator 语义描述。 | 抽象层级偏高，Runtime 仍需承担算子 Lowering 逻辑，非我们追求的 Low-level Launch 抽象。 |

---

## 5. 待探索的核心课题（Open Questions）

以下是项目中需要深入权衡与探索的关键方向，供后续设计与原型验证：

### 课题 A：Launch IR 的表达形态与基础设施选型
* **文本与二进制格式**：IR 是基于 MLIR Dialect 演进，还是基于 Rust/C++ 定义独立的 AST / Bytecode 格式？
* **原型验证工具链**：如何快速搭建物料以验证 IR 的表达能力？（如评估基于 Python + `xDSL` 构建 Prototype Dialect 的可行性与开发效率）。
* **静态标量计算系统**：IR 内需要支持多大粒度的标量表达式（如用于动态计算 Grid/Block Size、Dynamic Shared Memory、Offset 等）？

### 课题 B：Runtime 边界与 Serving State 交互模型
* State 管理器（如 Paged KV Cache Manager）如何与静态的 Function Boundary 优雅解耦？
* 函数内由 `state<T>` 展开为具体硬件 Buffer 指针/偏移量（`CUdeviceptr`）的机制应如何设计？

### 課題 C：内存规划与生命周期（Memory Lifetime & Planning）
* 拓扑图内部的临时中间变量（`%tmp` / Intermediate Buffers）应该在 Compile 阶段做静态 Workspace 规划，还是由 Runtime 做动态/虚拟内存分配？
* 跨 Kernel 依赖与 Stream / Event 同步机制应如何在 IR 中显式表达或隐式推导？

### 课题 D：前端 Producer 路径探索
* 验证用例的端到端生成路径：初期如何便捷地提取或构造一份真实的单层/单步 Forward（如基于 TorchDynamo、AOTInductor 生成产物或手动拼装），用于跑通 Runtime 最小闭环？
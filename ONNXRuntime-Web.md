## ONNXRuntime-Web - webgpu EP

当使用 onnxruntime-web 进行 WebGPU 加速时，底层仍然会调用由 ONNXRuntime C++ 编译而来的 WebAssembly (WASM) 文件。然而，onnxruntime-web 的 WebGPU 执行提供器（Execution Provider，EP）是专门为 Web 环境设计的，并且它将尝试在 WebGPU 上执行尽可能多的算子（operator）。

### onnxruntime-web 的实现逻辑
- 模型加载和初始化：

模型加载和初始化的主要逻辑通过 WebAssembly (WASM) 实现。这部分代码是从 C++ 编译到 WASM 的，确保在浏览器中运行时的高性能和跨平台兼容性。

- Execution Provider 初始化：

  onnxruntime-web 提供的 Execution Provider，如 WebGPU 和 WebNN，在初始化时会检查浏览器是否支持相应的 API。
如果浏览器支持 WebGPU 或 WebNN，这些执行提供器会被注册并初始化。

- 算子分配：

  对于模型中的每一个算子，onnxruntime-web 会检查该算子是否可以在 WebGPU 上执行。如果可以，它会使用 WebGPU 执行，否则回退到 WebAssembly 执行。

- 模型推理：

  当模型执行推理时，主要的调度和算子执行逻辑在 WASM 中进行。WASM 引擎负责将计算任务分配给相应的 Execution Provider。如果选择了 WebGPU 执行提供器，模型中能够被加速的部分将由 WebGPU 处理，无法加速的部分仍然由 WebAssembly 处理。

- 与硬件交互：

  - WebGPU Execution Provider：当需要执行硬件加速时，WASM 引擎会调用 JavaScript 层的 WebGPU API。这些 API 通过浏览器与底层 GPU 交互，执行具体的计算任务。
  - WebNN Execution Provider：同样地，WASM 引擎会调用 JavaScript 层的 WebNN API，这些 API 通过浏览器与底层硬件或加速库交互，执行神经网络运算。

### 主要部分编译为 WASM，与硬件交互时调用 Web API
- WASM 负责核心逻辑：

	ONNX Runtime 的核心逻辑，包括模型解析、调度、内存管理和大部分算子实现，都是通过 WASM 实现的。这些核心逻辑在浏览器中高效运行，确保了模型推理的性能和兼容性。

- JavaScript 调用 Web API：

	当需要执行硬件加速时，WASM 模块会通过 JavaScript 接口调用浏览器提供的 Web API（如 WebGPU 和 WebNN）。
JavaScript 层的这些 API 调用实际上是桥接 WASM 与硬件的纽带，它们将具体的计算任务提交给浏览器，并通过浏览器与硬件驱动进行交互。

```
+-------------------------+
|  JavaScript Layer       |
|  - Web API (WebGPU,     |
|    WebNN)               |
+------------+------------+
             |
+------------v------------+
|  WebAssembly (WASM)     |
|  - ONNX Runtime Core    |
|  - Execution Provider   |
+-------------------------+
             |
+------------v------------+
|  Browser                 |
|  - GPU / Hardware        |
+-------------------------+
```


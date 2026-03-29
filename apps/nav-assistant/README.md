# NavAssistantApp

`NavAssistantApp` 是 `OpenStreetMcp` monorepo 里的 macOS SwiftUI 客户端，目录位于 `apps/nav-assistant/`。
- 采集语音或文本导航需求
- 获取当前定位
- 把自然语言请求桥接到 `OpenStreetMcp` 的 Python MCP 能力
- 展示候选目的地、所选结果和路线摘要

## 目录结构

```text
apps/nav-assistant/
  Package.swift
  README.md
  Scripts/
    run-app-bundle.sh
  Sources/
    NavAssistantApp/
      ContentView.swift
      CandidateSelectionViewModel.swift
      AddressCandidateService.swift
      MapNavigationViewModel.swift
      ...
```

## 运行方式

```bash
swift build --package-path apps/nav-assistant
./apps/nav-assistant/Scripts/run-app-bundle.sh
```

首次使用时请允许：
- 麦克风
- 语音识别
- 定位

## 当前调用链路

1. 用户在 `ContentView` 中录音或直接编辑文本。
2. 用户点击“确认文本并规划”。
3. `CandidateSelectionViewModel` 检查文本和当前位置，组织规划参数。
4. `MCPAddressCandidateService` 启动 Python 桥接脚本。
5. Python 桥接脚本连接 `osm-mcp-server`。
6. MCP server 调用 `recommend_destination_and_waypoints` 生成候选目的地。
7. Python 脚本根据当前选中项再调用一次 `get_coarse_waypoints`。
8. Swift 侧解码 JSON，更新候选列表和地图状态。
9. `MapNavigationViewModel` 将所选目的地和路线回填到地图面板。

## Python 桥接解析

Swift 侧当前通过 `AddressCandidateService.swift` 调用：

- `../../examples/macos_prompt_recommendation_client.py`
- 或 `examples/macos_prompt_recommendation_client.py`
- 或环境变量 `MCP_BRIDGE_SCRIPT_PATH` 指定的桥接脚本

Python 可执行文件按以下优先级查找：

- `PYTHON_BIN`
- `/opt/anaconda3/envs/openstreet/bin/python3`
- `/opt/anaconda3/envs/openstreet/bin/python`
- 系统 `python3`

`uv` 也支持通过 `UV_BIN` 覆盖。

## 主要模块

- `ContentView.swift`：主界面，承接录音、文本确认、继续对话和候选选择。
- `SpeechRecognitionManager.swift`：语音识别权限请求和录音转写。
- `CandidateSelectionViewModel.swift`：管理规划请求、继续对话和候选状态。
- `AddressCandidateService.swift`：Swift -> Python -> MCP 的桥接层。
- `MapNavigationViewModel.swift`：当前位置、目标点、路线与地图摘要。
- `MapPanelView.swift`：地图渲染和镜头更新。
- `Models.swift`：Swift 侧 JSON 模型。

## 当前行为说明

- 规划请求默认由 app 传入 `mode = "car"`、`maxCandidates = 5`、`radius = 3000`。
- 当 Python / MCP 调用失败时，会回落到 `MockAddressCandidateService`。
- 候选被选中后，会再次触发桥接脚本，请求对应候选的路线结果。


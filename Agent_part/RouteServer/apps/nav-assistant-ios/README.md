# NavAssistantiOS

`NavAssistantiOS` 是 `OpenStreetMcp` 的 iOS SwiftUI 客户端 Phase 1 原型。

当前目标：

- 通过 HTTP API 调用 `OpenStreetMcp` 后端
- 支持定位、语音输入、目的地推荐、候选选择、路线展示、语音播报
- 支持 `route session`、WebSocket 刷新、添加途经点、waypoint 推送

## 目录

```text
apps/nav-assistant-ios/
  NavAssistantiOS.xcodeproj
  NavAssistantiOS/
    App/
    Features/
    Models/
    Services/
    ViewModels/
    Resources/
```

## 后端

默认读取 `Info.plist` 中的 `APIBaseURL`：

```text
http://127.0.0.1:8080
```

若在真机上运行，需要改成可被手机访问的后端地址。

`Info.plist` 已加入：

- `NSAllowsLocalNetworking`
- `NSLocalNetworkUsageDescription`

用于局域网开发时访问 Mac 上的后端和机器人服务。

## 启动后端 HTTP API

```bash
osm-http-api --host 0.0.0.0 --port 8080
```

## iPhone 部署思路

### 方案 1：同一台 Mac + iPhone 同一局域网

1. 在 Mac 上启动后端：

```bash
osm-http-api --host 0.0.0.0 --port 8080
```

2. 找到 Mac 的局域网 IP，例如 `192.168.1.20`
3. 把 `NavAssistantiOS/Resources/Info.plist` 里的 `APIBaseURL` 改成：

```text
http://192.168.1.20:8080
```

4. 确保：
   - iPhone 和 Mac 在同一 Wi-Fi
   - macOS 防火墙允许 8080 入站
   - 若要推 waypoint，机器人地址和端口也对 iPhone 可达

### 方案 2：远程服务器

推荐把 Python 后端部署到一台固定服务器，然后：

- HTTP 走 `https://`
- WebSocket 走 `wss://`
- 反向代理建议用 `Caddy` 或 `Nginx`
- `APIBaseURL` 改成公网域名，例如：

```text
https://nav-api.example.com
```

生产环境不建议继续用明文 `http://`。

### Xcode 真机安装

1. 打开 `apps/nav-assistant-ios/NavAssistantiOS.xcodeproj`
2. 修改 Bundle Identifier
3. 选择你的 Apple Team
4. 连接 iPhone
5. 选中真机设备后直接运行

如果真机请求不到后端，优先检查：

- `APIBaseURL` 是否仍然指向 `127.0.0.1`
- Mac 与手机是否同网段
- 端口 `8080` 是否被防火墙拦截
- 后端是否绑定在 `0.0.0.0`

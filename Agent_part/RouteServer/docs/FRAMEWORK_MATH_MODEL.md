# UrbanNav Agent 框架数学模型

> 历史设计记录：本文保留早期 RouteContext、Visual Prompt 与 Phase1 的数学推导。Visual Prompt 生成链及对应 CLI 已退役，旧命令仅适用于隔离 checkout `4879be4`。当前 Harness/System1/System2 的实现边界以 [architecture.md](architecture.md) 为准。

本文档描述当前 Agent/RouteServer/SideWalkNav 框架的数学抽象、模块边界和数据流。它重点回答三个问题：

1. 当前系统如何从自然语言、位置、路线和 FPV 图像生成导航 visual prompt。
2. SideWalk Phase1 如何从 FPV/BEV/pose 生成可训练轨迹监督。
3. Google Street View / 百度全景静态图应该接入到哪一层，以及当前仓库缺什么。

## 1. 总体链路

在线导航链路可以写成：

\[
(u_t, x_t, \psi_t, I_t, \mathcal S_{t-1})
\rightarrow
\mathcal S_t
\rightarrow
\Gamma_t
\rightarrow
r_t
\rightarrow
b_t
\rightarrow
\hat p_t
\rightarrow
O_t
\]

其中：

- \(u_t\)：用户自然语言指令。
- \(x_t=(\phi_t,\lambda_t)\)：当前位置，纬度 \(\phi_t\)、经度 \(\lambda_t\)。
- \(\psi_t\)：当前朝向 heading。
- \(I_t\)：当前第一视角图像，可以来自真实相机、MCAP 抽帧、SideWalk clip、Street View/Panorama 静态图。
- \(\mathcal S_t\)：有状态 route session。
- \(\Gamma_t\)：全局路线计划。
- \(r_t\)：局部 RouteContext。
- \(b_t\)：局部 BEV/拓扑上下文。
- \(\hat p_t\)：结构化 visual prompt 决策。
- \(O_t\)：最终 overlay，例如 SVG 轨迹箭头。

工程对应：

```text
Map/Route provider
-> Route Session
-> RouteContext
-> LocalBEVContext
-> VisualPromptAgent
-> overlay/json manifest
```

## 2. 地图与图像 provider 边界

当前仓库已有地图 provider：

\[
\mathcal M \in \{\text{osm}, \text{amap}, \text{baidu}, \text{google}\}
\]

它们通过统一接口提供：

\[
\mathrm{Geocode},\ \mathrm{ReverseGeocode},\ \mathrm{SearchPOI},\ \mathrm{Route}
\]

也就是：

\[
\mathcal M(u_t, x_t) \rightarrow C_t
\]

\[
\mathcal M(o_t,d_t,m) \rightarrow \Gamma_t
\]

其中 \(C_t\) 是 POI 候选，\(o_t\) 是起点，\(d_t\) 是目的地，\(m\) 是导航模式。

但 Street View / Panorama 静态图不是当前 `MapProviderClient` 的一部分。它应该被建模为独立图像 provider：

\[
\mathcal V \in \{\text{camera}, \text{mcap}, \text{sidewalk}, \text{google\_streetview}, \text{baidu\_panorama}\}
\]

\[
I_t, \kappa_t =
\mathcal V(x_t,\psi_t,\eta_t)
\]

其中：

- \(I_t\)：下载或读取到的 FPV 图像。
- \(\kappa_t\)：图像元数据，例如 `provider`、`pano_id`、`heading`、`pitch`、`fov`、`image_size`、坐标系和请求 URL。
- \(\eta_t\)：图像请求参数，例如 FOV、pitch、宽高、最近 pano 搜索半径。

因此当前结论是：

```text
仓库支持 Google/Baidu 地图路线能力；
仓库已有 Google Street View / 百度全景静态图的最小下载脚本；
仓库尚未接入 Google Maps JS StreetViewService 的 pano graph links；
接入时应该新增 image provider，而不是塞进现有 route provider。
```

## 3. Street View / Panorama 接入模型

### 3.1 请求参数

对一条轨迹采样点：

\[
\tau_i=(x_i,\psi_i,t_i)
\]

街景图请求可写成：

\[
q_i =
(\phi_i,\lambda_i,\psi_i,\theta_i,\alpha_i,W,H)
\]

其中：

- \((\phi_i,\lambda_i)\)：请求位置。
- \(\psi_i\)：heading / yaw。
- \(\theta_i\)：pitch。
- \(\alpha_i\)：horizontal FOV。
- \(W,H\)：输出图像尺寸。

Google Street View Static API 形式：

\[
I_i =
\mathrm{GoogleStreetView}(location=(\phi_i,\lambda_i), heading=\psi_i, pitch=\theta_i, fov=\alpha_i, size=(W,H))
\]

百度全景静态图形式：

\[
I_i =
\mathrm{BaiduPanorama}(location=(\lambda_i,\phi_i), heading=\psi_i, pitch=\theta_i, fov=\alpha_i, size=(W,H))
\]

注意经纬度顺序不同：

```text
Google: location=lat,lng
Baidu:  location=lng,lat
```

百度地图常见输入坐标系是 BD-09 或百度接口约定坐标；AMap 常见是 GCJ-02；GPS/OSM/Google 常见是 WGS-84。跨 provider 组合时必须显式记录坐标系：

\[
x^{provider} =
T_{\mathrm{WGS84}\rightarrow provider}(x^{wgs84})
\]

当前仓库已有路线和 BEV 逻辑，但没有统一坐标系转换层用于 panorama 下载；新增下载器时必须把 `coordinate_system` 写入 manifest。

### 3.2 可复现下载 manifest

为了让数据可审计，每一张下载图都应该有一条 manifest：

```json
{
  "frame_id": "000123",
  "provider": "google_streetview",
  "coordinate_system": "WGS84",
  "lat": 30.2645,
  "lng": 120.1180,
  "pano_id": null,
  "heading_deg": 35.0,
  "pitch_deg": 0.0,
  "fov_deg": 90.0,
  "width": 600,
  "height": 300,
  "image_path": "images/000123.jpg",
  "metadata_path": "metadata/000123.json",
  "request_url_without_key": "https://maps.googleapis.com/maps/api/streetview?...",
  "status": "ok"
}
```

早期下载器为已退役的 visual-prompt 流水线使用以下文件布局；当前共享读取器为 `mcp_server.visualization.frame_io`：

```text
<WORK_DIR>/
  images/
    000001.jpg
    000002.jpg
  trajectory.json
  frames_manifest.json
  streetview_manifest.json
```

其中 `trajectory.json` 至少需要：

```json
{
  "lat": 30.2645,
  "lng": 120.1180,
  "heading_deg": 35.0,
  "image_path": "/abs/path/images/000001.jpg"
}
```

这样下载的街景图可以直接作为：

\[
I_t \leftarrow \mathcal V_{\text{streetview}}(x_t,\psi_t)
\]

进入现有 Visual Prompt pipeline。

## 4. Route Session 数学模型

Route session 是有状态导航任务：

\[
\mathcal S_t=(o_t,d_t,V_t,m,\Gamma_t,h_t)
\]

其中：

- \(o_t\)：当前起点或当前位置。
- \(d_t\)：最终目的地。
- \(V_t=[v_1,\dots,v_n]\)：动态途经点。
- \(m \in \{\text{foot},\text{bike},\text{car}\}\)：导航模式。
- \(\Gamma_t\)：缓存路线。
- \(h_t\)：metadata。

创建：

\[
\mathcal S_0=\mathrm{CreateSession}(o_0,d_0,V_0,m)
\]

动态追加途经点：

\[
v_{new}=\mathrm{ResolveInstruction}(u_t,x_t,m)
\]

\[
V_{t+1}=\mathrm{Insert}(V_t,v_{new})
\]

\[
\Gamma_{t+1}=\mathrm{Route}(o_t,V_{t+1},d_t,m)
\]

默认读取 session 时复用缓存：

\[
\mathrm{GetSession}(\mathcal S_t)=(\mathcal S_t,\Gamma_t)
\]

只有 `force_replan=true` 时：

\[
\Gamma_t=\mathrm{Route}(o_t,V_t,d_t,m)
\]

## 5. 自然语言目的地与 itinerary

单目标推荐：

\[
d_t^*=\mathrm{Recommend}(u_t,x_t,m,\rho,k)
\]

其中 \(\rho\) 是搜索半径，\(k\) 是候选数量。

多目标需求先拆成 goals：

\[
G_t=\mathrm{Decompose}(u_t)=[g_1,g_2,\dots,g_N]
\]

每个 goal 检索 POI：

\[
C_i=\mathrm{RetrievePOI}(g_i,x_i,m,\rho_i,k)
\]

再由 LLM judge 选择：

\[
z_i^*=\mathrm{Judge}(g_i,C_i,x_i,z_{<i})
\]

组合为：

\[
V_t=[z_1^*,\dots,z_{N-1}^*],\quad d_t=z_N^*
\]

\[
\Gamma_t=\mathrm{Route}(o_t,V_t,d_t,m)
\]

当前实现是逐 goal 贪心选择。完整全局优化可以写成：

\[
(V_t^*,d_t^*)=
\arg\max_{\mathcal I}
\mathrm{Score}(\mathcal I \mid G_t,x_t,m)
\]

其中 score 可包含：

- goal satisfaction
- total walking time
- detour penalty
- fatigue penalty
- spatial compactness
- opening hours / availability

## 6. RouteContext

全局路线：

\[
\Gamma_t=(P_t,W_t,S_t)
\]

其中：

- \(P_t=[p_0,\dots,p_L]\)：路线 polyline。
- \(W_t\)：coarse waypoints。
- \(S_t\)：route summary。

当前 pose 投影到路线：

\[
\hat q_t=\Pi_{P_t}(x_t)
\]

匹配输出：

\[
(i_t,\alpha_t,\hat q_t)=\mathrm{Match}(x_t,P_t)
\]

其中 \(i_t\) 是 segment index，\(\alpha_t\in[0,1]\) 是 segment 内插值位置。

横向偏差：

\[
e_t=\|x_t-\hat q_t\|_{\perp}
\]

路线切向角：

\[
\theta_t=\mathrm{bearing}(p_{i_t},p_{i_t+1})
\]

朝向误差：

\[
\Delta\psi_t=\mathrm{wrap}_{180}(\psi_t-\theta_t)
\]

路线进度：

\[
s_t=\mathrm{arcpos}(\hat q_t)
\]

到下一路口或动作点距离：

\[
d^{turn}_t=s(w_{next})-s_t
\]

输出 `RouteContext`：

```text
pose_match
progress
local_semantics
geometry
```

它是给 LLM/VLM 的紧凑导航状态，而不是原始整条路线。

## 7. LocalBEVContext

LocalBEVContext 是以 ego 为中心的局部拓扑表示：

\[
b_t=\mathrm{LocalBEV}(r_t,\Gamma_t)
\]

输出：

\[
b_t=(p^{ego}_t,P^{local}_t,L^{local}_t,C_t,T_t,B_t)
\]

其中：

- \(p^{ego}_t\)：ego 在 BEV patch 中的位置。
- \(P^{local}_t\)：局部路线 polyline。
- \(L^{local}_t\)：当前局部 segment。
- \(C_t\)：connector geometry。
- \(T_t\)：next turn point。
- \(B_t\)：BEV bbox。

BEV 坐标一般定义为：

\[
x_{right} = R_{\psi_t}^{-1}(x-x_t)_x
\]

\[
y_{front} = R_{\psi_t}^{-1}(x-x_t)_y
\]

其中 \(R_{\psi_t}\) 是 heading 对应的二维旋转。

BEV 是 route-side prior / debugger state。当前在线主链路不强制要求：

```text
BEV proposal -> PoV candidate -> arbiter
```

而是允许 BEV 作为 optional context。

## 8. Visual Prompt 决策

给定 FPV 图像和导航上下文：

\[
\hat p_t=
A_{\theta}^{zero-shot}(I_t,r_t,m,\mathcal C_t)
\]

其中：

- \(A_{\theta}^{zero-shot}\)：VLM/LLM visual prompt agent。
- \(\mathcal C_t\)：可选候选集，例如 BEV prior、PoV weak projection、segmentation summary、SoM regions。

结构化输出：

```json
{
  "type": "arrow_path",
  "path_uv": [[160, 175], [190, 135], [220, 90]],
  "instruction": "Walk along the sidewalk opening.",
  "confidence": 0.72,
  "reason": "The route intent is forward and the visible sidewalk is the safest opening."
}
```

可行性约束：

\[
\hat p_t \in \mathcal F(I_t,m)
\]

对 `mode=foot`：

\[
\mathcal F(I_t,\mathrm{foot})=
\{\text{sidewalk, pedestrian path, crosswalk, open plaza}\}
\setminus
\{\text{vehicle lane, obstacle, wall, fence}\}
\]

最终 overlay：

\[
O_t=\mathrm{Overlay}(I_t,\hat p_t)
\]

## 9. SideWalk Phase1 几何监督

Phase1 的目标是训练：

\[
F_{\theta}(I_t,B_t,Q_t) \rightarrow Y_t
\]

其中：

- \(I_t\)：FPV 图像。
- \(B_t\)：BEV 图像。
- \(Q_t\)：文本化 QA / task prompt。
- \(Y_t\)：结构化 CoT 和 `<finish>` 中的轨迹。

### 9.1 Odom 到 robot frame

给当前帧位姿 \(T_t\) 和未来帧 \(T_{t+\Delta}\)，先计算世界坐标差：

\[
\Delta x=x_{t+\Delta}-x_t,\quad
\Delta y=y_{t+\Delta}-y_t
\]

按当前 yaw 变到 robot frame：

\[
x_{robot}=\Delta x\cos\psi_t+\Delta y\sin\psi_t
\]

\[
y_{robot}=-\Delta x\sin\psi_t+\Delta y\cos\psi_t
\]

约定：

```text
x_robot: forward
y_robot: left
```

### 9.2 Robot frame 到 camera frame

当前相机模型：

\[
z_{cam}=x_{robot}
\]

\[
x_{cam}=-y_{robot}
\]

\[
y_{cam}=-h_{cam}
\]

其中 \(h_{cam}\) 是相机高度。

### 9.3 Camera frame 到 FPV 像素

针孔投影：

\[
u=f_x\frac{x_{cam}}{z_{cam}}+c_x
\]

\[
v=c_y-f_y\frac{y_{cam}}{z_{cam}}
\]

归一化：

\[
\tilde u=\frac{u}{W},\quad \tilde v=\frac{v}{H}
\]

Phase1 输出轨迹：

\[
Y_t^{traj}=[(\tilde u_0,\tilde v_0),\dots,(\tilde u_{N-1},\tilde v_{N-1})]
\]

第一个点固定为 bottom-center anchor：

\[
(\tilde u_0,\tilde v_0)=(0.5,0.9)
\]

### 9.4 时间条件轨迹

动态环境下，future executed path 不等于当前帧静态 free-space。Phase1 同时写：

\[
Y_t^{fpv\_t}=[(\tilde u_i,\tilde v_i,\Delta t_i)]_{i=0}^{N-1}
\]

\[
Y_t^{fpv\_tv}=[(\tilde u_i,\tilde v_i,\Delta t_i,s_i)]_{i=0}^{N-1}
\]

\[
Y_t^{ego\_t}=[(x^{right}_i,y^{front}_i,\Delta t_i,s_i)]_{i=0}^{N-1}
\]

这些字段分别对应：

```text
pixel_trajectory
trajectory_fpv_t
trajectory_fpv_tv
trajectory_ego_t
```

没有 dynamic mask/tracking 时：

```text
dynamic_obstacle_audit.status = not_evaluated_no_dynamic_masks
```

不能自动把 future path 改写成避障路径。

## 10. CoT Teacher 与 Validator

LLM teacher 生成：

```xml
<think>
<Scene>{...}</Scene>
<FPVScene>{...}</FPVScene>
<BEVContext>{...}</BEVContext>
<CrossView>{...}</CrossView>
<Plan>{...}</Plan>
</think>
<finish>{"trajectory":[[x,y],...],"intent":"...","route_theta_deg":...}</finish>
```

但几何真值不由 LLM 决定：

\[
Y_t^{cot}=
\mathrm{Teacher}(I_t,B_t,Q_t,E_t)
\]

\[
\mathrm{Validator}(Y_t^{cot},Y_t^{geom}) \in \{0,1\}
\]

其中 \(E_t\) 是 geometry evidence。Validator 强制检查：

- `<finish>.trajectory` 与几何 GT 一致。
- `intent` 一致。
- `route_theta_deg` 在容差内。
- CoT section JSON 可解析。
- FPV/BEV 文件存在。

只有通过 validator 的样本进入 archive：

\[
\mathcal D_{train}=
\{(I_t,B_t,Q_t,Y_t^{cot}) \mid \mathrm{Validator}=1\}
\]

## 11. Baseline 与指标

Agent baseline 使用同样的输入，让模型真实预测：

\[
\hat Y_t=F_{\theta}(I_t,B_t,Q_t)
\]

解析 `<finish>.trajectory` 得到：

\[
\hat P=[\hat p_0,\dots,\hat p_{M-1}]
\]

GT：

\[
P^*=[p^*_0,\dots,p^*_{N-1}]
\]

离散 ADE/FDE：

\[
ADE=\frac{1}{K}\sum_{i=0}^{K-1}\|\hat p_i-p_i^*\|_2
\]

\[
FDE=\|\hat p_{K-1}-p^*_{K-1}\|_2
\]

推荐曲线级指标，把点列按弦长参数化：

\[
C(u),\quad u\in[0,1]
\]

\[
cADE=\int_0^1\|\hat C(u)-C^*(u)\|_2du
\]

\[
cFDE=\|\hat C(1)-C^*(1)\|_2
\]

\[
cMDE=\max_u\|\hat C(u)-C^*(u)\|_2
\]

横向误差：

\[
E_{lat}=\int_0^1 |(\hat C(u)-C^*(u))\cdot n(u)|du
\]

TubeIoU：

\[
\Omega_r(C)=\{q\mid \min_u\|q-C(u)\|_2\le r\}
\]

\[
TubeIoU=
\frac{|\Omega_r(\hat C)\cap\Omega_r(C^*)|}
{|\Omega_r(\hat C)\cup\Omega_r(C^*)|}
\]

## 12. 当前是否能获取街景图

当前仓库状态：

```text
可以：
- 用 MAP_PROVIDER=google 做 geocode / places / routes。
- 用 MAP_PROVIDER=baidu 做 geocode / POI / directionlite routes。
- 消费已有 FPV images + trajectory.json 运行 visual prompt pipeline。
- 用 tools/dataset_processing/fetch_streetview_images.py 调用 Google Street View Static API。
- 用 tools/dataset_processing/fetch_streetview_images.py 调用 Baidu panorama/v2 静态图 API。
- 把下载结果写成 images/trajectory.json/frames_manifest.json/streetview_manifest.json。

尚未实现：
- 根据 route polyline 自动采样街景点。
- 通过 Google Maps JS StreetViewService 拉取 pano graph links。
- 统一处理 WGS84/GCJ-02/BD-09 坐标转换和 provider metadata。
```

当前新增入口：

```text
tools/dataset_processing/fetch_streetview_images.py
```

该脚本支持：

```text
--provider google
--provider baidu
--provider auto  # 读取 MAP_PROVIDER，但只接受 google/baidu
--trajectory-json
--locations-csv
--lat/--lng
--grid-x/--grid-y
```

后续如果要把下载能力服务化，可抽成接口：

```python
class FPVImageProvider:
    async def fetch_frame(
        self,
        *,
        lat: float,
        lon: float,
        heading_deg: float,
        pitch_deg: float,
        fov_deg: float,
        width: int,
        height: int,
    ) -> dict:
        ...
```

输出必须兼容现有 pipeline：

```text
images/
trajectory.json
frames_manifest.json
streetview_manifest.json
```

这样 Street View/Panorama 只替换 \(I_t\) 的来源，不改变 RouteContext、BEV、VisualPromptAgent 和 Phase1 训练框架。

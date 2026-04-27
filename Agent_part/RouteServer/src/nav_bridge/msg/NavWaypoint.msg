# nav_waypoint.msg - 导航航点消息
std_msgs/Header header

# GPS坐标 (WGS84)
float64 latitude
float64 longitude

# 航点信息
int32 waypoint_index           # 全局航点序号
bool is_terminal               # 是否终点
bool is_via_stop               # 是否途经点
string action                  # 动作类型: go_straight / turn_left / turn_right / arrive 等
string instruction             # 中文导航指令
float32 distance_m             # 本段距离(m)
float32 duration_s             # 本段预估时间(s)

# 所属路段信息
int32 leg_index                # 路段索引
string leg_origin              # 起点 "lon,lat"
string leg_destination         # 终点 "lon,lat"

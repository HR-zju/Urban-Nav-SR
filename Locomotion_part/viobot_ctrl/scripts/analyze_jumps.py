import os
import argparse
import numpy as np
from mcap_ros2.reader import read_ros2_messages


def discover_mcap_files(path):
    """Discover .mcap files from a path (file or directory).
    Excludes files ending with '_repaired.mcap'.
    """
    if os.path.isfile(path):
        if path.endswith('.mcap'):
            return [path]
        else:
            print(f"[WARN] Not an mcap file: {path}")
            return []
    elif os.path.isdir(path):
        files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith('.mcap') and not f.endswith('_repaired.mcap')
        ])
        return files
    else:
        print(f"[ERROR] Path does not exist: {path}")
        return []


def analyze_single(bag_path):
    """Analyze a single mcap bag file for jumps."""
    print(f"\n{'=' * 87}")
    print(f"  Analyzing: {bag_path}")
    print(f"{'=' * 87}")

    data = {
        '/baton/gnss/spp_fix_raw': [],
        '/baton/gnss/spp_fix': [],
        '/baton/stereo3/odometry': []
    }

    for msg in read_ros2_messages(bag_path):
        topic = msg.channel.topic
        if topic in data:
            t_sec = msg.publish_time.timestamp()
            ros_msg = msg.ros_msg
            if topic in ['/baton/gnss/spp_fix_raw', '/baton/gnss/spp_fix']:
                data[topic].append({
                    't': t_sec,
                    'lat': ros_msg.latitude,
                    'lon': ros_msg.longitude,
                    'alt': ros_msg.altitude,
                    'status': ros_msg.status.status
                })
            elif topic == '/baton/stereo3/odometry':
                pos = ros_msg.pose.pose.position
                data[topic].append({
                    't': t_sec,
                    'x': pos.x,
                    'y': pos.y,
                    'z': pos.z
                })

    print("=======================================================================================")
    print("                         SUMMARY OF JUMPS IN MCAP BAG DATA                             ")
    print("=======================================================================================")

    results = {}

    for topic, records in data.items():
        if len(records) < 2:
            if len(records) == 0:
                print(f"\nTopic: {topic}  -- NO DATA --")
            continue

        t = np.array([r['t'] for r in records])
        dt = np.diff(t)

        if 'lat' in records[0]:
            lat = np.array([r['lat'] for r in records])
            lon = np.array([r['lon'] for r in records])
            alt = np.array([r['alt'] for r in records])

            mean_lat_rad = np.radians(np.mean(lat))
            R_earth = 6378137.0

            dx = np.radians(np.diff(lon)) * R_earth * np.cos(mean_lat_rad)
            dy = np.radians(np.diff(lat)) * R_earth
            dz = np.diff(alt)

            dist_2d = np.sqrt(dx**2 + dy**2)
            dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            speed_2d = np.where(dt > 0, dist_2d / dt, 0)
        else:
            x = np.array([r['x'] for r in records])
            y = np.array([r['y'] for r in records])
            z = np.array([r['z'] for r in records])

            dx = np.diff(x)
            dy = np.diff(y)
            dz = np.diff(z)

            dist_2d = np.sqrt(dx**2 + dy**2)
            dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            speed_2d = np.where(dt > 0, dist_2d / dt, 0)

        results[topic] = {
            'count': len(records),
            'duration': t[-1] - t[0],
            'dt_mean': np.mean(dt),
            'dist2d_mean': np.mean(dist_2d),
            'dist2d_max': np.max(dist_2d),
            'dist2d_95': np.percentile(dist_2d, 95),
            'dist2d_99': np.percentile(dist_2d, 99),
            'alt_max': np.max(np.abs(dz)),
            'speed_max': np.max(speed_2d),
            'jumps_05m': np.sum(dist_2d > 0.5),
            'jumps_1m': np.sum(dist_2d > 1.0),
            'jumps_2m': np.sum(dist_2d > 2.0),
            'jumps_5m': np.sum(dist_2d > 5.0),
            't': t,
            'dist_2d': dist_2d,
            'dz': dz,
            'speed_2d': speed_2d
        }

        r = results[topic]
        print(f"\nTopic: {topic}")
        print(f"  Messages: {r['count']} | Duration: {r['duration']:.2f}s | Avg Freq: {1.0/r['dt_mean']:.1f} Hz")
        print(f"  2D Step Dist (m)  => Mean: {r['dist2d_mean']:.4f} | Max: {r['dist2d_max']:.4f} | 95%: {r['dist2d_95']:.4f} | 99%: {r['dist2d_99']:.4f}")
        print(f"  Alt Step Dist (m) => Max: {r['alt_max']:.4f}")
        print(f"  Max Speed (m/s)   => {r['speed_max']:.2f}")
        print(f"  Jump Counts       => >0.5m: {r['jumps_05m']} | >1.0m: {r['jumps_1m']} | >2.0m: {r['jumps_2m']} | >5.0m: {r['jumps_5m']}")

    print("\n=======================================================================================")
    print("                     DETAILED JUMP EVENTS (> 2.0 meters in 2D)                         ")
    print("=======================================================================================")

    for topic, r in results.items():
        print(f"\n--- {topic} (Jumps > 2.0m) ---")
        idx_jumps = np.where(r['dist_2d'] > 2.0)[0]
        if len(idx_jumps) == 0:
            print("  None")
        else:
            for idx in idx_jumps:
                print(f"  [msg {idx:3d} -> {idx+1:3d}] t={r['t'][idx+1]:.3f}s (dt={r['t'][idx+1]-r['t'][idx]:.3f}s): 2D_jump={r['dist_2d'][idx]:.3f}m | Alt_jump={r['dz'][idx]:.3f}m | speed={r['speed_2d'][idx]:.1f} m/s")

    print("\n=======================================================================================")
    print("               COMPARISON & CORRELATION: odometry vs spp_fix vs spp_fix_raw            ")
    print("=======================================================================================")
    # Let's check specifically around the jump times of spp_fix
    # Guard against missing topics
    required_topics = ['/baton/gnss/spp_fix', '/baton/gnss/spp_fix_raw', '/baton/stereo3/odometry']
    missing = [t for t in required_topics if t not in results]
    if missing:
        print(f"\n  [SKIP] Missing topics for comparison: {', '.join(missing)}")
    else:
        r_fix = results['/baton/gnss/spp_fix']
        r_raw = results['/baton/gnss/spp_fix_raw']
        r_odom = results['/baton/stereo3/odometry']

        # Check all jumps > 0.5m in spp_fix
        fix_jumps = np.where(r_fix['dist_2d'] > 0.5)[0]
        print(f"\nAnalyzing all {len(fix_jumps)} events in /baton/gnss/spp_fix where step > 0.5m:")
        for idx in fix_jumps:
            t_jump = r_fix['t'][idx+1]
            dist_fix = r_fix['dist_2d'][idx]

            # Find closest time in odom
            odom_idx = np.argmin(np.abs(r_odom['t'][1:] - t_jump))
            dist_odom = r_odom['dist_2d'][odom_idx]
            dt_odom = r_odom['t'][odom_idx+1] - t_jump

            # Find closest time in raw
            raw_idx = np.argmin(np.abs(r_raw['t'][1:] - t_jump))
            dist_raw = r_raw['dist_2d'][raw_idx]
            dt_raw = r_raw['t'][raw_idx+1] - t_jump

            print(f"  t={t_jump:.3f}s | spp_fix jump = {dist_fix:6.3f}m | closest odom jump = {dist_odom:6.3f}m (delta_t={dt_odom:5.3f}s) | closest raw jump = {dist_raw:6.3f}m (delta_t={dt_raw:5.3f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GNSS jumps in MCAP bag files. '
                    'Supports single file or batch directory processing.'
    )
    parser.add_argument(
        'path', nargs='?', default='.',
        help='Path to a single .mcap file or a directory containing .mcap files. '
             'Defaults to current directory. Files ending with _repaired.mcap are excluded.'
    )
    args = parser.parse_args()

    mcap_files = discover_mcap_files(args.path)
    if not mcap_files:
        print("[ERROR] No .mcap files found.")
        return

    print(f"Found {len(mcap_files)} mcap file(s) to analyze:")
    for f in mcap_files:
        print(f"  - {f}")

    for bag_path in mcap_files:
        try:
            analyze_single(bag_path)
        except Exception as e:
            print(f"\n[ERROR] Failed to analyze {bag_path}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()

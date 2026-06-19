""" Automatic waypoint selection """
import numpy as np
import copy

from waypoint_extraction.traj_reconstruction import (
    pos_only_geometric_waypoint_trajectory,
    reconstruct_waypoint_trajectory,
    geometric_waypoint_trajectory,
)
from typing import List, Dict, Tuple, Optional, Union

""" Iterative waypoint selection """
def greedy_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    geometry=True,
    pos_only=False,
):
    # make the last frame a waypoint
    waypoints = [len(actions) - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(len(actions) - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                waypoints.append(i)
                waypoints.append(i + 1)
        waypoints.sort()

    # reconstruct the trajectory, and record the reconstruction error for each state
    for i in range(len(actions)):
        if pos_only or geometry:
            func = (
                pos_only_geometric_waypoint_trajectory
                if pos_only
                else geometric_waypoint_trajectory
            )
            total_traj_err, reconstruction_error = func(
                actions=actions,
                gt_states=gt_states,
                waypoints=waypoints,
                return_list=True,
            )
        else:
            _, reconstruction_error, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions,
                gt_states=gt_states,
                waypoints=waypoints,
                verbose=False,
                initial_state=initial_states[0],
                remove_obj=remove_obj,
            )
        # break if the reconstruction error is below the threshold
        if total_traj_err < err_threshold:
            break
        # add the frame of the highest reconstruction error as a waypoint, excluding frames that are already waypoints
        max_error_frame = np.argmax(reconstruction_error)
        while max_error_frame in waypoints:
            reconstruction_error[max_error_frame] = 0
            max_error_frame = np.argmax(reconstruction_error)
        waypoints.append(max_error_frame)
        waypoints.sort()

    print("=======================================================================")
    print(
        f"Selected {len(waypoints)} waypoints: {waypoints} \t total trajectory error: {total_traj_err:.6f}"
    )
    return waypoints

""" DP waypoint selection """
# use geometric interpretation
def dp_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    pos_only=False,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
        
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    func = (
        pos_only_geometric_waypoint_trajectory
        if pos_only
        else geometric_waypoint_trajectory
    )

    # Check if err_threshold is too small, then return all points as waypoints
    min_error = func(actions, gt_states, list(range(1, num_frames)))
    if err_threshold < min_error:
        print("Error threshold is too small, returning all points as waypoints.")
        return list(range(1, num_frames))

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            total_traj_err = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
            )

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    )
    print(f"waypoint positions: {waypoints}")

    return waypoints


import numpy as np
import copy

def SWR_waypoint_selection(
        env=None,
        actions=None,
        gt_states=None,
        err_threshold=None,
        initial_states=None,
        remove_obj=None,
        pos_only=False,
        geometry=True,
        window_size=None
):
    """
    优化的动态规划关键点选择方法
    参数:
        env: 环境对象(可选)
        actions: 动作序列
        gt_states: 真实状态序列
        err_threshold: 误差阈值
        initial_states: 初始状态(可选)
        remove_obj: 移除对象(可选)
        pos_only: 是否仅使用位置信息
        geometry: 是否使用几何方法
    返回:
        选中的关键点列表
    """

    # 数据预处理
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)

    num_frames = len(actions)

    # 1. 初始关键点选择
    waypoints = set([num_frames - 1])  # 最后一帧必须为关键点

    # 添加夹爪开关状态变化的关键点
    if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                waypoints.add(i)
                # waypoints.add(i + 1)

    # 2. 动态规划表初始化
    dp = [float('inf')] * num_frames
    dp[0] = 0
    prev = [-1] * num_frames

    # 3. 选择重建函数
    if pos_only or geometry:
        recon_func = (pos_only_geometric_waypoint_trajectory if pos_only
                      else geometric_waypoint_trajectory)
    else:
        recon_func = lambda a, g, w: reconstruct_waypoint_trajectory(
            env, a, g, w, False, initial_states[0] if initial_states else None, remove_obj)[1]

    # 4. 动态规划主循环
    for i in range(1, num_frames):
        if i - 1 in waypoints:
            dp[i] = dp[i - 1]
            prev[i] = i - 1
            continue

        for j in range(max(0, i - window_size), i):
            wp = [k - j for k in waypoints if j <= k < i] + [i - j]
            error = recon_func(actions[j:i + 1], gt_states[j:i + 1], wp)

            # 严格实现公式中的软约束和多目标优化
            adjusted_threshold = err_threshold * (1 + 0.05 * (i - j) / window_size)
            if error < adjusted_threshold:
                cost = 1 + 0.3 * error / adjusted_threshold  # λ=0.3
                if dp[j] + cost < dp[i]:
                    dp[i] = dp[j] + cost
                    prev[i] = j

    # 5. 回溯获取关键点
    result = list(waypoints)
    i = num_frames - 1
    while i >= 0 and prev[i] != -1:
        result.append(prev[i])
        i = prev[i]

    # 去重排序
    result = sorted(set(result))

    # 6. 验证最终结果
    final_error = recon_func(actions, gt_states, result)
    print(f"最终选择 {len(result)} 个关键点: {result} \t 总轨迹误差: {final_error:.6f}")

    return result



import importlib
from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import gym
from moviepy.editor import ImageSequenceClip
# import torch as th
    
import sys
import os
from queue import PriorityQueue
import csv

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (assuming the project root is three levels up from the current script directory)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# print(project_root)

# Add the project root directory to sys.path
sys.path.append(project_root)

# # Change the current working directory to the project root
# os.chdir(project_root)
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle
import highway_env.vehicle.controller 

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv

# configuration
speed_max = 30
speed_min = 10
block_length = 10
busyness_sample = 6
action_sample = 5

def busyness_map(env, ego_vehicle = None, actions = None, grid_size= block_length, observation_area=[0, 740], observation_shape=[600, 240]):
    """
    Calculate and plot the busyness map of the environment.
    
    :param env: Environment instance
    :param ego_vehicle: Ego vehicle instance
    :param actions: Actions taken by vehicles
    :param grid_size: Size of the grid blocks [m]
    :param observation_area: Observation area [m]
    :param observation_shape: Shape of the observation area [pixels]
    :return: Normalized busyness level
    """
    num_lanes = 4
    road_length = observation_area[1] - observation_area[0]
    num_grids = int(road_length / grid_size)
    
    busy_level = np.zeros((num_lanes, num_grids))
  
    # Heterogeneous traffic flow + ramp radiation
    busy_level = traffic_flow(env, busy_level, ego_vehicle)
    # Set action weights, unreasonable actions are directly set to 0
    action_weight = calculate_action_weight()
    # print("Action weight matrix:\n", action_weight)
    for vehicle in env.road.vehicles:
        action_select = ["LANE_LEFT", "IDLE", "LANE_RIGHT"]
        if vehicle == ego_vehicle:
            continue

        lane_id = int(vehicle.lane_index[2])
        if actions[vehicle.id] != None:
            ego_action = actions[vehicle.id]
            busy_assignments = {}  # Use a dictionary to store assignments for each block
            x, lane_index = predict_trajectory(vehicle, ego_action, 1 / env.config["policy_frequency"], busyness_sample, eval = False)
            for time, position in enumerate(x):
                ego_position = position
                ego_grid = int((ego_position - observation_area[0]) // grid_size)
                if ego_grid < 0 or ego_grid >= num_grids:
                    continue
                ego_lane = int(lane_index[time])
                # Construct key-value pairs for dictionary storage
                key = (ego_lane, ego_grid)
                value = 0.6*time_decay_linear_function(time, lambda_=0.1)
                if key in busy_assignments:
                    busy_assignments[key].append(value)
                else:
                    busy_assignments[key] = [value]
            for (lane, grid), values in busy_assignments.items():
                busy_level[lane, grid] += np.mean(values)
        
        else:
            # if vehicle not in env.controlled_vehicles:
            action_select = ["IDLE"]
            for ego_index, ego_action in enumerate(action_select):
                busy_assignments = {}  # Use a dictionary to store assignments for each block
                x, lane_index = predict_trajectory(vehicle, ego_action, 1 / env.config["policy_frequency"], busyness_sample, eval=False)
                for time, position in enumerate(x):
                    ego_position = position
                    ego_grid = int((ego_position - observation_area[0]) // grid_size)
                    if ego_grid < 0 or ego_grid >= num_grids:
                        continue
                    ego_lane = int(lane_index[time])
                    # Construct key-value pairs for dictionary storage
                    key = (ego_lane, ego_grid)
                    if vehicle not in env.controlled_vehicles:
                        value = time_decay_linear_function(time, lambda_=0.1)
                    else:
                        # value = action_weight[lane_id][ego_index] * time_decay_linear_function(time, lambda_=0.1)
                        value =  0.6*time_decay_linear_function(time, lambda_=0.1)
                    # value = action_weight[lane_id][ego_index] * time_decay_function(time)

                    # If the block already exists, append the value; otherwise, initialize
                    if key in busy_assignments:
                        busy_assignments[key].append(value)
                    else:
                        busy_assignments[key] = [value]

                # Calculate the mean and update the busyness level
                for (lane, grid), values in busy_assignments.items():
                    busy_level[lane, grid] += np.mean(values)
    
    for i in range(num_grids):
        if i >= (sum(env.ends[0:2])//grid_size + 5) and i < sum(env.ends[0:3])//grid_size:
            busy_level[3, i] += time_decay_linear_function(sum(env.ends[0:3])//grid_size - i, lambda_=0.2)
            # busy_level_normalized[3, i] += time_decay_function(sum(env.ends[0:3])//grid_size - i, lambda_=0.2)
        if i >= sum(env.ends[0:3])//grid_size:
            busy_level[3, i] += 1
    # print(busy_level_normalized)
    # print(busy_level)
    busy_level_normalized = busy_level / np.max(busy_level)
    return busy_level_normalized

def draw_busyness_map(busy_level_normalized, env, ego_vehicle = None, grid_size=block_length, observation_area=[0, 520], observation_shape=[600, 240], ax=None):
    """
    Draw the busyness map.
    
    :param busy_level_normalized: Normalized busyness level
    :param env: Environment instance
    :param ego_vehicle: Ego vehicle instance
    :param grid_size: Size of the grid blocks [m]
    :param observation_area: Observation area [m]
    :param observation_shape: Shape of the observation area [pixels]
    :param ax: Matplotlib axis object for plotting
    """
    num_lanes = 4
    lane_width = 3.5  # Lane width
    road_length = observation_area[1] - observation_area[0]
    num_grids = int(road_length // grid_size)

    cmap = LinearSegmentedColormap.from_list("traffic_light", [(0, "green"), (0.5, "yellow"), (1, "red")])
    busy_level_rgb = cmap(busy_level_normalized[:, :52])

    env_frame = env.render(mode='rgb_array')
    if ax is None:
        fig, ax = plt.subplots(figsize=(env_frame.shape[1] / 100, env_frame.shape[0] / 100), dpi=100)
    else:
        ax.clear()

    ax.imshow(busy_level_rgb, origin='lower', extent=[observation_area[0], observation_area[1], 0, num_lanes * lane_width])

    for lane in range(num_lanes):
        for i in range(num_grids):
            rect = Rectangle((observation_area[0] + i * grid_size, lane * lane_width), grid_size, lane_width, linewidth=0.5, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    for vehicle in env.road.vehicles:
        if observation_area[0] < vehicle.position[0] and vehicle.position[0]< observation_area[1]:
            lane_id = int(vehicle.lane_index[2])
            car_x = vehicle.position[0]
            car_y = lane_id * lane_width + lane_width / 2
            car = Rectangle((car_x - 2.5, car_y - 1), 5, 2, linewidth=1, edgecolor='blue', facecolor='blue')
            ax.add_patch(car)

    ax.invert_yaxis()  # Invert the y-axis

    if ax is None:
        plt.title('Traffic Busy Levels on Highway')
        plt.xlabel('Position (m)')
        plt.ylabel('Lanes')
        plt.show()
    else:
        ax.set_title('Traffic Busy Levels on Highway')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Lanes')
    
def get_action_busyness(env, ego_vehicle, action_select, actions ,return_best_action = False, observation_area=[0, 740]):
    """
    Evaluate the busyness level for each action.
    
    :param env: Environment instance
    :param ego_vehicle: Ego vehicle instance
    :param action_select: List of selectable actions
    :param actions: Actions taken by vehicles
    :param return_best_action: Whether to return the best action
    :param observation_area: Observation area [m]
    :return: Busyness level for each action or the best action
    """
    actions_wight = {
        "LANE_LEFT": 1.2,
        "IDLE": 0.99,
        "LANE_RIGHT": 1.21,
        "FASTER": 0.98,
        "SLOWER": 1.0
    }
    ego_busyness_map = busyness_map(env, ego_vehicle, actions)
    action_busyness = []
    for ego_action in action_select:
        busy_sum = 0
        if int(ego_vehicle.lane_index[2]) == 3:
            x, lane_index = predict_trajectory(ego_vehicle, ego_action, 1 / env.config["policy_frequency"], action_sample, eval = True)
        else:
            x, lane_index = predict_trajectory(ego_vehicle, ego_action, 1 / env.config["policy_frequency"], action_sample, eval = True)
        for index ,position in enumerate(x):
            # Calculate the grid where the trajectory point is located
            ego_grid = int((position - observation_area[0]) // 10)
            if ego_grid < 0 or ego_grid >= ego_busyness_map.shape[1]:
                continue
            
            lane_id = int(lane_index[index])
            # busy_sum += max(ego_busyness_map[lane_id, ego_grid],0.001)*actions_wight[ego_action]*time_decay_function(index)
            busy_sum += max(ego_busyness_map[lane_id, ego_grid],0.001)*actions_wight[ego_action]*time_decay_linear_function(index, lambda_=0.2)
        
        action_busyness.append(busy_sum)

    if return_best_action:
        best_action = action_select[np.argmin(action_busyness)]
        # if best_action == "LANE_LEFT":
        #     if action_busyness[action_select.index("LANE_LEFT")] < action_busyness[action_select.index("IDLE")] - 0.1:
        #         best_action = "LANE_LEFT"
        #     else:
        #         best_action = "IDLE"
        return best_action
    else:
        return action_busyness
    
def predict_trajectory(vehicle, action, times, simulation_step, eval = True):
    """
    Predict the trajectory for a series of actions.
    
    :param vehicle: Vehicle instance
    :param action: Action to be taken
    :param times: Time interval for sampling points
    :param simulation_step: Number of simulation steps
    :param eval: Whether to evaluate the trajectory
    :return: List of trajectories
    """
    x = []
    lane = []
    if action == "LANE_LEFT":
        if vehicle.lane_index == ("a", "b", 3) or vehicle.lane_index == ("b", "c", 3):
            lane_index = 3
        else:
            lane_index = max(int(vehicle.lane_index[2]) - 1,0)
        speed = vehicle.speed*0.98
        # x.append(vehicle.position[0] + 8)
    elif action == "LANE_RIGHT":
        if vehicle.lane_index[2] == 2:
            lane_index = 2
        else:
            lane_index = min(int(vehicle.lane_index[2]) + 1,3)
        speed = vehicle.speed*0.98
    elif action == "FASTER":
        lane_index = int(vehicle.lane_index[2])
        speed = min(vehicle.speed*1.1, 30)
    elif action == "SLOWER":
        lane_index = int(vehicle.lane_index[2])
        speed = max(vehicle.speed*0.85, 10)
    else:
        lane_index = int(vehicle.lane_index[2])
        speed = vehicle.speed
    for i in range (simulation_step):
        if i == 0:
            x.append(vehicle.position[0])
            lane.append(lane_index if eval else vehicle.lane_index[2])
        else:
            x.append(x[-1] + speed * times)
            lane.append(lane_index)
    return x, lane

def simulate_and_display(env, num_steps=1000):
    """
    Embed the plot into the simulation environment for display.
    
    :param env: Environment instance
    :param num_steps: Number of simulation steps
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(20, 3))
    
    for _ in range(num_steps):
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            print("Simulation ended due to collision or other termination condition.")
            break
        # draw_busyness_map(busyness_map(env), env, ax=ax)
        draw_busyness_map(busyness_map(env, ego_vehicle=env.road.vehicles[0]), env, ax=ax)
        get_action_busyness(env, env.road.vehicles[0], ["LANE_LEFT", "IDLE", "LANE_RIGHT"], return_best_action=True)
        # plt.pause(0.1)  # Pause to update the image
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

def simulate_and_save_videos(env, num_steps=200, env_video_file="./environment23.mp4", busyness_video_file="./busyness23.mp4", output_csv_file="vehicle_positions_speeds5.csv", fps=10):
    """
    Simulate and save videos of the environment and busyness map.
    
    :param env: Environment instance
    :param num_steps: Number of simulation steps
    :param env_video_file: File path to save the environment video
    :param busyness_video_file: File path to save the busyness video
    :param output_csv_file: File path to save the vehicle positions and speeds
    :param fps: Frames per second for the video
    """
    env_frames = []
    busyness_frames = []
    action_mapping = {
        "LANE_LEFT": 0,
        "IDLE": 1,
        "LANE_RIGHT": 2,
        "FASTER": 3,
        "SLOWER": 4
    }
    select_action = [["IDLE", "LANE_RIGHT", "FASTER", "SLOWER"], ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"], ["LANE_LEFT", "IDLE", "FASTER", "SLOWER"], ["LANE_LEFT", "IDLE", "FASTER", "SLOWER"]]
    average_speed_eval = 0
    change_interval = 30
    # Add a counter for each vehicle to indicate the number of lane changes in the last 5 steps
    vehicle_lane_change_counter = {vehicle.id: change_interval for vehicle in env.road.vehicles}  # Initial value is 5
    # Open the CSV file in write mode, write the header
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(num_steps):

            q = PriorityQueue()
            vehicles_prior = []

            # Get the rendered environment image
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)

            index = 0
            speed_step = 0
            for vehicle in env.road.vehicles:
                speed_step += vehicle.speed
                priority_number = 0
                if vehicle.lane_index == ("c", "d", 3):
                    priority_number = -2
                    distance_to_merging_end = env.distance_to_merging_end(vehicle)
                    priority_number -= 2 * (env.ends[2] - distance_to_merging_end) / env.ends[2]
                else:
                    distance_to_merging_end = env.distance_to_merging_end(vehicle)
                    priority_number -= 2 * (env.ends[2] - distance_to_merging_end) / env.ends[2]

                priority_number += np.random.rand() * 0.001  # to avoid the same priority number for two vehicles
                q.put((priority_number, [vehicle, index]))
                index += 1

            while not q.empty():
                next_item = q.get()
                vehicles_prior.append(next_item[1])

            # Perform one step of simulation
            actions = [None for _ in range(len(env.road.vehicles))]
            for vehicle in vehicles_prior:
                # If it is a CAV
                if vehicle[0] in env.controlled_vehicles:
                    actions[vehicle[1]] = get_action_busyness(env, vehicle[0], select_action[vehicle[0].lane_index[2]], actions, return_best_action=True)
                # If it is an HDV
                # else:
                #     actions[vehicle[1]] = ide(env, vehicle[0])
                if actions[vehicle[1]] in ["LANE_LEFT", "LANE_RIGHT"]:
                    if vehicle_lane_change_counter[vehicle[0].id] >= change_interval:
                        vehicle_lane_change_counter[vehicle[0].id] = 1
                    else:
                        actions[vehicle[1]] = "FASTER"
                        vehicle_lane_change_counter[vehicle[0].id] += 1
                else:
                    vehicle_lane_change_counter[vehicle[0].id] += 1
            actions_select = []
            for vehicle in env.controlled_vehicles:
                actions_select.append(actions[vehicle.id])
            indexed_actions = [action_mapping[action] for action in actions_select]
            obs, reward, done, info = env.step(indexed_actions)
            actions = info["new_action"]
            print("step: ", step, "actions: ", actions)
            # Save each vehicle's position and speed to CSV
            for vehicle in env.road.vehicles:
                # Save vehicle information: time step, vehicle ID, lane index, position, speed
                writer.writerow({
                    'step': step,
                    'vehicle_id': vehicle.id,
                    'lane_index': vehicle.lane_index[2],
                    'position': vehicle.position[0],
                    'speed': vehicle.speed
                })
            average_speed_eval += speed_step / len(env.road.vehicles)
        

            # Create a matplotlib image of the same size as the environment rendering
            fig, ax = plt.subplots(figsize=(3*env_frame.shape[1] / 100, 1*env_frame.shape[0] / 100), dpi=100)
            ax.axis('off')  # Turn off axis display
        
            # Draw the busyness map
            draw_busyness_map(busyness_map(env, actions= [None for _ in range(len(vehicles_prior))]), env, ax=ax)
        
            # Save the busyness map as a numpy array
            fig.canvas.draw()
            busyness_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            busyness_frame = busyness_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            busyness_frames.append(busyness_frame)
            
            plt.close(fig)  # Close the current figure to free memory

            # Check if a collision occurred
            if done:  # If the simulation is complete or a collision occurred
                average_speed_eval /= step
                print(f"Simulation ended at step {step} due to collision or other termination condition.")
                break
    print("average_speed: ", average_speed_eval)


    # Use moviepy to convert the frame sequences into two mp4 videos
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    busyness_clip = ImageSequenceClip(busyness_frames, fps=fps)
    
    env_clip.write_videofile(env_video_file, codec="libx264")
    busyness_clip.write_videofile(busyness_video_file, codec="libx264")

    print(f"Environment video saved as {env_video_file}")
    print(f"Busyness video saved as {busyness_video_file}")

def calculate_action_weight(num_lanes = 3, straight_preference=2.0, alpha=1.0):
    """
    Calculate action weights based on traffic flow and straight preference.
    
    :param num_lanes: Number of lanes
    :param straight_preference: Preference factor for straight action to increase its weight
    :param alpha: Additional parameter for future use
    :return: 4x3 action weight matrix, where each row corresponds to a lane and each column represents an action
    """
    base_weight = [1.0, straight_preference, 1.0]  # Base weights, ensuring the straight (IDLE) action has the highest base weight
    
    action_weight = np.zeros((num_lanes, 3))
    
    for i in range(num_lanes):
        for j in range(num_lanes):
            if j == 1:  # IDLE action
                action_weight[i][j] = base_weight[j]
            else:  # LANE_LEFT and LANE_RIGHT actions
                if i == 0 and j == 0:  # No LANE_LEFT for the leftmost lane
                    action_weight[i][j] = 0.0
                elif i == num_lanes - 1 and j == 2:  # No LANE_RIGHT for the rightmost lane
                    action_weight[i][j] = 0.0
                else:
                    action_weight[i][j] = base_weight[j]
    
    # Normalize action weights for each lane so that the sum of each row is 1
    action_weight = action_weight / action_weight.sum(axis=1, keepdims=True)

    # Add weights for the merging lane
    merging_action_weight = np.array([0.8, 0.2, 0])
    action_weight = np.vstack([action_weight, merging_action_weight])
    
    return action_weight

def time_decay_function(time, lambda_=0.3):
    """
    Calculate the time step decay value.
    
    :param time: Time step
    :param lambda_: Decay rate
    :return: Decay value
    """
    return max(0, np.exp(-lambda_ * time))

def time_decay_linear_function(time, lambda_=0.2):
    """
    Calculate the linear time step decay value.
    
    :param time: Time step
    :param lambda_: Decay rate
    :return: Linear decay value
    """
    return max(0, 1 - lambda_ * time)

def traffic_flow(env, busyness_map, ego_vehicle = None ,alpha_h= 0.05, alpha_m = 0.4 , decay_m = 0.3 , length = 1):
    """
    Calculate the traffic flow and update the busyness map.
    
    :param env: Environment instance
    :param busyness_map: Current busyness map
    :param ego_vehicle: Ego vehicle instance
    :param alpha_h: Weight factor for main road traffic
    :param alpha_m: Weight factor for merging traffic
    :param decay_m: Decay rate for merging traffic
    :param length: Length of the observation area
    :return: Updated busyness map
    """
    # Consider vehicles on the main road
    for line in range(3):
        num_lane = 0
        speed_all = 0
        num_merge = 0
        for vehicle in env.road.vehicles:
            if vehicle == ego_vehicle:
                continue
            if vehicle.lane_index[2] == line:
                speed_all += vehicle.speed
                num_lane += 1
            if vehicle.lane_index in [("b", "c", 3), ("c", "d", 3)]:
                num_merge += 1
        density = num_lane / length
        average_speed = speed_all / max(num_lane,1)
        density_merge = num_merge / length
        busyness_map[line, :] += (alpha_h * density / max(average_speed,0.01) + alpha_m * density_merge * decay_m ** (3 - line)) 
    busyness_map[3,:] += 0.4
    return busyness_map

def ide(env, vehicle):
    """
    Determine the action for a vehicle based on its surrounding vehicles.
    
    :param env: Environment instance
    :param vehicle: Vehicle instance
    :return: Action to be taken ("SLOWER", "FASTER", or "IDLE")
    """
    v_fe, v_re = env.road.surrounding_vehicles(vehicle, True)
    if v_fe:
        if v_fe.position[0] - vehicle.position[0] < 10:
            return "SLOWER"
        elif v_fe.position[0] - vehicle.position[0] > 30:
            return "FASTER"
        else:
            return "IDLE"
    else:
        return "IDLE"
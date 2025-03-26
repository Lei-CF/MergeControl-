"""
This environment is built on HighwayEnv with three main roads and one merging lane.
modified on the basis of highway-env/highway_env/envs/merge_env_v1.py
Date: 03/26/2025
"""
import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle
import random


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    n_a = 6         # 离散动作数量，这里包括加速、减速、保持速度、左移、右移
    n_s = 21        # 状态空间数量，这里的25为5*5,前一个5为每个车辆状态包括横向、纵向位置和速度以及是否出现在当前车的视野里，
                    # 后一个5则为周围可观察车的数量，实验发现5为最好值，

    @classmethod
    def default_config(cls) -> dict:
        ##调取父类的config，获取默认配置
        config = super().default_config()
        ##更新自己的配置，以字典的形式
        config.update({
            "observation": {
                "type": "Kinematics"},        #使用车辆的动力学信息进行观测
            "action": {
                "type": "DiscreteMetaAction", #可以进行离散的纵向和横向动作
                "longitudinal": True,         #允许纵向动作
                "lateral": True},             #允许横向动作
            "controlled_vehicles": 1,         #定义受控车辆数量
            "screen_width": 1520,              #界面显示的宽度
            "screen_height": 120,             #界面显示的高度
            "centering_position": [0.6, 0.5], #界面中心位置比例
            "scaling":3,                     #界面的缩放比例，放大3倍
            "simulation_frequency": 15,  #[Hz]#模拟器的频率
            "duration": 60,  # time step      #模拟时间
            "policy_frequency": 5,  # [Hz]    #策略更新频率
            "reward_speed_range": [10, 30],   #奖励计算中的速度范围，只有在这个范围内的速度才能有奖励
            "COLLISION_REWARD": 200,  # default=200 #碰撞奖励
            "HIGH_SPEED_REWARD": 1,  # default=0.5  #高速奖励
            "HEADWAY_COST": 4,  # default=1         #其余参数配置
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 4,  # default=4
            "LANE_CHANGE_COST": 2,  # default=0.5
            "traffic_density": 3,  # easy or hard modes #模式更换
        })
        return config

    ##接受一个int类型的action参数，返回float类型的奖励值
    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        #使用agent_reward函数计算受控车辆的奖励平均值
        # return sum(self._agent_reward(action[index], vehicle) for index, vehicle in enumerate(self.controlled_vehicles)) \
        #        / len(self.controlled_vehicles)
        return [self._agent_reward(action[index], vehicle) for index, vehicle in enumerate(self.controlled_vehicles)]
               

   ##接受一个int类型的action参数，饭返回一个float类型的奖励值
    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        # the optimal reward is 0，最佳奖励为0
        
        #将车辆速度缩放映射到【0，1】区间
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # compute cost for staying on the merging lane
        #如果车辆位于匝道上，则计算该车道的代价，代价为基于车辆与匝道尽头距离的指数函数
        if vehicle.lane_index == ("c", "d", 3):
            Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
                    10 * self.ends[2]))
        else:
            Merging_lane_cost = 0

        # lane change cost to avoid unnecessary/frequent lane changes
        Lane_change_cost = -1 * self.config["LANE_CHANGE_COST"] if action == 0 or action == 2 else 0

        # compute headway cost
        #计算车辆的车头时距，并根据车头时距计算一个代价，车头时距除以 某个时间常量与车速的乘积，如果车速为0，则代价为0
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        # compute overall reward
        ##计算总的代价，由碰撞代价+高速代价+匝道代价+车头时距代价+换道惩罚五部分组成
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (-1 * self.config["HIGH_SPEED_REWARD"] * np.clip(1 - scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0) \
                    + Lane_change_cost
        return reward


    #计算某个车辆的局部奖励
    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # vehicle is on the main road，车辆在主道路上时
            if vehicle.lane_index[2] in [0, 1, 2]:
                ##获取车辆的前方临近车辆fl，后方临近车辆rl
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                ##如果当前车道存在侧方车道时，还需观测侧方车道的临近车辆
                v_fr = []  # 侧前方车辆
                v_rr = []  # 侧后方车辆
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    #将车辆和道路传递给该函数，得出其临近车辆
                    for side_lane in self.road.network.side_lanes(vehicle.lane_index):
                        v_fr.append(self.road.surrounding_vehicles(vehicle, lane_index = side_lane)[0])
                        v_rr.append(self.road.surrounding_vehicles(vehicle, lane_index = side_lane)[1])
                    # v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                    #                                             self.road.network.side_lanes(
                    #                                                 vehicle.lane_index)[0])
                # assume we can observe the ramp on this road
                # 这里假设了在优化控制区域的时候，可以观测到匝道上的车辆
                elif vehicle.lane_index == ("b", "c", 2) :
                    v_fr.append(self.road.surrounding_vehicles(vehicle, lane_index = ("b", "c", 3))[0])
                    v_rr.append(self.road.surrounding_vehicles(vehicle, lane_index = ("b", "c", 3))[1]) 
                else:
                    v_fr, v_rr = [], []
            else:
                # vehicle is on the ramp
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                v_fr = []  # 侧前方车辆
                v_rr = []  # 侧后方车辆
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr.append(self.road.surrounding_vehicles(vehicle, lane_index = self.road.network.side_lanes(vehicle.lane_index)[0])[0])
                    v_rr.append(self.road.surrounding_vehicles(vehicle, lane_index = self.road.network.side_lanes(vehicle.lane_index)[0])[1])
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("b", "c", 3):
                    v_fr.append(self.road.surrounding_vehicles(vehicle, lane_index = ("b", "c", 2))[0])
                    v_rr.append(self.road.surrounding_vehicles(vehicle, lane_index = ("b", "c", 2))[1])
                else:
                    v_fr, v_rr = [], []
            #遍历观测到的临近车辆，只有类型为MDP的才会被加入neighbor中
            for v in v_fr + [v_fl] + [vehicle] + [v_rl] + v_rr:
                if type(v) is MDPVehicle and v is not None:
                    neighbor_vehicle.append(v)
            ##计算车辆的临近奖励之和，并除以非空数量，得到平均奖励，最后成为区域的局部奖励
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = [] 
        obs, reward, done, info = super().step(action)
        # action = info["new_action"]
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info

        for index, vehicle in enumerate(self.controlled_vehicles):
            vehicle.local_reward = self._agent_reward(action[index], vehicle)
        # local reward
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # regional reward
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        obs = np.asarray(obs).reshape((len(obs), -1))
        # obs = None
        # print("info~~~~~\n", info)
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.road.vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
               or all(vehicle.position[0] > 515 for vehicle in self.road.vehicles)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self, num_CAV=0) -> None:
        self._make_road()
        if self.config["traffic_density"] == 1:
            # easy mode: 2-4 CAVs
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(2, 5), 1)[0]
            else:
                num_CAV = num_CAV
            # num_HDV = np.random.choice(np.arange(1, 4), 1)[0]

        elif self.config["traffic_density"] == 2:
            # hard mode: 4-6 CAVs 
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(4, 7), 1)[0]
            else:
                num_CAV = num_CAV
            # num_HDV = np.random.choice(np.arange(2, 5), 1)[0]

        elif self.config["traffic_density"] == 3:
            # hard mode: 7-9 CAVs 
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(7, 10), 1)[0]
            else:
                num_CAV = num_CAV
            # num_HDV = np.random.choice(np.arange(3, 6), 1)[0]
        self._make_vehicles(num_CAV = num_CAV, num_HDV = 0)

        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self, ) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        # LineType就是车道线的类型，包括实线、虚线、无线，是个1*2的数组，第一个元素是左边的线，第二个元素是右边的线
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        # 添加三条主路，每条主路有三条车道
        for i in range(3):  # i 代表车道编号 0, 1, 2
            if i == 0:
                net.add_lane("a", "b", StraightLane([0, i * 3.5], [sum(self.ends[:1]), i * 3.5], line_types=[c, s]))
                net.add_lane("b", "c", StraightLane([sum(self.ends[:1]), i * 3.5], [sum(self.ends[:2]), i * 3.5], line_types=[c, s]))
                net.add_lane("c", "d", StraightLane([sum(self.ends[:2]), i * 3.5], [sum(self.ends[:3]), i * 3.5], line_types=[c, s]))
                net.add_lane("d", "e", StraightLane([sum(self.ends[:3]), i * 3.5], [sum(self.ends), i * 3.5], line_types=[c, s]))
            elif i == 1:
                net.add_lane("a", "b", StraightLane([0, i * 3.5], [sum(self.ends[:1]), i * 3.5], line_types=[n, s]))
                net.add_lane("b", "c", StraightLane([sum(self.ends[:1]), i * 3.5], [sum(self.ends[:2]), i * 3.5], line_types=[n, s]))
                net.add_lane("c", "d", StraightLane([sum(self.ends[:2]), i * 3.5], [sum(self.ends[:3]), i * 3.5], line_types=[n, s]))
                net.add_lane("d", "e", StraightLane([sum(self.ends[:3]), i * 3.5], [sum(self.ends), i * 3.5], line_types=[n, s]))
            else:
                net.add_lane("a", "b", StraightLane([0, i * 3.5], [sum(self.ends[:1]), i * 3.5], line_types=[n, c]))
                net.add_lane("b", "c", StraightLane([sum(self.ends[:1]), i * 3.5], [sum(self.ends[:2]), i * 3.5], line_types=[n, c]))
                net.add_lane("c", "d", StraightLane([sum(self.ends[:2]), i * 3.5], [sum(self.ends[:3]), i * 3.5], line_types=[n, s]))
                net.add_lane("d", "e", StraightLane([sum(self.ends[:3]), i * 3.5], [sum(self.ends), i * 3.5], line_types=[n, c]))
        amplitude = 3.25
        lab = StraightLane([0, 6.5 + 3*3.5], [self.ends[0], 6.5 + 3*3.5], line_types=[c, c], forbidden=True)
        lbc = SineLane(lab.position(self.ends[0], -amplitude), lab.position(sum(self.ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * self.ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lcd = StraightLane(lbc.position(self.ends[1], 0), lbc.position(self.ends[1], 0) + [self.ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("a", "b", lab)    # 匝道尽量不要和主道名字一致，否则在换道时需要特殊处理
        net.add_lane("b", "c", lbc)
        net.add_lane("c", "d", lcd)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lcd.position(self.ends[2], 0)))
        self.road = road

    def _make_vehicles(self, num_CAV=4, num_HDV=3) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        # 不能超过end[0]
        spawn_points_s1 = [10, 40, 70, 100, 130, 160, 190, 215]
        spawn_points_s2 = [10, 40, 70, 100, 130, 160, 190, 215]
        spawn_points_s3 = [10, 40, 70, 100, 130, 160, 190, 215]
        # spawn_points_m = [5, 35, 65, 95, 125, 155, 185, 215]
        spawn_points_m = [5, 55, 95, 195]
        # spawn_points_s = [10, 50, 90, 130, 170, 210]
        # spawn_points_m = [5, 45, 85, 125, 165, 205]

        if self.config["traffic_density"] == 1:
            max_main_road = 3
            max_merging_road = 2
        elif self.config["traffic_density"] == 2:
            max_main_road = 5
            max_merging_road = 3
        elif self.config["traffic_density"] == 3:
            max_main_road = 8
            max_merging_road = 4
        # random.seed()
        CAV_ratio_main = 1.0
        CAV_ratio_ramp = 1.0
        if self.seed != 1:
            num_lane1 = random.randint(0,max_main_road)
            num_lane2 = random.randint(0,max_main_road)
            num_lane3 = random.randint(0,max_main_road)
            num_lane4 = random.randint(0,max_merging_road)
        else:
            num_lane1 = 6
            num_lane2 = 0
            num_lane3 = 6
            num_lane4 = 3
        # print("num_lane1", num_lane1)
        # print("num_lane2", num_lane2)
        # print("num_lane3", num_lane3)
        # print("num_lane4", num_lane4)
        num_main1 = int(num_lane1*CAV_ratio_main)
        num_main2 = int(num_lane2*CAV_ratio_main)
        num_main3 = int(num_lane3*CAV_ratio_main)
        num_main4 = int(num_lane4*CAV_ratio_ramp)
        # num_main3 = int((num_lane1 + num_lane2 + num_lane3 + num_lane4)*CAV_ratio_main) - num_main1 - num_main2 - num_main4
        """Spawn points for CAV"""
        # spawn point indexes on the straight road
        spawn_point_s0_c = np.random.choice(spawn_points_s1, num_main1, replace=False)
        spawn_point_s1_c = np.random.choice(spawn_points_s2, num_main2, replace=False)
        spawn_point_s2_c = np.random.choice(spawn_points_s3, num_main3, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_c = np.random.choice(spawn_points_m, num_main4,
                                           replace=False)
        spawn_point_s0_c = list(spawn_point_s0_c)
        spawn_point_s1_c = list(spawn_point_s1_c)
        spawn_point_s2_c = list(spawn_point_s2_c)
        spawn_point_m_c = list(spawn_point_m_c)
        # remove the points to avoid duplicate
        for a0 in spawn_point_s0_c:
            spawn_points_s1.remove(a0)
        for a1 in spawn_point_s1_c:
            spawn_points_s2.remove(a1)
        for a2 in spawn_point_s2_c:
            spawn_points_s3.remove(a2)
        for b in spawn_point_m_c:
            spawn_points_m.remove(b)

        """Spawn points for HDV"""
        num_main1_h = num_lane1 - num_main1
        num_main2_h = num_lane2 - num_main2
        num_main3_h = num_lane3 - num_main3
        num_main4_h = num_lane4 - num_main4
        # spawn point indexes on the straight road
        spawn_point_s0_h = np.random.choice(spawn_points_s1, num_main1_h, replace=False)
        spawn_point_s1_h = np.random.choice(spawn_points_s2, num_main2_h, replace=False)
        spawn_point_s2_h = np.random.choice(spawn_points_s3, num_main3_h, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(spawn_points_m, num_main4_h,
                                           replace=False)
        spawn_point_s0_h = list(spawn_point_s0_h)
        spawn_point_s1_h = list(spawn_point_s1_h)
        spawn_point_s2_h = list(spawn_point_s2_h)
        spawn_point_m_h = list(spawn_point_m_h)

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_lane1 + num_lane2 + num_lane3 + num_lane4 ) * 2 + 25  # range from [25, 27]
        loc_noise = np.random.rand(num_lane1 + num_lane2 + num_lane3 + num_lane4) * 3 - 1.5  # range from [-1.5, 1.5]
        # loc_noise = np.random.rand(num_CAV + num_HDV) * 1 - 0.5  # range from [-0.5, 0.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)
        # loc_noise_m = list(loc_noise_m)

        """spawn the CAV on the straight road first"""
        # print("num_CAV", num_CAV)
        for _ in range(num_main1):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 0)).position(
                spawn_point_s0_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            # print("ab0", ego_vehicle.position)
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)
        for _ in range(num_main2):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 1)).position(
                spawn_point_s1_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            # print("ab1", ego_vehicle.position)
            self.controlled_vehicles.append(ego_vehicle)    
            road.vehicles.append(ego_vehicle) 
        for _ in range(num_main3):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 2)).position(
                spawn_point_s2_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            # print("ab2", ego_vehicle.position)
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)
        """spawn the rest CAV on the merging road"""
        for _ in range(num_main4):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 3)).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            # print("ab3", ego_vehicle.position)
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the HDV on the main road first"""
        for _ in range(num_main1_h):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
                    spawn_point_s0_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))
        for _ in range(num_main2_h):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(
                    spawn_point_s1_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))
        for _ in range(num_main3_h):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 2)).position(
                    spawn_point_s2_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))

        """spawn the rest HDV on the merging road"""
        for _ in range(num_main4_h):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 3)).position(
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

class MergeEnvMARL(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 8
        })
        return config



register(
    id='merge-v1',
    entry_point='envs:MergeEnv',
)

register(
    id='merge-multi-agent-v0',
    entry_point='envs:MergeEnvMARL',
)
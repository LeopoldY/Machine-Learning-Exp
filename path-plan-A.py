import cv2
import numpy as np
import math

import matplotlib.pyplot as plt

show_animation = False     # 启动动画显示的选项


class AStarPlanner:

    def __init__(self, grid, resolution, rr):
        #   初始化A*路径规划算法的网格地图
        #   ox：障碍物的x位置列表（以米为单位）
        #   oy：障碍物的y位置列表（以米为单位）
        #   resolution：网格分辨率（以米为单位），用于定义每个网格的大小。
        #   rr：机器人的半径（以米为单位）

        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0 
        self.max_x, self.max_y = 500, 800
        #   self.min_x、self.min_y、self.max_x 和 self.max_y 变量被初始化为0，用于记录地图中x和y方向上的最小和最大值。
        self.obstacle_map = None # 障碍物地图
        self.x_width, self.y_width = 0, 0
        #   self.x_width 和 self.y_width 也被初始化为0，它们将用于存储地图的宽度和高度。
        self.motion = self.get_motion_model()
        #   self.motion = self.get_motion_model() 调用 get_motion_model() 函数，获取机器人的运动模型。这个模型描述了机器人可以移动的方向和方式。
        self.calc_obstacle_map(grid)
        #    self.calc_obstacle_map(ox, oy) 调用 calc_obstacle_map() 函数，计算栅格地图中的障碍物信息。
        #    传入的 ox 和 oy 列表包含了障碍物的位置信息，函数将根据分辨率和机器人半径来计算栅格地图中的障碍物。

    class Node:
        # x：网格中的x坐标索引。
        # y：网格中的y坐标索引。
        # cost：从起始节点到当前节点的路径代价。
        # parent_index：父节点的索引，用于构建路径。
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1) # 起始节点
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1) # 目标节点

        open_set, closed_set = dict(), dict() # open_set 和 closed_set 分别用于存储待探索的节点和已经探索过的节点。
        open_set[self.calc_grid_index(start_node)] = start_node # 将起始节点添加到 open_set 中。

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            #  将具有最小总代价的节点标记为当前节点，以便进行进一步的探索和路径构建。
            current = open_set[c_id] # 获取当前节点

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry


    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        '''
        计算x或y坐标在栅格地图中的索引
        :param position: x或y坐标
        :param min_pos: x或y方向上的最小值
        :return: x或y坐标在栅格地图中的索引
        '''
        return round((position - min_pos) / self.resolution) 

    def calc_grid_index(self, node):
        '''
        计算节点在栅格地图中的索引
        :param node: 节点
        :return: 节点在栅格地图中的索引
        '''
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        '''
        验证节点是否在栅格地图中
        :param node: 节点
        :return: True or False
        '''
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, grid):
        '''
        计算栅格地图中的障碍物
        :param grid: 栅格地图
        :return: None
        '''
        self.y_width = grid.shape[0]
        self.x_width = grid.shape[1]
        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                if grid[iy, ix] != 1:
                    self.obstacle_map[ix][iy] = True


    @staticmethod
    def get_motion_model():
        '''
        获取机器人的运动模型
        :return: 运动模型
        '''
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)] ]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 234  # [m]
    sy = 253  # [m]
    gx = 275  # [m]
    gy = 385  # [m]
    grid_size = 1.0  # [m] 
    robot_radius = 1.0  # [m]

    # set obstacle positions
    # 障碍物及边框
    small_birdseye = cv2.imread('img_bev.jpg')
    grid = np.zeros((small_birdseye.shape[0], small_birdseye.shape[1]), dtype=np.uint8) # 创建一个与图像大小相同的二维数组
    white_lower = np.array([200, 200, 200], dtype=np.uint8)  # 白色下界
    white_upper = np.array([255, 255, 255], dtype=np.uint8)  # 白色上界

    for y in range(small_birdseye.shape[0]): # 遍历图像的每一个像素点
        for x in range(small_birdseye.shape[1]): # 遍历图像的每一个像素点
            pixel_color = small_birdseye[y, x] # 获取当前像素点的颜色
            if cv2.inRange(pixel_color, white_lower, white_upper).all():  # 如果颜色在指定的白色范围内
                grid[y, x] = 1
            else:
                grid[y, x] = 0

    a_star = AStarPlanner(grid, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    
    plt.imshow(small_birdseye)
    plt.plot(rx, ry, "-r")  #“红色实线”
    plt.pause(0.1)
    plt.show()


if __name__ == '__main__':
    main()

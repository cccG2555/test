import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation


np.random.seed(0)



# 맵 크기 설정
MAP_WIDTH = 25  # 가로 25미터
MAP_HEIGHT = 15  # 세로 15미터
CELL_SIZE = 1.0  # 각 셀의 크기 (1미터)


# SLAM 시작 위치
START = (22.5,12.5)


# 맵 생성
grid_map = np.zeros((MAP_HEIGHT, MAP_WIDTH))

def visualize_grid_map(map_data):
    plt.figure(figsize=(12, 7))
    

    colors = ['white', 'black', 'lightblue', 'blue']
    custom_cmap = ListedColormap(colors)
    
    # 맵 표시
    plt.imshow(map_data, cmap=custom_cmap, vmin=0, vmax=3, origin='lower', extent=[0, MAP_WIDTH, 0, MAP_HEIGHT])

    plt.text(START[0],START[1],'S',ha='center',va='center', color='black', fontsize=16, fontweight='bold')
    
    # 그리드 라인 추가
    # 세로선
    for x in range(MAP_WIDTH + 1):
        plt.axvline(x, color='gray', linewidth=1)
    # 가로선
    for y in range(MAP_HEIGHT + 1):
        plt.axhline(y, color='gray', linewidth=1)
    
    # 축 눈금 설정
    plt.xticks(range(MAP_WIDTH+1))
    plt.yticks(range(MAP_HEIGHT+1))
    
    # 축 레이블
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    # 제목
    plt.title('Grid Map (1m x 1m cells)')
    
    #plt.grid(True)
    plt.show()




# 랜드마크 표시
grid_map[4,5]=3
grid_map[9,16]=3
grid_map[11,5]=3
grid_map[2,15]=3


# 맵 시각화
visualize_grid_map(grid_map)

# 특정 셀의 실제 미터 단위 좌표 확인을 위한 함수
def get_cell_center(cell_x, cell_y):
    """셀의 중심점 좌표를 반환"""
    return cell_x * CELL_SIZE + CELL_SIZE/2, cell_y * CELL_SIZE + CELL_SIZE/2



landmark1 = get_cell_center(5,4)
landmark2 = get_cell_center(16,9)
landmark3 = get_cell_center(5,11)
landmark4 = get_cell_center(15,2)




START_CENTER = get_cell_center(START[0],START[1])

WORLD_SIZE = 25

MEASUREMENT_RANGE = 25


# 노이즈
MOTION_NOISE = 0.3

MEASUREMENT_NOISE = 0.6


N_LANDMARKS = 4

LANDMARKS = np.array([landmark1, landmark2, landmark3, landmark4])

omega = np.array([[[0.0, 0.0] for _ in range(N_LANDMARKS + 1)]

                              for _ in range(N_LANDMARKS + 1)])

xi = np.array([[0.0, 0.0] for _ in range(N_LANDMARKS + 1)])

omega[0, 0, :] = np.array([1.0, 1.0])

xi[0, :] = np.array([START_CENTER[0],START_CENTER[1]])




def slam(i, dx, dy, Z):

    global omega, xi

    omega = np.insert(omega, i + 1, 0, axis=0)

    omega = np.insert(omega, i + 1, 0, axis=1)

    xi = np.insert(xi, i + 1, 0, axis=0)

    epsilon = 1e-10

    omega[i + 1, i + 1] += epsilon

    for meas in Z:

        j, x, y = meas

        omega[i, i] = omega[i, i] + 1/MEASUREMENT_NOISE

        omega[i, i + j + 2] = omega[i, i + j + 2] - 1/MEASUREMENT_NOISE

        omega[i + j + 2, i] = omega[i + j + 2, i] - 1/MEASUREMENT_NOISE

        omega[i + j + 2, i + j + 2] = omega[i + j + 2, i + j + 2] + 1/MEASUREMENT_NOISE

        xi[i, :] = xi[i, :] - np.array([x, y])/MEASUREMENT_NOISE

        xi[i + j + 2, :] = xi[i + j + 2, :] + np.array([x, y])/MEASUREMENT_NOISE

    omega[i, i] = omega[i, i] + 1/MOTION_NOISE

    omega[i + 1, i + 1] = omega[i + 1, i + 1] + 1/MOTION_NOISE

    omega[i + 1, i] = omega[i + 1, i] - 1/MOTION_NOISE

    omega[i, i + 1] = omega[i, i + 1] - 1/MOTION_NOISE

    xi[i, :] = xi[i, :] - np.array([dx, dy])/MOTION_NOISE

    xi[i + 1, :] = xi[i + 1, :] + np.array([dx, dy])/MOTION_NOISE



    try:

        mu_x = np.linalg.inv(omega[:, :, 0]).dot(xi[:, 0])

        mu_y = np.linalg.inv(omega[:, :, 1]).dot(xi[:, 1])

    except np.linalg.LinAlgError:

        mu_x = np.linalg.pinv(omega[:, :, 0]).dot(xi[:, 0])

        mu_y = np.linalg.pinv(omega[:, :, 1]).dot(xi[:, 1])

    return np.c_[mu_x, mu_y]






# bycicle 모델

class Robot(object):

    def __init__(self, length=0.5, world_size=WORLD_SIZE,

                 measurement_range=MEASUREMENT_RANGE,

                 motion_noise=MOTION_NOISE,

                 measurement_noise=MEASUREMENT_NOISE):

        """

        Creates robot and initializes location/orientation to 0, 0, 0.

        """


        self.world_size = world_size

        self.measurement_range = measurement_range

        self.motion_noise = motion_noise

        self.measurement_noise = measurement_noise



        self.x = START[0]

        self.y = START[1]


        self.orientation = 0.0

        self.length = length

        self.steering_noise = MOTION_NOISE

        self.distance_noise = MOTION_NOISE

        self.steering_drift = 0.0

    def set(self, x, y, orientation):

        """

        Sets a robot coordinate.

        """

        self.x = x

        self.y = y

        self.orientation = orientation % (2.0 * np.pi)

    def set_noise(self, steering_noise, distance_noise):

        """

        Sets the noise parameters.

        """

        # makes it possible to change the noise parameters

        # this is often useful in particle filters

        self.steering_noise = steering_noise

        self.distance_noise = distance_noise

    def set_steering_drift(self, drift):

        """

        Sets the systematical steering drift parameter

        """

        self.steering_drift = drift

    def move(self, steering, distance, tolerance=0.001, max_steering_angle=np.pi / 4.0):

        """

        steering = front wheel steering angle, limited by max_steering_angle

        distance = total distance driven, most be non-negative

        """

        if steering > max_steering_angle:

            steering = max_steering_angle

        if steering < -max_steering_angle:

            steering = -max_steering_angle

        if distance < 0.0:

            distance = 0.0

        # apply noise

        steering2 = random.gauss(steering, self.steering_noise)

        distance2 = random.gauss(distance, self.distance_noise)

        # apply steering drift

        steering2 += self.steering_drift

        # Execute motion

        turn = np.tan(steering2) * distance2 / self.length

        if abs(turn) < tolerance:
            # 새로운 위치 계산
            new_x = self.x + distance2 * np.cos(self.orientation)
            new_y = self.y + distance2 * np.sin(self.orientation)
            new_orientation = (self.orientation + turn) % (2.0 * np.pi)
        else:
            radius = distance2 / turn
            cx = self.x - (np.sin(self.orientation) * radius)
            cy = self.y + (np.cos(self.orientation) * radius)
            new_orientation = (self.orientation + turn) % (2.0 * np.pi)
            new_x = cx + (np.sin(new_orientation) * radius)
            new_y = cy - (np.cos(new_orientation) * radius)


    
        # 맵 안에 있으면 위치 업데이트
        self.x = new_x
        self.y = new_y
        self.orientation = new_orientation
        return True




    def sense(self):

        Z = []

        for i, landmark in enumerate(LANDMARKS):


            dx = landmark[0] - self.x + random.gauss(0, self.measurement_noise)

            dy = landmark[1] - self.y + random.gauss(0, self.measurement_noise)

            if self.measurement_range < 0.0 or sqrt(dx**2 + dy**2) <= self.measurement_range:

                Z.append([i, dx, dy])

        return Z
    

    def __repr__(self):

        return '[x=%.5f y=%.5f orient=%.5f]' % (self.x, self.y, self.orientation)





print("start 지점: ", START_CENTER)

robot = Robot()

robot.set(START_CENTER[0],START_CENTER[1],1.5*np.pi)

n_steps = 50
distance = 0.5
steering_increment = 2*np.pi / 50






fig, ax = plt.subplots()

ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'g*', label='Actual Landmark')

estimated_landmarks, = ax.plot([], [], 'k*', label='Estimated Landmark')

actual_position, = ax.plot([], [], 'r.', label='Actual Position')

estimated_position, = ax.plot([], [], 'b.', label='Estimated Position')

actual_path, = ax.plot([], [], 'r--')

estimated_path, = ax.plot([], [], 'b:')

actual_values = []

estimated_values = []

def init():

    ax.set_xlim(-1, 35)

    ax.set_ylim(-1, 20)

    plt.legend(loc='upper left')

    fig.set_size_inches(12, 7.2)  # 25:15 비율에 맞게 설정

    return (actual_position, estimated_position, estimated_landmarks,

            actual_path, estimated_path,)


def animate(i):

    print("i: ", i)

    Z = robot.sense()

    robotpose1 = get_cell_center(robot.x, robot.y)
    robot.move(steering_increment, distance)
    robotpose2 = get_cell_center(robot.x, robot.y)

    mu = slam(i, robotpose2[0]-robotpose1[0], robotpose2[1]-robotpose1[1], Z)

    actual_values.append([robot.x, robot.y])

    estimated_values.append([mu[i + 1, 0], mu[i + 1, 1]])

    # 경로 업데이트

    actual_path.set_data([pos[0] for pos in actual_values], [pos[1] for pos in actual_values])

    estimated_path.set_data([pos[0] for pos in estimated_values], [pos[1] for pos in estimated_values])

    # 현재 위치 업데이트 (리스트로 변경)

    actual_position.set_data([robot.x], [robot.y])  # 단일 값을 리스트로 변환

    estimated_position.set_data([mu[i + 1, 0]], [mu[i + 1, 1]])  # 단일 값을 리스트로 변환

    # 랜드마크 위치 업데이트

    est_lm = np.array([[mu[i + j + 2, 0], mu[i + j + 2, 1]] for j in range(N_LANDMARKS)])

    estimated_landmarks.set_data(est_lm[:, 0], est_lm[:, 1])

    return (actual_position, estimated_position, estimated_landmarks,

            actual_path, estimated_path,)

# 애니메이션 생성 및 저장

anim = animation.FuncAnimation(fig, animate, frames=n_steps,

                             interval=550, init_func=init, blit=True)

# 애니메이션 저장 시도


try:

    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=20)

    anim.save('project_slam.gif', writer=writer)


except Exception as e:

    print(f"Error saving animation: {e}")

    plt.show()  # 저장 실패 시 화면에 표시

import numpy as np
import math as math
from scipy.spatial import distance


def reward_function(params):

    ################## INPUT PARAMETERS ###################

    # Read all input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    heading = params['heading']
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering_angle = params['steering_angle']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    is_offtrack = params['is_offtrack']

    #################### RACING LINE ######################

    # Optimal racing line for the Spain track
    # Each row: [x,y,speed,timeFromPreviousPoint]
    racing_track = np.array([[0.2222, 2.82648, 1.98165, 0.07461],[0.23058, 2.67919, 1.99828, 0.07383],[0.25374, 2.53382, 2.01534, 0.07304],[0.29112, 2.39177, 2.03718, 0.0721],[0.34198, 2.25422, 2.06462, 0.07103],[0.40552, 2.12217, 2.09879, 0.06982],[0.48083, 1.99645, 2.14089, 0.06846],[0.56698, 1.87764, 2.19237, 0.06694],[0.66302, 1.76618, 2.25502, 0.06524],[0.76799, 1.66231, 2.33098, 0.06335],[0.88095, 1.5661, 2.42301, 0.06124],[1.00095, 1.4775, 2.53469, 0.05885],[1.12701, 1.3963, 2.67072, 0.05614],[1.25809, 1.32221, 2.83762, 0.05306],[1.3931, 1.25478, 3.04562, 0.04955],[1.53087, 1.19349, 3.31164, 0.04553],[1.6703, 1.13764, 3.66661, 0.04096],[1.81041, 1.08642, 4.0, 0.0373],[1.95048, 1.03889, 4.0, 0.03698],[2.08923, 0.99431, 4.0, 0.03643],[2.2065, 0.95772, 4.0, 0.03071],[2.32273, 0.92121, 4.0, 0.03046],[2.43881, 0.88427, 4.0, 0.03045],[2.55467, 0.84668, 4.0, 0.03045],[2.67024, 0.80826, 4.0, 0.03045],[2.78546, 0.76879, 4.0, 0.03045],[2.90025, 0.72807, 3.93149, 0.03098],[3.0148, 0.68663, 3.65867, 0.03329],[3.12916, 0.64462, 3.44624, 0.03535],[3.2563, 0.59902, 3.27772, 0.04121],[3.38537, 0.55448, 3.14259, 0.04345],[3.51655, 0.5116, 3.03392, 0.04549],[3.64985, 0.47105, 2.94676, 0.04728],[3.78516, 0.43353, 2.87757, 0.0488],[3.92228, 0.39973, 2.82373, 0.05001],[4.06093, 0.37032, 2.78355, 0.05092],[4.20086, 0.34589, 2.75538, 0.05155],[4.34178, 0.32698, 2.73821, 0.05193],[4.48343, 0.31405, 2.73118, 0.05208],[4.62552, 0.30749, 2.73118, 0.05208],[4.76781, 0.30762, 2.73118, 0.0521],[4.91003, 0.31472, 2.73118, 0.05214],[5.05194, 0.32899, 2.73118, 0.05222],[5.19327, 0.35061, 2.73118, 0.05235],[5.33376, 0.3797, 2.73118, 0.05253],[5.47314, 0.41633, 2.73371, 0.05272],[5.61114, 0.46054, 2.7454, 0.05278],[5.74746, 0.51229, 2.76604, 0.05272],[5.88183, 0.57154, 2.79544, 0.05253],[6.01394, 0.63814, 2.83356, 0.05221],[6.1435, 0.71192, 2.88069, 0.05176],[6.27023, 0.79263, 2.93708, 0.05116],[6.39386, 0.87996, 3.0032, 0.0504],[6.51416, 0.97354, 3.07967, 0.04949],[6.63091, 1.07295, 3.16736, 0.04841],[6.74397, 1.17771, 3.26753, 0.04717],[6.85323, 1.28731, 3.3815, 0.04577],[6.95865, 1.40124, 3.51134, 0.04421],[7.06027, 1.51895, 3.6596, 0.04249],[7.15818, 1.63994, 3.82979, 0.04064],[7.25251, 1.76371, 4.0, 0.03891],[7.34347, 1.88981, 4.0, 0.03887],[7.43133, 2.01782, 4.0, 0.03882],[7.51633, 2.14735, 4.0, 0.03873],[7.59916, 2.27864, 3.85467, 0.04027],[7.67974, 2.41137, 3.57723, 0.04341],[7.75803, 2.54554, 3.35363, 0.04632],[7.83394, 2.68117, 3.16921, 0.04904],[7.90742, 2.81824, 3.01426, 0.0516],[7.97828, 2.9568, 2.88244, 0.05399],[8.04582, 3.09712, 2.76916, 0.05624],[8.10934, 3.2394, 2.67107, 0.05833],[8.16809, 3.38375, 2.58582, 0.06027],[8.22131, 3.53019, 2.51158, 0.06204],[8.2682, 3.6786, 2.4468, 0.06361],[8.308, 3.82872, 2.39048, 0.06497],[8.33997, 3.98017, 2.34171, 0.0661],[8.36343, 4.13248, 2.29974, 0.06701],[8.37778, 4.28511, 2.26413, 0.06771],[8.38253, 4.43746, 2.23442, 0.06822],[8.37724, 4.58888, 2.21023, 0.06855],[8.36159, 4.73868, 2.19128, 0.06873],[8.33535, 4.88612, 2.17733, 0.06878],[8.29838, 5.03046, 2.16825, 0.06872],[8.25061, 5.17091, 2.16388, 0.06856],[8.19208, 5.30668, 2.16388, 0.06833],[8.12291, 5.43696, 2.16388, 0.06817],[8.0433, 5.56093, 2.16388, 0.06808],[7.95357, 5.67776, 2.16388, 0.06808],[7.85413, 5.78665, 2.16388, 0.06815],[7.74548, 5.88677, 2.16388, 0.06828],[7.62825, 5.97737, 2.1642, 0.06846],[7.50318, 6.05773, 2.16916, 0.06853],[7.37112, 6.12721, 2.17879, 0.06849],[7.23303, 6.18528, 2.19318, 0.0683],[7.08992, 6.23151, 2.21241, 0.06798],[6.94288, 6.26564, 2.23672, 0.06749],[6.79299, 6.28756, 2.26636, 0.06684],[6.64132, 6.29732, 2.30159, 0.06603],[6.48889, 6.29511, 2.34287, 0.06507],[6.33661, 6.28126, 2.39067, 0.06396],[6.18533, 6.2562, 2.44567, 0.0627],[6.03574, 6.22045, 2.50861, 0.06131],[5.88843, 6.17462, 2.58049, 0.05978],[5.74387, 6.11934, 2.66252, 0.05813],[5.60239, 6.05528, 2.75623, 0.05635],[5.46422, 5.98311, 2.86367, 0.05443],[5.32948, 5.90353, 2.98758, 0.05238],[5.1982, 5.81721, 3.13148, 0.05018],[5.0703, 5.7248, 3.3004, 0.04781],[4.94565, 5.62697, 3.50151, 0.04525],[4.82405, 5.52435, 3.74549, 0.04248],[4.70524, 5.41757, 3.2647, 0.04893],[4.58889, 5.30726, 2.82931, 0.05667],[4.47468, 5.19401, 2.53161, 0.06353],[4.36221, 5.07844, 2.31165, 0.06976],[4.27749, 4.98931, 1.85, 0.06647],[4.19232, 4.90159, 1.85, 0.06608],[4.10628, 4.81661, 1.85, 0.06537],[4.01895, 4.73552, 1.85, 0.06442],[3.92991, 4.65932, 1.85, 0.06335],[3.83878, 4.58885, 1.85, 0.06227],[3.74516, 4.52486, 1.85, 0.0613],[3.64772, 4.47043, 1.90619, 0.05855],[3.54677, 4.42467, 1.97162, 0.05621],[3.44261, 4.38695, 2.05061, 0.05402],[3.33549, 4.35672, 2.14848, 0.05181],[3.22562, 4.33346, 2.27205, 0.04943],[3.11322, 4.31662, 2.43054, 0.04676],[2.99851, 4.30563, 2.6379, 0.04368],[2.88176, 4.2998, 2.91728, 0.04007],[2.76324, 4.29838, 3.1148, 0.03805],[2.6433, 4.30052, 2.77845, 0.04317],[2.52231, 4.30527, 2.55149, 0.04746],[2.40066, 4.31164, 2.39014, 0.05096],[2.24779, 4.31885, 2.27199, 0.06736],[2.0946, 4.32322, 2.18407, 0.07017],[1.94114, 4.32278, 2.11828, 0.07245],[1.78776, 4.31568, 2.06921, 0.0742],[1.63515, 4.30026, 2.03292, 0.07545],[1.48431, 4.27511, 2.0066, 0.07621],[1.3364, 4.23917, 1.98816, 0.07656],[1.19268, 4.1917, 1.97594, 0.0766],[1.0544, 4.1323, 1.96861, 0.07645],[0.9228, 4.06084, 1.96544, 0.07619],[0.7991, 3.97746, 1.96544, 0.0759],[0.68453, 3.88252, 1.96544, 0.0757],[0.5803, 3.77665, 1.96544, 0.07559],[0.48756, 3.66069, 1.96544, 0.07555],[0.40741, 3.53572, 1.96544, 0.07554],[0.3408, 3.40305, 1.96544, 0.07553],[0.28852, 3.26415, 1.96567, 0.0755],[0.25116, 3.12063, 1.96905, 0.07532],[0.22904, 2.97417, 1.97889, 0.07485]])

   
    ################## HELPER FUNCTIONS ###################

    def dist_2_points(x1, x2, y1, y2):

        return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5

    def closest_2_racingnpoints(x, y, racingtrack):
        # Convert to Numpy Array
        race_data = np.array(racingtrack)
        racecarxy = np.array([x, y]).reshape(1, -1)

        # Extract just the x,y coordinates from Racetradk

        r = (race_data[:, :2])
        # Get the closest point information
        distx = distance.cdist(racecarxy, r)
        point_index1 = distx.argmin()

        # Update the X,Y point to Eliminate from next Check
        r[point_index1] = [999, 999]

        # Get the Second Closest point Info
        distx1 = distance.cdist(racecarxy, r)
        point_index2 = distx1.argmin()

        return racingtrack[point_index1], racingtrack[point_index2], point_index1, point_index2

    def dist_to_racing_line(closest_coords, second_closest_coords, car_coords):

        d = np.cross(second_closest_coords-closest_coords, car_coords -
                     closest_coords)/np.linalg.norm(second_closest_coords-closest_coords)

        return abs(d)

     # Calculate which one of the closest racing points is the next one and which one the previous one

    def next_prev_racing_point(closest_coords, second_closest_coords, car_coords, heading):

        # Virtually set the car more into the heading direction
        heading_vector = [math.cos(math.radians(
            heading)), math.sin(math.radians(heading))]
        new_car_coords = [car_coords[0]+heading_vector[0],
                          car_coords[1]+heading_vector[1]]

        # Calculate distance from new car coords to 2 closest racing points
        distance_closest_coords_new = dist_2_points(
            x1=new_car_coords[0], x2=closest_coords[0], y1=new_car_coords[1], y2=closest_coords[1])
        distance_second_closest_coords_new = dist_2_points(
            x1=new_car_coords[0], x2=second_closest_coords[0], y1=new_car_coords[1], y2=second_closest_coords[1])

        if distance_closest_coords_new <= distance_second_closest_coords_new:
            next_point_coords = closest_coords
            prev_point_coords = second_closest_coords
        else:
            next_point_coords = second_closest_coords
            prev_point_coords = closest_coords

        return [next_point_coords, prev_point_coords]

    def racing_direction_diff(closest_coords, second_closest_coords, car_coords, heading):

        # Calculate the direction of the center line based on the closest waypoints
        next_point, prev_point = next_prev_racing_point(
            closest_coords, second_closest_coords, car_coords, heading)

        # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
        track_direction = math.atan2(
            next_point[1] - prev_point[1], next_point[0] - prev_point[0])

        # Convert to degree
        track_direction = math.degrees(track_direction)

        # Calculate the difference between the track direction and the heading direction of the car
        direction_diff = abs(track_direction - heading)
        if direction_diff > 180:
            direction_diff = 360 - direction_diff

        return direction_diff

    optimals, optimals_second, closest_index, second_closest_index = closest_2_racingnpoints(
        x, y, racing_track)

    ################ REWARD AND PUNISHMENT ################

    ## Define the default reward ##
    reward = 1

    ## Reward if car goes close to optimal racing line ##
    DISTANCE_MULTIPLE = 1
    closest_coordinates1 = np.array([optimals[0],optimals[1]])
    closest_coordinates2 = np.array([optimals_second[0],optimals_second[1]])
    dist = dist_to_racing_line(
        closest_coordinates1, closest_coordinates2, [x, y])
    distance_reward = max(1e-3, 1 - (dist))
    reward += distance_reward * DISTANCE_MULTIPLE

    ## Reward if speed is close to optimal speed ##
    SPEED_DIFF_NO_REWARD = 1
    SPEED_MULTIPLE = 2
    speed_diff = abs(optimals[2]-speed)
    if speed_diff <= SPEED_DIFF_NO_REWARD:
        # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
        # so, we do not punish small deviations from optimal speed
        speed_reward = (1 - (speed_diff/(SPEED_DIFF_NO_REWARD))**2)**2
    else:
        speed_reward = 0
    reward += speed_reward * SPEED_MULTIPLE

    # Zero reward if obviously wrong direction (e.g. spin)
    direction_diff = racing_direction_diff(
        optimals[0:2], optimals_second[0:2], [x, y], heading)
    if direction_diff > 30:
        reward = 1e-3

    # Zero reward of obviously too slow
    speed_diff_zero = optimals[2]-speed
    if speed_diff_zero > 0.5:
        reward = 1e-3

    ## Incentive for finishing the lap in less steps ##
    # should be adapted to track length and other rewards
    REWARD_FOR_FASTEST_TIME = 1500
    STANDARD_TIME = 12  # seconds (time that is easily done by model)
    FASTEST_TIME = 9  # seconds (best time of 1st place on the track)
    if progress == 100:
        finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                   (15*(STANDARD_TIME-FASTEST_TIME)))*(steps-STANDARD_TIME*15))
    else:
        finish_reward = 0
    reward += finish_reward

    if is_offtrack:
        reward = 1e-3

    return float(reward)

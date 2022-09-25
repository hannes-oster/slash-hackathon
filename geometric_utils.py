"""Some helper functions for geometric operations on pointclouds."""
import logging
from math import sqrt, ceil, floor, degrees, atan2
import math
import numpy as np
from skspatial.objects import Plane


def split_into_tiles(points, tile_size_x, tile_size_y):
    """Split the dataset into tiles in x-y-plane.

    This method copys some functionality provided in pcnn/preprocess.py density_reduce_3d-function

    Args:
        pointcloud_dataset (pandas.Dataframe): the dataframe containing all point data
        tile_size (float): the size of the tiles in meters

    Returns:
        ndarray: an array where the subarrays are the different tiles
    """
    points_xy = np.copy(points[:, :2])
    points_xy[:, 0] /= tile_size_x
    points_xy[:, 1] /= tile_size_y
    # assigns every point to a certain tile
    bins = np.floor(points_xy).astype(np.int32)

    # sorts the tiles (needed for np.unique)
    sorted_indices = np.lexsort(np.transpose(bins)[::-1])
    bins = bins[sorted_indices]
    points_sorted = points[sorted_indices]

    del sorted_indices
    # splits the points using an 1D-Array of indices. this 1D-array is created using the returned counts of np.unique.
    # for better understanding check the documentation of the specific np functions
    tiles = np.split(
        points_sorted, np.cumsum(
            np.unique(bins, return_counts=True, axis=0)[1])[:-1]
    )
    return tiles


def split_into_fixed_tiles(points, num_tiles):
    """splits the given points into a fixed number of tiles.

    Args:
        points (list): list containing all the points
        num_tiles (int): the number of tiles to split the points into. if num_tiles is not a square of an integer the next
        square number will be selected for num_tiles.

    Returns:
        list: the list containing num_tiles lists of points
    """
    points_bb = bounding_box(points)
    origin = (points_bb[0], points_bb[1])
    num_tiles = round(sqrt(num_tiles)) ** 2
    tiles_side_len = int(sqrt(num_tiles))
    logging.info(f"num tiles adjusted to {num_tiles}")
    x_tile_size = (points_bb[3] - points_bb[0]) / tiles_side_len
    y_tile_size = (points_bb[4] - points_bb[1]) / tiles_side_len

    tiles = []
    for _ in range(num_tiles):
        tiles.append([])
    for point in points:
        x_dist = point[0] - origin[0]
        y_dist = point[1] - origin[1]
        # logging.info(f"xdist: {x_dist}, x_ts: {x_tile_size}")

        x_idx = min(int(floor(x_dist / x_tile_size)), tiles_side_len - 1)
        y_idx = min(int(floor(y_dist / y_tile_size)), tiles_side_len - 1)
        tiles[y_idx * tiles_side_len + x_idx].append(point)

    return [np.array(tile) for tile in tiles]


"""
Adjustable variables for heuristic ground cell selection
"""

ground_level_deviation = 1.0  # optimizable
max_slope_inside_cell = 0.04  # optimizable
point_median_factor = 1.0  # optimizable


def probably_ground_tile(tile, probable_ground_level, tile_median):
    """Determine if tile is a ground tile based on a heuristic.

    Args:
        tile (np.array): array containing the points of the tile
        probable_ground_level (float): the probable height of the ground
        tile_density_media (float): the median of points in a tile

    Returns:
        bool: True if probably ground, False otherwise
    """
    min_height = np.min(tile[:, 2])
    max_height = np.max(tile[:, 2])

    # a tile is considered a ground tile if it does not differ too much from the probable ground level, if the height
    # difference inside the tile is not too big and if it contains enough points
    return (
        (min_height >= (probable_ground_level - ground_level_deviation))
        and (max_height <= (probable_ground_level + ground_level_deviation))
        and ((max_height - min_height) <= max_slope_inside_cell)
        and (len(tile) > point_median_factor * tile_median)
    )


def weaker_probably_ground_tile(tile, probable_ground_level):
    """Determine if tile qualifies as ground under a weaker condition. Only used if no tiles fullfilled the stronger
    condition

    Args:
        tile (np.array): array containing the points of the tile
        probable_ground_level (float): the guessed ground level

    Returns:
        bool: true if this tile is probably ground under a weaker condition.
    """
    min_height = np.min(tile[:, 2])
    max_height = np.max(tile[:, 2])
    return (
        (min_height >= (probable_ground_level - ground_level_deviation))
        and (max_height <= (probable_ground_level + ground_level_deviation))
        and ((max_height - min_height) <= max_slope_inside_cell)
    )


def even_weaker_probably_ground_tile(tile, probable_ground_level):
    """Determine if tile qualifies as ground under an yet even weaker condition. Only used if no tiles fullfilled the stronger
    conditions.

    Args:
        tile (np.array): array containing the points of the tile
        probable_ground_level (float): the guessed ground level

    Returns:
        bool: true if this tile is probably ground under an even weaker condition.
    """
    min_height = np.min(tile[:, 2])
    max_height = np.max(tile[:, 2])
    return (
        (min_height >= (probable_ground_level - ground_level_deviation))
        and (max_height <= (probable_ground_level + ground_level_deviation))
    )


def filter_weak_probable_ground_points(tiles, points):
    """filters the points that have a height in the lowest fifth of all point heights.

    Args:
        points (np.array): array containing the points

    Returns:
        np.array: the points in the lowest height fifth
    """

    index_of_lowest_five_percent = int(ceil(len(points) / 5.0))
    # index_of_lowest_five_percent = int(ceil(len(points) / 3.0))
    probable_ground_level = np.average(
        np.partition(points[:, 2], index_of_lowest_five_percent, axis=0)[
            :index_of_lowest_five_percent
        ]
    )
    ground_tiles = [tile for tile in tiles if weaker_probably_ground_tile(
        tile, probable_ground_level)]
    if not ground_tiles:
        logging.info(
            "found no ground tiles even under weaker condition, trying much weaker condition....")
        ground_tiles = [tile for tile in tiles if even_weaker_probably_ground_tile(
            tile, probable_ground_level)]

    return ground_tiles


def filter_probable_ground_tiles(tiles, all_points=None):
    """Determine all tiles that are probably ground tiles.

    Args:
        tiles (list): the list with the tiled points
        all_points (list): all points included in tiles (can also be omitted an recomputed in this function)

    Returns:
       list: array containing tiles which are probably gound tiles
    """
    if all_points is None:
        all_points = []
        for tile in tiles:
            all_points += tile.tolist()
    index_of_lowest_five_percent = int(ceil(0.05 * len(all_points)))
    probable_ground_level = np.average(
        np.partition(all_points[:, 2], index_of_lowest_five_percent, axis=0)[
            :index_of_lowest_five_percent
        ]
    )
    tile_median = np.median([len(tile) for tile in tiles])
    logging.debug(f"got avg tile density: {tile_median}")
    ground_tiles = [tile for tile in tiles if probably_ground_tile(
        tile, probable_ground_level, tile_median)]
    if not ground_tiles:
        logging.info(
            "found no ground tiles under strong condition, trying weaker condition....")
        ground_tiles = [tile for tile in tiles if weaker_probably_ground_tile(
            tile, probable_ground_level)]
    if not ground_tiles:
        logging.info(
            "found no ground tiles even under weaker condition, trying much weaker condition....")
        ground_tiles = [tile for tile in tiles if even_weaker_probably_ground_tile(
            tile, probable_ground_level)]

    return ground_tiles


def tile_density(tile, tile_size):
    """Calculate the density of points within this tile.

    Args:
        tile (np.array): array contaning all points within this tile
        tile_size (float): size of tile

    Returns:
        float: avg density of points
    """
    # maybe test out 2D-density vs 3D-density (for 3D, use max height as upper limit of bounding box)

    # now trying 2D-density
    return len(tile) / (tile_size ** 2)


def clamp(x, min_val, max_val):
    """Clamp the value of x to min_val and max_val.

    Args:
        x (float): the value to be clamped
        min_val (float): lower bound
        max_val (float): upper bound

    Returns:
        float: the clamped value
    """
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    return max(min_val, min(max_val, x))


def get_point_angle(point_a, point_b):
    """Calculate angle between points.

    Args:
        point_a (list): first point in format [x, y, z]
        point_b (list): second point in format [x, y, z]

    Returns:
        angle: angle between points
    """
    xydist = sqrt((point_a[0] - point_b[0]) ** 2 +
                  (point_a[1] - point_b[1]) ** 2)
    zdist = abs(point_a[2] - point_b[2])
    return 0 if xydist == 0 else degrees(atan2(zdist, xydist))


def pointcloud_to_array(pointcloud):
    """converts the points from the pointcloud object into an np array

    Args:
        pointcloud (PointCloud): datastructure to hold all the points

    Returns:
        np.array: the array containing the points
    """
    return pointcloud._dataset.to_numpy()


def pointcloud_bounding_box(pointcloud):
    """calculates the 2D bounding box of given pointcloud

    Args:
        pointcloud (Pointcloud): datastructure containing all points

    Returns:
        Tupe[float]: the bounding box in format (min_x, min_y, max_x, max_y)
    """

    all_points = pointcloud_to_array(pointcloud)
    return bounding_box(all_points)


def bounding_box(points):
    """calculates the 2D bounding box of given array of points

    Args:
        points (np.array): array containing all points

    Returns:
        Tupe[float]: the bounding box in format (min_x, min_y, max_x, max_y)
    """
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    min_z = np.min(points[:, 2])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])
    max_z = np.max(points[:, 2])
    return (min_x, min_y, min_z, max_x, max_y, max_z)


def bounding_box_2d_area(bounding_box):
    """returns the area of the given 2D bounding box

    Args:
        bounding_box (Tuple[float]): bounding box in format (min_x, min_y, max_x, max_y)

    Returns:
        float: the area of the given bounding box
    """
    return (bounding_box[3] - bounding_box[0]) * (bounding_box[4] - bounding_box[1])


def tiles_to_array(tiles):
    """unpacks the points contained in tiles into an 1D-Array containing all points.

    Args:
        tiles (list): list of tiles containing points

    Returns:
        np.array: 2D array of points (2nd dimension per point data)
    """
    return np.array([point for tile in tiles for point in tile])


def tile_center_point(tile, tile_size):
    """calculates an imaginary point in the center of the tile with average height of all points in tile as z value

    Args:
        tile (np.array): the tile to compute the center point of
        tile_size (float): the size of the tile

    Returns:
        np.array: an array containing x, y and z value of the center point
    """
    point_in_tile = tile[0]
    tile_origin_x = floor(point_in_tile[0] / tile_size) * tile_size
    tile_origin_y = floor(point_in_tile[1] / tile_size) * tile_size
    center_point_x = tile_origin_x + tile_size / 2
    center_point_y = tile_origin_y + tile_size / 2
    avg_height = np.average(tile[:, 2])
    return np.array([center_point_x, center_point_y, avg_height])


def arg_filter_numerical_outliers(values):
    """detects numerical outliers in the given list of numbers using the box plot statistical method. This function
    returns the indices of the elemtents of the given list, that are not outliers.

    Args:
        values (list): list of float numbers to compute outliers from

    Returns:
        list: indices of entries of given list that are !no! outliers
    """
    q1_percentile = 1  # optimizable
    q3_percentile = 99  # optimizable
    q1, q3 = np.percentile(sorted(values), [q1_percentile, q3_percentile])
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    filtered_indices = [idx for idx in range(
        len(values)) if values[idx] >= lower_bound and values[idx] <= upper_bound]
    logging.debug(f"removed {len(values) - len(filtered_indices)} outliers!")
    return filtered_indices


def height_values_of_pointcloud(pointcloud):
    """extracts and returns the z colum of the pointcloud

    Args:
        pointcloud (PointCloud): datastructure containing all de points

    Returns:
        np.array: the height values of all points in the given dataset
    """
    points = pointcloud._dataset.to_numpy()
    return points[:, 2]


def plane_from_points(points):
    """Uses skspatial for approximating the best fitting plane to the given points.

    Args:
        points (np.array): array containing the 3d points

    Returns:
        Plane: the best fitting plane
    """
    num_selected_points = 10000  # optimizable
    selected_point_indices = np.random.choice(len(points), num_selected_points)
    selected_points = [points[idx] for idx in selected_point_indices]
    return Plane.best_fit(selected_points)


def slope_in_bounding_box(plane, bounding_box):
    """computes the max height diff of the given plane inside the bounding box

    Args:
        plane (Plane): the plane in the bounding box
        bounding_box (list): the coordinates of the bounding box

    Returns:
        float: the maximum height diff of points on the plane inside the bounding box
    """

    lower_left = (bounding_box[0], bounding_box[1])
    lower_right = (bounding_box[0], bounding_box[4])
    upper_left = (bounding_box[3], bounding_box[1])
    upper_right = (bounding_box[3], bounding_box[4])

    projected_lower_left = plane.project_point([*lower_left, 0])
    projected_lower_right = plane.project_point([*lower_right, 0])
    projected_upper_left = plane.project_point([*upper_left, 0])
    projected_upper_right = plane.project_point([*upper_right, 0])

    max_height = max(projected_lower_left[2], projected_lower_right[2],
                     projected_upper_left[2], projected_upper_right[2])
    min_height = min(projected_lower_left[2], projected_lower_right[2],
                     projected_upper_left[2], projected_upper_right[2])

    return max_height - min_height


class Vector2D():
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], Vector2D):
                self.x = args[0].x
                self.y = args[0].y
            else:
                self.x = args[0][0]
                self.y = args[0][1]
        else:
            self.x = args[0]
            self.y = args[1]

    def __len__(self):
        return 2

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normed(self):
        return Vector2D(self.x / self.length(), self.y / self.length())

    def orth(self):
        return Vector2D(self.y, self.x * -1)

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, Vector2D):
            return Vector2D(self.x * other.x, self.y * other.y)
        else:
            return Vector2D(self.x * other, self.y * other)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def angle(self, other):
        return math.acos(self.dot(other) / (len(self) * len(other)))

    def rotated(self, angle):
        cosAng = math.cos(angle)
        sinAng = math.sin(angle)
        return Vector2D(cosAng * self.x - sinAng * self.y, sinAng * self.x + cosAng * self.y)

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        else:
            raise ValueError(f"{idx} out of reach for index of vector2d")

    def __str__(self):
        return f"Vector2D({self.x}, {self.y})"

    def __repr__(self) -> str:
        return self.__str__()


def squared_dist(pointA, pointB):
    dist = (pointA[0] - pointB[0])**2
    if len(pointA) >= 2:
        dist += (pointA[1] - pointB[1])**2
    if len(pointA) == 3:
        dist += (pointA[2] - pointB[2])**2
    return dist


def add_vec_to_point(point, vector):
    return point[0] + vector[0], point[1] + vector[1]


def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_statistics(array):
    return np.min(array), np.max(array), np.std(array), np.var(array)


def points_within_2D_box(array, box):
    in_box = (array[:, 0] >= box[0]) & (array[:, 0] <= box[1]) & (
        array[:, 1] >= box[2]) & (array[:, 1] <= box[3])
    return array[in_box]

def normalized_pixel_value(array, min_value = None, max_value = None, ignore_outliers = False):
    if min_value is None:
        min_value = array.min()
    if max_value is None:
        max_value = array.max()
    if ignore_outliers:
        min_value, max_value = np.percentile(array, [5, 95])
    values = (array - min_value) / (max_value - min_value)
    if ignore_outliers:
        values = np.clip(values, 0.0, 1.0)
    return (values * 255.0).astype(np.uint8)

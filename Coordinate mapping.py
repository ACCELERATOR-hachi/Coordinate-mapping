import numpy as np

def compute_transformation_lstsq(src_points, dst_points):
    """ 使用最小二乘法计算旋转和平移 """
    assert src_points.shape == dst_points.shape
    assert src_points.shape[1] == 2

    # 计算源点和目标点的均值
    src_center = np.mean(src_points, axis=0)
    dst_center = np.mean(dst_points, axis=0)

    # 中心化
    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center

    # 组合源点的坐标
    A = np.zeros((src_centered.shape[0], 3))
    A[:, 0] = src_centered[:, 0]
    A[:, 1] = -src_centered[:, 1]
    A[:, 2] = 1  # 常数项

    # 组合目标点的坐标
    b = dst_centered[:, 0]

    # 最小二乘拟合
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # 提取旋转角度和平移量
    theta = np.arctan2(params[1], params[0])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    t = dst_center - R @ src_center

    return R, t


def transform_point(point, R, t):
    """ 使用旋转矩阵 R 和平移向量 t 转换点 """
    return R @ point + t


def main():
    # 示例：已知的对应点
    src_points = np.array([[1, 0], [0, 1], [1, 2], [2, 2]])
    dst_points = np.array([[9, 4], [8, 3], [7, 4], [7, 5]])

    # 计算变换参数
    R, t = compute_transformation_lstsq(src_points, dst_points)

    # 变换源点
    transformed_points = np.array([transform_point(p, R, t) for p in src_points])

    # 输出旋转矩阵和平移向量
    print("Rotation Matrix R:\n", R)
    print("Translation Vector t:\n", t)

    # 接口：输入原坐标系的坐标
    input_point = input("请输入原坐标系坐标（格式：x,y）：")
    x, y = map(float, input_point.split(','))
    original_point = np.array([x, y])

    # 计算对应坐标
    transformed_point = transform_point(original_point, R, t)
    print(f"对应的目标坐标为：({transformed_point[0]:.2f}, {transformed_point[1]:.2f})")


if __name__ == "__main__":
    main()

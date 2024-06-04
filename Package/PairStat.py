import numpy as np
# from pykalman import KalmanFilter


def concat_two_str_list(list_1:list, list_2:list):
    # FIXME: list 안에 타입이 맞는지 체크하는 함수 넣어줘야함
    new_list = []
    for i in range(len(list_1)):
        new_list.append(list_1[i] + list_2[i])

    return new_list


# Simulation 및 모델 계산
def ar_series_generator(alpha, scale, size):
    series_list = []
    for i in range(size):
        if i > 0:
            e = alpha * series_list[-1] + np.random.normal(scale=scale)
            series_list.append(e)
        else:
            e = np.random.normal(scale=scale)
            series_list.append(e)

    return np.array(series_list)


def trend_generator(slope, size):
    trend_seq = np.linspace(0, size * slope, size)
    return trend_seq


def residual_generator(x, y):
    residual = y - np.matmul(x, np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)), np.matmul(np.transpose(x), y)))
    return residual


def get_ols_estimator(x, y):
    """
    공적분 관계에 있으면 추정량은 Super consistent하고 기존의 t-statistic은 루트n order에서 t-distribution을 따르지 않고 확률적으로 수렴한다.
    따라서 따로 t-stat에 대한 p value는 report 하지 않음.
    (하지 않아도 굉장히 낮게 나올것임, 실제 t distribution을 따르지 않기 때문에 의미 없는 값.)
    """
    estimates = np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)), np.matmul(np.transpose(x),y))

    return estimates


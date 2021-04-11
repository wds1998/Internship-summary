import numpy as np
import time

class Polygon_similarity(object):
    def __init__(self, filename1, filename2):
        self.source_f = open(filename2, "r")
        self.test_f = open(filename1, "r")

    def generate_data(self, filename):
        """
        读取文件中的坐标
        :param filename:文件名
        :return:坐标对的列表
        """
        with open(filename) as f:
            input = f.readline().split(" ")
            coordinates = []
            i = 0
            while i < len(input) - 3:
                coordinates.append(np.array([float(input[i]), float(input[i + 1])]))
                i = i + 2
        return coordinates

    def angle(self, vectorx, vectory):
        """
        计算两向量的夹角
        :param vectorx: 向量x
        :param vectory: 向量y
        :return:
        """
        Lx = np.sqrt(vectorx.dot(vectorx))
        Ly = np.sqrt(vectory.dot(vectory))
        cos_angle = vectorx.dot(vectory) / (Lx * Ly)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        return angle

    def is_bulge(self, vectorx, vectory):
        """
        使用向量叉乘判断是凸or凹
        :param vectorx: 向量x
        :param vectory: 向量y
        :return: 1 or -1
        """
        tmp = vectorx[0] * vectory[1] - vectorx[1] * vectory[0]
        if tmp >= 0:
            return 1
        else:
            return -1

    def L2_distance(self, vector):
        """
        两点之间L2距离
        :param vector: 两点组成的向量
        :return: 距离
        """
        tmp = (vector[0]) ** 2 + (vector[1]) ** 2
        return np.sqrt(tmp)

    def turn_fun(self, coordinates):
        """
        计算多边形的turning function
        :param filename: 文件名
        :return: turning function
        """
        vectors = []
        vectors.append(np.array([1.0, 0.0]))
        for i in range(1, len(coordinates)):
            vectors.append(
                np.array([coordinates[i][0] - coordinates[i - 1][0], coordinates[i][1] - coordinates[i - 1][1]]))
        vectors.append(np.array([coordinates[0][0] - coordinates[-1][0], coordinates[0][1] - coordinates[-1][1]]))
        result = []
        x = 0
        y = 0
        for i in range(len(vectors) - 1):
            y -= self.is_bulge(vectors[i], vectors[i + 1]) * self.angle(vectors[i], vectors[i + 1])
            # y += is_bulge(vectors[i], vectors[i + 1]) * angle(vectors[i], vectors[i + 1])
            result.append([x, round(y, 4)])
            x += self.L2_distance(vectors[i + 1])
        if result[-1][1] - result[0][1] < 0:
            result = []
            x = 0
            y = 0
            for i in range(len(vectors) - 1):
                y += self.is_bulge(vectors[i], vectors[i + 1]) * self.angle(vectors[i], vectors[i + 1])
                result.append([x, round(y, 4)])
                x += self.L2_distance(vectors[i + 1])
        for i in range(len(result)):
            result[i][0] = round(result[i][0] / x, 4)
        return result

    def theta(self, t, f, g):
        """
        theta在偏移t的时候的最优解
        :param t: 偏移量
        :param f: 多边形A的turning function
        :param g: 多边形B的turning function
        :return: theta在偏移t的时候的最优解
        """
        sum_f = 0.0
        sum_g = 0.0
        for i in range(1, len(f)):
            sum_f += (f[i][0] - f[i - 1][0]) * f[i - 1][1]
        sum_f += (1.0 - f[-1][0]) * f[-1][1]
        for i in range(1, len(g)):
            sum_g += (g[i][0] - g[i - 1][0]) * g[i - 1][1]
        sum_g += (1.0 - g[-1][0]) * g[-1][1]
        sum_g = round(sum_g, 4)
        sum_f = round(sum_f, 4)
        return sum_g - sum_f - 2 * np.pi * t

    def D_fun(self, f, g, theta):
        """
        f和g在某个极点的相似度
        :param f:多边形A的turning function
        :param g:多边形B的turning function
        :param theta: 偏移最优解
        :return: 相似度
        """
        x = []
        f_value = []
        g_value = []
        i = 0
        j = 0
        while i < len(f) and j < len(g):
            if f[i][0] == g[j][0]:
                x.append(f[i][0])
                f_value.append(f[i][1])
                g_value.append(g[j][1])
                i += 1
                j += 1
            elif f[i][0] < g[j][0]:
                x.append(f[i][0])
                f_value.append(f[i][1])
                g_value.append(g[j - 1][1])
                i += 1
            else:
                x.append(g[j][0])
                f_value.append(f[i - 1][1])
                g_value.append(g[j][1])
                j += 1
        if i == len(f) and j < len(g):
            for k in range(j, len(g)):
                g_value.append(g[k][1])
                f_value.append(f[-1][1])
                x.append(g[k][0])
        elif j == len(g) and i < len(f):
            for k in range(i, len(f)):
                g_value.append(f[k][1])
                f_value.append(g[-1][1])
                x.append(f[k][0])
        sum_d = 0.0
        for i in range(1, len(x)):
            sum_d += ((f_value[i - 1] - g_value[i - 1]) ** 2) * (x[i] - x[i - 1])
        sum_d += (f_value[-1] - g_value[-1]) ** 2 * (1.0 - x[-1])
        return np.sqrt(np.abs(sum_d - theta ** 2))

    def move(self, f, g, m, n):
        """
        f的第m个顶点和g的第n个顶点对齐
        :param f: 多边形A的turning function
        :param g: 多边形B的turning function
        :param m: f的第m个顶点
        :param n: g的第n个顶点
        :return: 移动后的f和移动距离
        """
        if f[m][0] == g[n][0]:
            return f, 0.0
        # 向右移动
        elif f[m][0] < g[n][0]:
            t = f[m][0] - g[n][0] + 1.0
            m_d = round(g[n][0] - f[m][0], 4)
            tmp1 = []
            tmp2 = []
            for i in range(len(f)):
                if f[i][0] + m_d < 1.0:
                    tmp1.append([round(f[i][0] + m_d, 4), f[i][1]])
                else:
                    tmp2.append([round(f[i][0] + m_d - 1.0, 4), f[i][1]])
            if len(tmp2) > 0 and tmp2[0][0] != 0.0:
                tmp2 = [[0.0, tmp1[-1][1]]] + tmp2

            for i in range(len(tmp1)):
                tmp1[i][1] += 6.2832
            tmp2 += tmp1
            return tmp2, t
        # 向左移动
        else:
            t = f[m][0] - g[n][0]
            m_d = round(g[n][0] - f[m][0], 4)
            tmp1 = []
            tmp2 = []
            for i in range(len(f)):
                if f[i][0] + m_d < 0:
                    tmp1.append([round(f[i][0] + m_d + 1.0, 4), f[i][1]])
                else:
                    tmp2.append([round(f[i][0] + m_d, 4), f[i][1]])
            if len(tmp2) > 0 and tmp2[0][0] != 0.0:
                tmp2 = [[0.0, tmp1[-1][1]]] + tmp2
            for i in range(len(tmp1)):
                tmp1[i][1] += 6.2832
            tmp2 += tmp1
            return tmp2, t

    def takeSecond(self,elem):
        return elem[1]

    def turning_dist(self):
        test_input = []
        source_input = []
        test_num = 1
        start = time.time()
        while True:
            test_input = self.test_f.readline().split(" ")
            coordinates1 = []
            i = 0
            if len(test_input) < 2:
                break
            while i < len(test_input) - 4:
                coordinates1.append(np.array([float(test_input[i]), float(test_input[i + 1])]))
                i = i + 2
            f = self.turn_fun(coordinates1)
            result = []
            source_num = 1
            while True:
                source_input = self.source_f.readline().split(" ")
                coordinates2 = []
                j = 0
                if len(source_input) < 2:
                    break
                while j < len(source_input) - 4:
                    coordinates2.append(np.array([float(source_input[j]), float(source_input[j + 1])]))
                    j = j + 2
                g = self.turn_fun(coordinates2)
                min_distance = float('inf')
                theta_res = 0.0
                t_res = 0.0
                for i in range(len(f)):
                    for j in range(len(g)):
                        f_2, t = self.move(f, g, i, j)
                        theta_star = self.theta(t, f, g)
                        distance = self.D_fun(f_2, g, theta_star)
                        if distance < min_distance:
                            min_distance = distance
                            theta_res = theta_star
                            t_res = t
                result.append([source_num, min_distance])
                source_num += 1
            self.source_f = open("source.data")
            result.sort(key = self.takeSecond)
            for i in range(9):
                print("test.data中第"+str(test_num)+"个地基的top"+str(i)+"为source.data中第"+str(result[i][0])+"个数据,相似度为"+str(result[i][1]))
            test_num += 1
        print("total time:"+str(time.time()-start))


P = Polygon_similarity("test.data", "source.data")
P.turning_dist()



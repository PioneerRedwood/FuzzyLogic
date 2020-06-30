# 2020.6.29, 30 퍼지 이론
# 인공지능 마지막 과제

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


def FuzzyEnvironment():
    pass


temperature = np.arange(28, 41, 1)

temp_low = fuzz.trimf(temperature, [29, 31, 33])
temp_mid = fuzz.trimf(temperature, [31, 33, 35])
temp_high = fuzz.trimf(temperature, [33, 35, 38])

humidity = np.arange(20, 101, 8)
humidity_low = fuzz.trimf(humidity, [24, 40, 56])
humidity_mid = fuzz.trimf(humidity, [48, 64, 80])
humidity_high = fuzz.trimf(humidity, [72, 100, 100])

rest = np.arange(0, 40, 5)
rest_low = fuzz.trimf(rest, [0, 0, 10])
rest_mid = fuzz.trimf(rest, [0, 10, 20])
rest_high = fuzz.trimf(rest, [10, 20, 30])
rest_extreme_high = fuzz.trimf(rest, [30, 40, 40])

fig, (graph0, graph1, graph2) = plt.subplots(nrows=3, figsize=(9, 6))

graph0.plot(temperature, temp_low, linewidth=1.5, label='attention')
graph0.plot(temperature, temp_mid, linewidth=1.5, label='caution')
graph0.plot(temperature, temp_high, linewidth=1.5, label='alert')
graph0.set_title('Temperature')
graph0.legend()

graph1.plot(humidity, humidity_low, linewidth=1.5, label='optimum')
graph1.plot(humidity, humidity_mid, linewidth=1.5, label='moist')
graph1.plot(humidity, humidity_high, linewidth=1.5, label='too moist')
graph1.set_title('Humidity')
graph1.legend()

graph2.plot(rest, rest_low, linewidth=1.5, label='low')
graph2.plot(rest, rest_mid, linewidth=1.5, label='medium')
graph2.plot(rest, rest_high, linewidth=1.5, label='high')
graph2.plot(rest, rest_extreme_high, linewidth=1.5, label='extremely high')
graph2.set_title('Break Time')
graph2.legend()

for graph in (graph0, graph1, graph2):
    graph.spines['top'].set_visible(False)
    graph.spines['right'].set_visible(False)
    graph.get_xaxis().tick_bottom()
    graph.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

temp_level_low = fuzz.interp_membership(temperature, temp_low, 33.2)
temp_level_mid = fuzz.interp_membership(temperature, temp_mid, 33.2)
temp_level_high = fuzz.interp_membership(temperature, temp_high, 33.2)

humidity_level_low = fuzz.interp_membership(humidity, humidity_low, 20.0)
humidity_level_mid = fuzz.interp_membership(humidity, humidity_mid, 20.0)
humidity_level_high = fuzz.interp_membership(humidity, humidity_high, 20.0)

# 규칙은 아직 지정 안함
active_rule1 = np.fmax(temp_level_low, humidity_level_low)
rest_activation_low = np.fmin(active_rule1, rest_low)

rest0 = np.zeros_like(rest)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(rest, rest0, rest_activation_low, facecolor='b', alpha=0.7)
ax0.plot(rest, rest_low, 'b', linewidth=0.5, linestyle='--', )
ax0.set_title('Output')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


def FuzzyFunc(picked_temperature, picked_humidity):
    temperature = np.arange(20, 35, 1)

    temp_low = fuzz.trimf(temperature, [20, 23, 26])
    temp_mid = fuzz.trimf(temperature, [23, 26, 29])
    temp_high = fuzz.trimf(temperature, [26, 29, 32])

    humidity = np.arange(35, 96, 1)
    humidity_low = fuzz.trimf(humidity, [35, 35, 60])
    humidity_mid = fuzz.trimf(humidity, [35, 60, 96])
    humidity_high = fuzz.trimf(humidity, [60, 96, 96])

    rest = np.arange(0, 26, 1)
    rest_low = fuzz.trimf(rest, [0, 0, 13])
    rest_mid = fuzz.trimf(rest, [0, 13, 25])
    rest_high = fuzz.trimf(rest, [13, 25, 25])

    fig, (graph0, graph1, graph2) = plt.subplots(nrows=3, figsize=(7, 6))

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
    graph2.set_title('Break Time')
    graph2.legend()

    for graph in (graph0, graph1, graph2):
        graph.spines['top'].set_visible(False)
        graph.spines['right'].set_visible(False)
        graph.get_xaxis().tick_bottom()
        graph.get_yaxis().tick_left()

    plt.tight_layout()

    temp_level_low = fuzz.interp_membership(temperature, temp_low, picked_temperature)
    temp_level_mid = fuzz.interp_membership(temperature, temp_mid, picked_temperature)
    temp_level_high = fuzz.interp_membership(temperature, temp_high, picked_temperature)

    humidity_level_low = fuzz.interp_membership(humidity, humidity_low, picked_humidity)
    humidity_level_mid = fuzz.interp_membership(humidity, humidity_mid, picked_humidity)
    humidity_level_high = fuzz.interp_membership(humidity, humidity_high, picked_humidity)

    # Fuzzy AND Operation
    # break time set low rules #1~3
    act_rule1 = np.fmin(temp_level_low, humidity_level_low)
    rest_act_low1 = np.fmin(act_rule1, rest_low)

    act_rule2 = np.fmin(temp_level_low, humidity_level_mid)
    rest_act_low2 = np.fmin(act_rule2, rest_low)

    act_rule3 = np.fmin(temp_level_low, humidity_level_high)
    rest_act_low3 = np.fmin(act_rule3, rest_mid)

    # break time set mid rules #4~6
    act_rule4 = np.fmin(temp_level_mid, humidity_level_low)
    rest_act_mid1 = np.fmin(act_rule4, rest_low)

    act_rule5 = np.fmin(temp_level_mid, humidity_level_mid)
    rest_act_mid2 = np.fmin(act_rule5, rest_mid)

    act_rule6 = np.fmin(temp_level_mid, humidity_level_high)
    rest_act_mid3 = np.fmin(act_rule6, rest_high)

    # break time set high rules #7~9
    act_rule7 = np.fmin(temp_level_high, humidity_level_low)
    rest_act_high1 = np.fmin(act_rule7, rest_mid)

    act_rule8 = np.fmin(temp_level_high, humidity_level_mid)
    rest_act_high2 = np.fmin(act_rule8, rest_high)

    act_rule9 = np.fmin(temp_level_high, humidity_level_high)
    rest_act_high3 = np.fmin(act_rule9, rest_high)
    rest0 = np.zeros_like(rest)

    fig, ax0 = plt.subplots(figsize=(8, 3))

    # fill_between(x 축, 집합1 혹은 스칼라, 집합2 혹은 스칼라)
    ax0.fill_between(rest, rest0, rest_act_low1, color=[0.4, 0.2, 0.2, 0.3])
    ax0.fill_between(rest, rest0, rest_act_low2, color=[0.6, 0.2, 0.2, 0.3])
    ax0.fill_between(rest, rest0, rest_act_low3, color=[0.8, 0.2, 0.2, 0.3])
    ax0.plot(rest, rest_low, 'r', linewidth=0.5, linestyle='--')

    ax0.fill_between(rest, rest0, rest_act_mid1, color=[0.2, 0.4, 0.2, 0.3])
    ax0.fill_between(rest, rest0, rest_act_mid2, color=[0.2, 0.6, 0.2, 0.3])
    ax0.fill_between(rest, rest0, rest_act_mid3, color=[0.2, 0.8, 0.2, 0.3])
    ax0.plot(rest, rest_mid, 'g', linewidth=0.5, linestyle='--')

    ax0.fill_between(rest, rest0, rest_act_high1, color=[0.2, 0.2, 0.4, 0.3])
    ax0.fill_between(rest, rest0, rest_act_high2, color=[0.2, 0.2, 0.6, 0.3])
    ax0.fill_between(rest, rest0, rest_act_high3, color=[0.2, 0.2, 0.8, 0.3])
    ax0.plot(rest, rest_high, 'b', linewidth=0.5, linestyle='--')

    # 멤버십 함수 출력
    ax0.set_title('Output membership activity')

    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Fuzzy OR Operation
    # total all output membership
    total = np.fmax(rest_act_low1, np.fmax(rest_act_low2, np.fmax(rest_act_low3,
                    np.fmax(rest_act_mid1, np.fmax(rest_act_mid2, np.fmax(rest_act_mid3,
                    np.fmax(rest_act_high1, np.fmax(rest_act_high2, rest_act_high3))))))))

    # defuzzified
    break_time = fuzz.defuzz(rest, total, 'centroid')
    break_time_act = fuzz.interp_membership(rest, total, break_time)

    # visualize
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(rest, rest_low, 'r', linewidth=0.5, linestyle='--')
    ax0.plot(rest, rest_mid, 'g', linewidth=0.5, linestyle='--')
    ax0.plot(rest, rest_high, 'b', linewidth=0.5, linestyle='--')
    ax0.fill_between(rest, rest0, total, facecolor='Black', alpha=0.7)
    ax0.plot([break_time, break_time], [0, break_time_act], 'w', linewidth=2)
    ax0.set_title('Total break time and result(white line)')

    print(break_time)

    for ax in (ax0, ):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()
    pass

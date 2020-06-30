# Fuzzy Theory 퍼지 이론
# 퍼지 집합, 진리도과 소속도, 소속 함수
# 퍼지화
# 퍼지추론 규칙 기반, and / or
# 역퍼지화 무게 중심법
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FuzzyTheory.Environment


def main():
    fig, (graph) = plt.subplots(figsize=(10, 7))
    _ = (
        pd.read_csv('STCS_190701_190831.csv', sep=',',
                    usecols=['date', 'temp_c', 'humidity'],
                    parse_dates=['date']).set_index('date').plot(ax=graph, picker=10)
    )
    graph.set_xlabel('Date')
    graph.set_ylabel('Temperature_red, Humidity_blue')
    graph.set_title('190701 ~ 190831')
    fig.canvas.callbacks.connect('pick_event', onPickItem)
    plt.show()


# 그래프 클릭 이벤트 처리 가능
def onPickItem(event):
    pick_artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x_idx, y_idx = pick_artist.get_xdata(), pick_artist.get_ydata()
    index = event.ind

    print('Artist picked:', event.artist)
    print('{} vertices picked'.format(len(index)))
    print('Pick between vertices {} and {}'.format(min(index), max(index) + 1))
    print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
    print('Data point', x_idx[index[0]], y_idx[index[0]])


if __name__ == "__main__":
    print("welcome")
    main()

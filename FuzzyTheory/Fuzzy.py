import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FuzzyTheory.Environment


def main():
    fig, (graph) = plt.subplots(figsize=(10, 7))
    _ = (
        pd.read_csv('STCS_190701_190831.csv', sep=',',
                    usecols=['date', 'temp_c', 'humidity'],
                    parse_dates=['date']).set_index('date').plot(ax=graph, picker=True)
    )
    graph.set_xlabel('Date')
    graph.set_ylabel('Temperature, Humidity')
    graph.set_title('190701 ~ 190831')
    fig.canvas.callbacks.connect('pick_event', onPickItem)
    plt.show()


# 그래프 클릭 이벤트 처리 가능
def onPickItem(event):
    pick_artist = event.artist
    x_idx = pick_artist.get_xdata()
    index = event.ind

    picked_date = pd.Period(x_idx[index[0]])
    date_int = int((picked_date.strftime('%Y%m%d')))

    file = np.loadtxt('STCS_190701_190831_forNP.csv', delimiter=",")
    date_array = np.array(file[:, :1]).reshape(-1)

    select_idx = np.where(date_array == date_int)[0][0]
    selected_tuple = file[select_idx, :]
    FuzzyTheory.Environment.FuzzyFunc(selected_tuple[1], selected_tuple[3])


if __name__ == "__main__":
    print("welcome")
    main()

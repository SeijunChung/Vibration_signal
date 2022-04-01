import json
import glob
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy.spatial import distance
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


def get_euclidean_distance(normal, abnormal):
    return distance.euclidean(np.array(normal), np.array(abnormal))


def get_cosine_similarity(normal, abnormal):
    return np.dot(np.array(normal), np.array(abnormal)) / (norm(np.array(normal)) * norm(np.array(abnormal)))


def get_arg_min(mat, how_many=50):
    return np.dstack(np.unravel_index(np.argsort(mat, axis=None), mat.shape))[0, :how_many, :]


def get_arg_max(mat, how_many=50):
    return np.dstack(np.unravel_index(np.argsort(mat, axis=None), mat.shape))[0, -how_many:, :]


def get_fft_attritubes(signal, freq=10544):
    N = len(signal)
    T = N / freq
    k = np.arange(N) / T
    ffted = np.fft.fft(signal) / N
    ran = range(int(N/2))

    return {"k": k[ran], "signal_fft": abs(ffted[ran])}


def get_figure_plotly_1channel(normal_td, abnormal_td, normal_freq, abnormal_freq, label, dist):
    fig = make_subplots(
        rows=2, cols=2,
        start_cell="top-left", subplot_titles=("Time-domain", "Frequency-domain"), shared_yaxes=False)

    color = "blue"

    fig.add_trace(go.Scatter(x=[value/10544 for value in range(len(normal_td))], y=normal_td, name="signal_time_domain",
                             marker_color=color), row=1, col=1)
    fig.update_xaxes(title_text="Time (Seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, len(normal_freq["k"]))),
                             y=normal_freq["signal_fft"][1:], name="fft", marker_color=color), row=1, col=2)
    fig.update_xaxes(title_text="Freq (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="|Y(freq)|", row=1, col=2)

    color = "red"
    fig.add_trace(go.Scatter(x=[value/10544 for value in range(len(abnormal_td))], y=abnormal_td, name="signal_time_domain",
                             marker_color=color), row=2, col=1)
    fig.update_xaxes(title_text="Time (Seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, len(abnormal_freq["k"]))),
                             y=abnormal_freq["signal_fft"][1:], name="fft", marker_color=color), row=2, col=2)
    fig.update_xaxes(title_text="Freq (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="|Y(freq)|", row=2, col=2)

    fig.update_layout(title_text="Data Exploration")
    fig.write_html(f"../res/3rd/long_term_comparison/normal_vs_{label}_distance_{dist}.html")


def get_figure_plotly(sample_normal, sample_abnormal, sr, sequence_normal, sequence_abnormal, length, label=None, freq=10544):
    fig = make_subplots(
        rows=3, cols=2,
        start_cell="top-left",
        subplot_titles=("Normal", "Abnormal"),
        shared_xaxes=True, vertical_spacing=0.02, horizontal_spacing=0.04
    )

    for j in range(1, 3):
        for i in range(1, 4):
            data = sample_normal if j == 1 else sample_abnormal
            # color = "blue" if j == 1 else "red"
            color = 'black'
            Y = get_fft_attritubes(data[f"channel{i}"][:sr])

            fig.add_trace(go.Scatter(x=[value/freq for value in range(1, sr+1)],
                                     y=data[f"channel{i}"],
                                     name=f"channel{i}-signal",
                                     marker_color=color), row=i, col=j)
            if i == 3:
                fig.update_xaxes(title_text="Time", row=i, col=j)
            if j == 1:
                fig.update_yaxes(title_text=f"channel{i}\n", row=i, col=j)

            # fig.add_trace(go.Scatter(x=Y["k"][1:], y=Y["signal_fft"][1:],
            #                          name=f"channel{i}-fft",
            #                          marker_color=color), row=i, col=2*j)
            # fig.update_xaxes(title_text="Freq (Hz)", row=i, col=2 * j)
            # fig.update_yaxes(title_text="|Y(freq)|", row=i, col=2 * j)

    fig.update_layout(showlegend=False, font_size=15)
    fig.write_html(f"../res/3rd/short_term/{length}/sample_at_normal_{sequence_normal}s_abnormal_{sequence_abnormal}s.html")


if __name__ == "__main__":

    np.random.seed(0)
    x = np.arange(12)
    y = np.random.rand(len(x)) * 51
    c = np.random.rand(len(x)) * 3 + 1.5

    df = pd.DataFrame({"x": x, "y": y, "c": c})

    cmap = plt.cm.bwr
    norm = matplotlib.colors.Normalize(vmin=1, vmax=5)
    print(norm(df.c.values))

    fig, ax = plt.subplots()
    ax.bar(df.x, df.y, color=cmap(norm(df.c.values)))
    ax.set_xticks(df.x)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # only needed for matplotlib < 3.1
    fig.colorbar(sm)

    plt.show()
    quit()

    history = [108.5, 107.7, 106.8, 107.2, 106.2, 105.5, 105.0, 105.6, 105.2, 104.3, 104.9, 104.8,
               104.1, 103.1, 103.2, 102.8, 103.6, 103.7, 103.2, 101.9, 102.1, 101.2, 101.0, 100.7,
               100.9, 100.8, 99.2, 98.8, 98.5, 98.3, 99.1, 98.7, 98.6, 97.9, 97.2, 97.1, 96.3, 96.2,
               96.5, 96.9, 95.4, 95.0, 94.4, 94.0]

    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.grid()
    plt.show()
    quit()


    sample_rate = 10544

    # Normal
    flist_normal = glob.glob("./data/1st_Normal/*.json")  # 1st_Normal
    # flist_normal = glob.glob("../data/original/3rd_Normal/*.json")  # 3rd Normal

    # Abnormal
    # flist_abnormal = glob.glob("../data/original/3rd_Unbalance/*.json")
    flist_unbalance = glob.glob("./data/1st_Unbalance/*.json")  # 1st_Unbalance
    flist_looseness = glob.glob("./data/1st_Looseness/*.json") # 1st_Looseness
    flist_looseness_high = glob.glob("./data/1st_Looseness_high/*.json")  # 1st_Looseness_high
    flist_bearing = glob.glob("./data/1st_Bearing/*.json")  # 1st_Bearing

    print(f"Normal: {len(flist_normal)}")
    print(f"Unbalance: {len(flist_unbalance)}, Looseness: {len(flist_looseness)}, Looseness_high: {len(flist_looseness_high)}, Bearing: {len(flist_bearing)}")

    # get empty array
    # distance_between_1st = dict()
    # cosine_similarity_1st = dict()

    classes = ["Unbalance", "Looseness", "Looseness_high", "Bearing"]  # classes in Dataset 1

    start_idx = 850
    end_idx = 1450
    n = end_idx - start_idx
    # distance_between_3rd = np.zeros((n, n), dtype=float)
    # cosine_similarity_3rd = np.zeros((n, n), dtype=float)

    # 3rd Dataset
    # for i in range(start_idx, end_idx):
    #     for j in range(start_idx, end_idx):
    #         with open(flist_normal[i], "r") as fp_n:
    #             sample_data_normal = json.load(fp_n)
    #         with open(flist_abnormal[j], "r") as fp_ab:
    #             sample_data_abnormal = json.load(fp_ab)
    #         d = get_euclidean_distance(sample_data_normal["channel1"], sample_data_abnormal["channel1"])
    #         s = get_cosine_similarity(sample_data_normal["channel1"], sample_data_abnormal["channel1"])
    #         if i % 100 == 0 and j % 100 == 0:
    #             print(f"Normal {i} - Abnormal {j} is proceeding ~ distance {d}, cos_sim {s}")
    #         distance_between_3rd[i-start_idx, j - start_idx] = d
    #         cosine_similarity_3rd[i - start_idx, j - start_idx] = s
    #
    # with open(f"../res/3rd/1s_distance_between_3rd_{n}.pickle", "wb") as f:
    #     pickle.dump(distance_between_3rd, f, pickle.HIGHEST_PROTOCOL)
    # with open(f"../res/3rd/1s_cosine_similarity_3rd_{n}.pickle", "wb") as f:
    #     pickle.dump(cosine_similarity_3rd, f, pickle.HIGHEST_PROTOCOL)

    with open(f"../res/3rd/1s_distance_between_3rd_{n}.pickle", "rb") as f:
        d1 = pickle.load(f)
    with open(f"../res/3rd/1s_cosine_similarity_3rd_{n}.pickle", "rb") as f:
        s1 = pickle.load(f)

    for ind_nor, ind_ab in get_arg_min(d1):
        with open(flist_normal[start_idx+ind_nor], "r") as fp:
            sample_data_normal = json.load(fp)
        with open(flist_normal[start_idx+ind_ab], "r") as fp:
            sample_data_abnormal = json.load(fp)
        get_figure_plotly(sample_data_normal, sample_data_abnormal, 2000, start_idx+ind_nor, start_idx+ind_ab, n, "Unbalance")

    quit()

    # 1st Dataset
    # for cla in classes:
    #     flist_abnormal = glob.glob(f"../data/original/1st_{cla}/*.json")
    #     dist = np.zeros((n, n), dtype=float)
    #     cosine_similarity = np.zeros((n, n), dtype=float)
    #
    #     for i in range(start_idx, end_idx):
    #         for j in range(start_idx, end_idx):
    #             with open(flist_normal[i], "r") as fp_n:
    #                 sample_data_normal = json.load(fp_n)
    #             with open(flist_abnormal[j], "r") as fp_ab:
    #                 sample_data_abnormal = json.load(fp_ab)
    #             d = get_euclidean_distance(sample_data_normal["channel1"], sample_data_abnormal["channel1"])
    #             s = get_cosine_similarity(sample_data_normal["channel1"], sample_data_abnormal["channel1"])
    #             if i % 100 == 0:
    #                 print(f"In {cla}: Normal {i} - Abnormal {j} is proceeding ~ distance {d}, cos_sim {s}")
    #             dist[i-start_idx, j - start_idx] = d
    #             cosine_similarity[i - start_idx, j - start_idx] = s
    #
    #     distance_between_1st.update({cla: dist})
    #     cosine_similarity_1st.update({cla: cosine_similarity})
    #     print(f"In {cla}:", distance_between_1st[cla])

    # with open(f"../res/1st/1s_distance_between_1st_{n}.pickle", "wb") as f:
    #     pickle.dump(distance_between_1st, f, pickle.HIGHEST_PROTOCOL)
    # with open(f"../res/1st/1s_cosine_similarity_1st_{n}.pickle", "wb") as f:
    #     pickle.dump(cosine_similarity_1st, f, pickle.HIGHEST_PROTOCOL)

    # with open(f"../res/1st/1s_distance_between_1st_{n}.pickle", "rb") as f:
    #     d1 = pickle.load(f)
    # with open(f"../res/1st/1s_cosine_similarity_1st_{n}.pickle", "rb") as f:
    #     s1 = pickle.load(f)

    # for cla in classes:
    #     print("mean distance unbalance:", np.mean(d1[cla]))
    #     print("min distance unbalance:", np.min(d1[cla]), get_arg_min(d1[cla], 1))
    #     for ind_nor, ind_ab in get_arg_min(d1[cla]):
    #         with open(flist_normal[start_idx+ind_nor], "r") as fp:
    #             sample_data_normal = json.load(fp)
    #         with open(flist_normal[start_idx+ind_ab], "r") as fp:
    #             sample_data_abnormal = json.load(fp)
    #         get_figure_plotly(sample_data_normal, sample_data_abnormal, 2000, start_idx+ind_nor, start_idx+ind_ab, cla)

    for cla in classes:
        fft_distance = []
        fft_cos_sim = []
        flist_abnormal = glob.glob(f"../data/original/1st_{cla}/*.json")
        print(f"{cla} : {np.where(d1[cla] == np.min(d1[cla]))}")
        print(f"{cla} : {get_arg_min(d1[cla])[:10]}")
        print(f"{cla} : {np.where(s1[cla] == np.max(s1[cla]))}")
        print(f"{cla} : {get_arg_max(s1[cla])[:10]}")

        for idx_nor, idx_abn in get_arg_min(d1[cla]):
            with open(flist_normal[start_idx + idx_nor], "r") as fp:
                sample_data_normal = json.load(fp)
                normal_fft = get_fft_attritubes(sample_data_normal["channel1"])
            with open(flist_normal[start_idx + idx_abn], "r") as fp:
                sample_data_abnormal = json.load(fp)
                abnormal_fft = get_fft_attritubes(sample_data_abnormal["channel1"])
            fft_cos_sim.append(get_cosine_similarity(normal_fft["signal_fft"], abnormal_fft["signal_fft"]))
            fft_distance.append(get_euclidean_distance(normal_fft["signal_fft"], abnormal_fft["signal_fft"]))
        print(f"{cla} : fft_distance min = {min(fft_distance)} at {fft_distance.index(min(fft_distance))} 번째")
        print(f"{cla} : fft_cos_sim = {max(fft_cos_sim)} at {fft_cos_sim.index(max(fft_cos_sim))} 번째")

    # for cla in classes:
    #     flist_abnormal = glob.glob(f"../data/original/1st_{cla}/*.json")
    #     dist = np.zeros((len(flist_normal) // 100, len(flist_abnormal) // 100), dtype=float)
    #     cosine_similarity = np.zeros((len(flist_normal) // 100, len(flist_abnormal) // 100), dtype=float)
    #     for i in range(n_files, len(flist_normal), n_files):
    #         normal = []
    #         for n in flist_normal[i - n_files:i]:
    #             with open(n, "r") as fp:
    #                 sample_data_normal = json.load(fp)
    #             normal.extend(sample_data_normal["channel1"])
    #         for j in range(n_files, len(flist_abnormal), n_files):
    #             abnormal = []
    #             for ab in flist_abnormal[j - n_files:j]:
    #                 with open(ab, "r") as fp:
    #                     sample_data_abnormal = json.load(fp)
    #                 abnormal.extend(sample_data_abnormal["channel1"])
    #             d = get_euclidean_distance(normal, abnormal)
    #             s = get_cosine_similarity(normal, abnormal)
    #             print(f"distance between {i} and {j} is {d}")
    #             dist[i//n_files-1, j//n_files-1] = d
    #             cosine_similarity[i//n_files-1, j//n_files-1] = s
    #             # distance_between_3rd[i//n_files-1, j//n_files-1] = d
    #             # cosine_similarity_3rd[i//n_files-1, j//n_files-1] = s
    #     distance_between_1st.update({cla: dist})
    #     cosine_similarity_1st.update({cla: cosine_similarity})

    # with open("../res/1st/long_term_distance_between_1st.pickle", "wb") as f:
    #     pickle.dump(distance_between_1st, f, pickle.HIGHEST_PROTOCOL)
    # with open("../res/1st/long_term_cosine_similarity_1st.pickle", "wb") as f:
    #     pickle.dump(cosine_similarity_1st, f, pickle.HIGHEST_PROTOCOL)

    with open("../res/1st/long_term_distance_between_1st.pickle", "rb") as f:
        d1 = pickle.load(f)
    with open("../res/1st/long_term_cosine_similarity_1st.pickle", "rb") as f:
        s1 = pickle.load(f)

    # Cosine similarity 구하기....

    for cla in classes:
        normal = []
        abnormal = []
        flist_abnormal = glob.glob(f"../data/original/1st_{cla}/*.json")
        ind_nor, ind_ab = get_arg_min(d1[cla])[0]
        for n in flist_normal[ind_nor*n_files:(ind_nor+1)*n_files]:
            with open(n, "r") as fp:
                sample_data_normal = json.load(fp)
            normal.extend(sample_data_normal["channel1"])
        for ab in flist_abnormal[ind_ab*n_files:(ind_ab+1)*n_files]:
            with open(ab, "r") as fp:
                sample_data_abnormal = json.load(fp)
            abnormal.extend(sample_data_abnormal["channel1"])

        print(ind_nor, ind_ab)
        print(np.min(d1[cla]))

        print(f"{cla}:", get_euclidean_distance(normal, abnormal))
        print(f"{cla}:", get_cosine_similarity(normal, abnormal))

    # for i, nor in enumerate(flist_normal):
    #     for j, ab in enumerate(flist_abnormal):
    #         with open(nor, "r") as fp:
    #             sample_data_normal = json.load(fp)
    #         with open(ab, "r") as fp:
    #             sample_data_abnormal = json.load(fp)
    #         d = get_euclidean_distance(sample_data_normal["channel1"], sample_data_abnormal["channel1"])
    #         s = get_cosine_similarity(sample_data_normal["channel1"], sample_data_abnormal["channel1"])
    #         print(f"distance between {i} and {j} is {d}")
    #         distance_between_3rd[i, j] = d
    #         cosine_similarity_3rd[i, j] = s

    # with open("../res/3rd/long_term_distance_between_3rd.pickle", "wb") as f:
    #     pickle.dump(distance_between_3rd, f, pickle.HIGHEST_PROTOCOL)
    # with open("../res/3rd/long_term_cosine_similarity_3rd.pickle", "wb") as f:
    #     pickle.dump(cosine_similarity_3rd, f, pickle.HIGHEST_PROTOCOL)
    # with open("../res/3rd/long_term_distance_between_3rd.pickle", "rb") as f:
    #     d3 = pickle.load(f)
    # with open("../res/3rd/long_term_cosine_similarity_3rd.pickle", "rb") as f:
    #     s3 = pickle.load(f)

    # TODO : Euclidean Distance between normal-fft and abnormal-fft

    distance_between_ffts = []
    cosine_similarity_ffts = []

    for i, (nor, ab) in enumerate(sorted):
        normal = []
        abnormal = []
        for n in flist_normal[nor*n_files:(nor+1)*n_files]:
            with open(n, "r") as fp:
                sample_data_normal = json.load(fp)
            normal.extend(sample_data_normal["channel1"])
        normal_fft = get_fft_attritubes(normal, "channel1", sample_rate)

        for abn in flist_abnormal[ab*n_files:(ab+1)*n_files]:
            with open(abn, "r") as fp:
                sample_data_abnormal = json.load(fp)
            abnormal.extend(sample_data_abnormal["channel1"])
        abnormal_fft = get_fft_attritubes(abnormal, "channel1", sample_rate)

        distance_between_ffts.append(get_euclidean_distance(normal_fft["signal_fft"], abnormal_fft["signal_fft"]))
        cosine_similarity_ffts.append(get_cosine_similarity(normal_fft["signal_fft"], abnormal_fft["signal_fft"]))
        # get_figure_plotly_1channel(normal, abnormal, normal_fft, abnormal_fft, "unbalance", i)

    with open("../res/3rd/long_term_distance_between_fft_3rd.pickle", "wb") as f:
        pickle.dump(distance_between_ffts, f, pickle.HIGHEST_PROTOCOL)
    with open("../res/3rd/long_term_cosine_similarity_fft_3rd.pickle", "wb") as f:
        pickle.dump(cosine_similarity_ffts, f, pickle.HIGHEST_PROTOCOL)

    with open("../res/3rd/long_term_distance_between_fft_3rd.pickle", "rb") as f:
        d1 = pickle.load(f)
    with open("../res/3rd/long_term_cosine_similarity_fft_3rd.pickle", "rb") as f:
        s1 = pickle.load(f)

    idx = distance_between_ffts.index(min(distance_between_ffts))
    print("min distance index fft:", idx)
    print("min distance fft:", min(distance_between_ffts))
    print("cos similarity fft:", cosine_similarity_ffts[idx])
    print("mean distance fft:", sum(distance_between_ffts)/200)

    quit()

    # TODO: plotting consecutive time-series

    # normal = []
    # abnormal = []

    # for i, nor in enumerate(flist_normal):
    #     with open(nor, "r") as fp:
    #         sample_data_normal = json.load(fp)
    #         normal.extend(sample_data_normal["channel1"])

    # with open("../data/normal_all_time_1st.pickle", "wb") as f:
    #     pickle.dump(normal, f, pickle.HIGHEST_PROTOCOL)

    # for j, ab in enumerate(flist_abnormal):
    #     with open(ab, "r") as fp:
    #         sample_data_abnormal = json.load(fp)
    #         abnormal.extend(sample_data_abnormal["channel1"])

    # with open("../data/abnormal_all_time_1st_Looseness_high.pickle", "wb") as f:
    #     pickle.dump(abnormal, f, pickle.HIGHEST_PROTOCOL)

    # TODO: get fft attributes in a long-term

    length = 3000000

    with open("../data/all_time/dataset1/normal_all_time_1st.pickle", "rb") as f:
        normal_1st = pickle.load(f)
    with open("../data/all_time/dataset1/abnormal_all_time_1st_Looseness.pickle", "rb") as f:
        abnormal_1st_loose = pickle.load(f)
    with open("../data/all_time/dataset1/abnormal_all_time_1st_Looseness_high.pickle", "rb") as f:
        abnormal_1st_loose_high = pickle.load(f)
    with open("../data/all_time/dataset1/abnormal_all_time_1st_bearing.pickle", "rb") as f:
        abnormal_1st_bearing = pickle.load(f)
    with open("../data/all_time/dataset1/abnormal_all_time_1st_unbalance.pickle", "rb") as f:
        abnormal_1st_unbalance = pickle.load(f)

    normal_1st = normal_1st[:length]
    abnormal_1st_loose = abnormal_1st_loose[:length]
    abnormal_1st_loose_high = abnormal_1st_loose_high[:length]
    abnormal_1st_bearing = abnormal_1st_bearing[:length]
    abnormal_1st_unbalance = abnormal_1st_unbalance[:length]

    signal_normal_1st_fft = get_fft_attritubes(normal_1st, "channel1", sample_rate)
    abnormal_1st_loose_fft = get_fft_attritubes(abnormal_1st_loose, "channel1", sample_rate)
    abnormal_1st_loose_high_fft = get_fft_attritubes(abnormal_1st_loose_high, "channel1", sample_rate)
    abnormal_1st_bearing_fft = get_fft_attritubes(abnormal_1st_bearing, "channel1", sample_rate)
    abnormal_1st_unbalance_fft = get_fft_attritubes(abnormal_1st_unbalance, "channel1", sample_rate)

    with open("../data/all_time/dataset1/fft/signal_normal_1st_fft.pickle", "wb") as f:
        pickle.dump(signal_normal_1st_fft, f, pickle.HIGHEST_PROTOCOL)
    with open("../data/all_time/dataset1/fft/signal_abnormal_1st_loose_fft.pickle", "wb") as f:
        pickle.dump(abnormal_1st_loose_fft, f, pickle.HIGHEST_PROTOCOL)
    with open("../data/all_time/dataset1/fft/signal_abnormal_1st_loose_high_fft.pickle", "wb") as f:
        pickle.dump(abnormal_1st_loose_high_fft, f, pickle.HIGHEST_PROTOCOL)
    with open("../data/all_time/dataset1/fft/signal_abnormal_1st_bearing_fft.pickle", "wb") as f:
        pickle.dump(abnormal_1st_bearing_fft, f, pickle.HIGHEST_PROTOCOL)
    with open("../data/all_time/dataset1/fft/signal_abnormal_1st_unbalance_fft.pickle", "wb") as f:
        pickle.dump(abnormal_1st_unbalance_fft, f, pickle.HIGHEST_PROTOCOL)

    with open("../data/all_time/dataset1/fft/signal_normal_fft_1st.pickle", "rb") as f:
        signal_normal_1st_fft = pickle.load(f)
    with open("../data/all_time/dataset1/fft/signal_abnormal_1st_loose_fft.pickle", "rb") as f:
        abnormal_1st_loose_fft = pickle.load(f)
    with open("../data/all_time/dataset1/fft/signal_abnormal_1st_loose_high_fft.pickle", "rb") as f:
        abnormal_1st_loose_high_fft = pickle.load(f)
    with open("../data/all_time/dataset1/fft/signal_abnormal_1st_bearing_fft.pickle", "rb") as f:
        abnormal_1st_bearing_fft = pickle.load(f)
    with open("../data/all_time/dataset1/fft/signal_abnormal_1st_unbalance_fft.pickle", "rb") as f:
        abnormal_1st_unbalance_fft = pickle.load(f)

    get_figure_plotly_1channel(normal_1st, abnormal_1st_loose, signal_normal_1st_fft, abnormal_1st_loose_fft, "loose")
    get_figure_plotly_1channel(normal_1st, abnormal_1st_loose_high, signal_normal_1st_fft, abnormal_1st_loose_high_fft, "loosehigh")
    get_figure_plotly_1channel(normal_1st, abnormal_1st_bearing, signal_normal_1st_fft, abnormal_1st_bearing_fft, "bearing")
    get_figure_plotly_1channel(normal_1st, abnormal_1st_unbalance, signal_normal_1st_fft, abnormal_1st_unbalance_fft, "unbalance")

    # distance_between_normal_abnormal_all_time = get_euclidean_distance(normal, abnormal)
    # cos_sim_normal_abnormal_all_time = get_cosine_similarity(normal, abnormal)
    # print(distance_between_normal_abnormal_all_time, cos_sim_normal_abnormal_all_time)

    # distance_between_normal_1st_3rd = get_euclidean_distance(normal_1st, normal_3rd)
    # cos_sim_between_normal_1st_3rd = get_cosine_similarity(normal_1st, normal_3rd)
    # print(distance_between_normal_1st_3rd,cos_sim_between_normal_1st_3rd)

    # plt.figure(figsize=(40, 10))
    # plt.subplot(2, 1, 1)
    # plt.title("Normal", fontsize=20)
    # plt.plot(normal[:length], c="b")
    # plt.xlabel("time", fontsize=20)
    # plt.tick_params(axis='x', labelsize=20)
    # plt.xticks(np.arange(0, length, 100 * sample_rate),
    #            labels=[i for i in range(int(length / sample_rate) + 1) if i % 100 == 0])
    # plt.ylabel("amplitude", fontsize=20)
    # plt.grid()
    #
    # plt.subplot(2, 1, 2)
    # plt.title("Abnormal", fontsize=20)
    # plt.plot(abnormal[:length], c="r")
    # plt.xlabel("time", fontsize=20)
    # plt.xticks(np.arange(0, length, 100*sample_rate), labels=[i for i in range(int(length/sample_rate)+1) if i%100 == 0])
    # plt.tick_params(axis='x', labelsize=20)
    # plt.ylabel("amplitude", fontsize=20)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig("../res/3rd/consecutive_data_3rd_1mil_vis.png", dpi=100)
    # plt.show()





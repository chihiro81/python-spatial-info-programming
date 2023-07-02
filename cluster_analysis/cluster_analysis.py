"""
Author: Benjamin Lee, Chihiro Matsumoto
Date: 08/06/2022
Description: GEOM90042 Assignment4

This is a module caontaining functions for cluster analysis.
"""

# Import libraries
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import data_processing as dp


def create_DBSCAN_subplot(dataset1, dataset2,
                          title, subtitle1="", subtitle2="", tocrs=4283):
    """This function creates subplots to compare DBSCANS of two datasets"""

    coord1 = []
    coord2 = []

    for coord in dataset1.to_crs(tocrs).geometry:
        coord1.append((coord.x, coord.y))

    for coord in dataset2.to_crs(tocrs).geometry:
        coord2.append((coord.x, coord.y))

    X1 = StandardScaler(with_mean=False, with_std=False).fit_transform(coord1)
    X2 = StandardScaler(with_mean=False, with_std=False).fit_transform(coord2)

    db1 = DBSCAN(eps=.0015, min_samples=15).fit(X1)
    core_samples_mask1 = np.zeros_like(db1.labels_, dtype=bool)
    core_samples_mask1[db1.core_sample_indices_] = True
    labels1 = db1.labels_

    db2 = DBSCAN(eps=.0015, min_samples=15).fit(X2)
    core_samples_mask2 = np.zeros_like(db2.labels_, dtype=bool)
    core_samples_mask2[db2.core_sample_indices_] = True
    labels2 = db2.labels_

    # Black removed and is used for noise instead.
    unique_labels1 = set(labels1)
    unique_labels2 = set(labels2)

    colors1 = [
        plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels1))]
    colors2 = [
        plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels2))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)

    ax1.set_title(subtitle1)
    for k, col in zip(unique_labels1, colors1):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels1 == k

        xy = X1[class_member_mask & ~core_samples_mask1]
        ax1.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=.25,
        )

        xy = X1[class_member_mask & core_samples_mask1]
        ax1.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

    ax2.set_title(subtitle2)
    for k, col in zip(unique_labels2, colors2):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels2 == k

        xy = X2[class_member_mask & ~core_samples_mask2]
        ax2.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=.25,
        )

        xy = X2[class_member_mask & core_samples_mask2]
        ax2.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

    return


def create_DBSCAN_with_pop(dataset, title, tocrs=4283):
    """This function creates a color map based on population """

    # Read population data
    pop = dp.import_pop_gpkg()

    # Retrive only Melbourne city area
    LGA17 = dp.import_shp("LGA_2017_VIC.shp")
    dp.adjust_lga_names(LGA17)
    LGA_mel = LGA17[LGA17['LGA_NAME_adj'] == 'MELBOURNE']

    # Clip population layer
    tmp = gpd.clip(pop, LGA_mel)
    tmp = tmp[tmp.geom_type == 'Polygon']
    list = tmp["SA2_name_2016"].tolist()\
        + ['Port Melbourne Industrial', 'Carlton North - Princes Hill']
    pop_mel = pop.loc[pop["SA2_name_2016"].isin(list)]

    coords = []
    for coord in dataset.to_crs(tocrs).geometry:
        coords.append((coord.x, coord.y))

    X = StandardScaler(with_mean=False, with_std=False).fit_transform(coords)

    db = DBSCAN(eps=.0015, min_samples=15).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [
        plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels))]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title, fontsize=8)
    ax = pop_mel.plot(
                    ax=ax,
                    column='ERP_2016',
                    cmap='Greys',
                    edgecolor='black',
                    alpha=0.8)
    plt.tick_params(axis='x', labelsize=7)
    plt.tick_params(axis='y', labelsize=7)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=.25,
        )

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=5,
        )

    return

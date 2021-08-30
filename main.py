import time
from io import StringIO
from datetime import datetime
from typing import Tuple, Union, List

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry.multipolygon import MultiPolygon
from matplotlib.animation import FuncAnimation, MovieWriter


def format_date(x: str) -> str:
    """
    format("%Y/%-m/%-d")をformat("%Y-%m-%d")に変換する
    errorなら""
    """
    try:
        return datetime.strptime(x, "%Y/%m/%d").strftime("%Y-%m-%d")
    except:
        return ""


class MapData:
    def __init__(self):
        self.__read_file()
        self.map_data: List[np.ndarray] = []

    def __read_file(self):
        geojson_url = "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"
        geo_df = gpd.read_file(geojson_url)

        self.geometry = geo_df["geometry"]

    def __append_exterior(self, exterior: Union[LinearRing, None]):
        if exterior:
            _coords = np.array(exterior.coords)
            if _coords.shape[0] >= self.threshold:
                self.map_data.append(_coords)

    def data(self, threshold: int = 0) -> List[np.ndarray]:
        self.threshold = threshold
        for geo in self.geometry:
            if isinstance(geo, Polygon):
                self.__append_exterior(geo.exterior)

            elif isinstance(geo, MultiPolygon):
                for i in range(len(geo.geoms)):
                    self.__append_exterior(geo[i].exterior)

        return self.map_data

    @classmethod
    def get(cls, threshold: int):
        mapData = cls()
        return mapData.data(threshold)


def fetch_npatients_data(from_: str, to_: str) -> Tuple[pd.DataFrame, pd.Series]:
    url = "https://www3.nhk.or.jp/n-data/opendata/coronavirus/nhk_news_covid19_prefectures_daily_data.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3864.0 Safari/537.36"
    }

    res = requests.get(url, headers=headers)
    df = pd.concat(
        [d for d in pd.read_csv(StringIO(res.text), chunksize=1000)],
        ignore_index=True
    )
    df = df[["日付", "都道府県名", "各地の感染者数_1日ごとの発表数"]]
    df["日付"] = df["日付"].apply(lambda x: format_date(x))

    latlng = pd.read_csv("./data/latlng.csv")
    df = pd.merge(df, latlng, how="left", on="都道府県名")
    df = df.drop(["都道府県名"], axis=1)

    max_moving_av = (
        df
        .groupby("日付")["各地の感染者数_1日ごとの発表数"]
        .max()
        .rolling(7)
        .apply(lambda x: np.nanmean(x))
        .fillna(0)
    )

    return df.query(f"'{from_}' <= 日付 <= '{to_}'"), max_moving_av


def _check_data(animation_class) -> None:
    required_data = ["map_data", "df", "max_moving_av"]
    attrs = all([hasattr(animation_class, data) for data in required_data])

    if not attrs:
        raise Exception(
            "data is not found. Call 'set_data' before using this method."
        )


class Bar3dAnimation:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(left=0, right=1, bottom=0)
        self.__set_style()

    def set_data(self,
                 map_data: List[np.ndarray],
                 df: pd.DataFrame,
                 max_moving_av: pd.Series):
        self.map_data = map_data
        self.df = df
        self.max_moving_av = max_moving_av

    def __draw_map(self):
        for data in self.map_data:
            self.ax.plot3D(data[:, 0], data[:, 1], 0,
                           color="black", linewidth=1, alpha=0.5)

    def __update(self, date: str, *frags):
        plt.cla()
        self.__draw_map()
        _df = self.df[self.df["日付"] == date]
        self.ax.bar3d(np.array(_df["経度"]),
                      np.array(_df["緯度"]),
                      0, 0.5, 0.5,
                      np.array(_df["各地の感染者数_1日ごとの発表数"]),
                      color="blue")
        self.ax.set_title(date)
        self.__set_style(zlim_top=self.max_moving_av[date]+1)

    def __set_style(self, zlim_top: int = 1):
        self.ax.set_xlim(125, 149)
        self.ax.set_ylim(23, 49)
        self.ax.set_zlim(0, zlim_top)
        self.ax.grid(False)
        self.ax.xaxis.pane.set_edgecolor("white")
        self.ax.yaxis.pane.set_edgecolor("white")
        self.ax.zaxis.pane.set_edgecolor("white")
        self.ax.xaxis.set_major_locator(ticker.NullLocator())
        self.ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_major_locator(ticker.NullLocator())
        self.ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    def __animate(self):
        _check_data(self)
        date_stream = self.df["日付"].unique()
        self.ani = FuncAnimation(self.fig,
                                 self.__update,  # type: ignore
                                 frames=date_stream,
                                 interval=100)

    def show(self):
        self.__animate()
        plt.show()

    def save(self,
             file_path: str,
             writer: Union[MovieWriter, str, None]):
        self.__animate()
        self.ani.save(file_path, writer=writer)


def covid_3d_mapping(from_: str, to_: str) -> None:
    map_data = MapData.get(100)
    patients_df, max_moving_av = fetch_npatients_data(from_, to_)

    ani = Bar3dAnimation()
    ani.set_data(map_data, patients_df, max_moving_av)
    # ani.show()
    ani.save("./result.gif", writer="pillow")


if __name__ == "__main__":
    s = time.time()

    from_ = "2021-04-01"
    to_ = "2021-12-31"
    covid_3d_mapping(from_, to_)

    print(time.time() - s)

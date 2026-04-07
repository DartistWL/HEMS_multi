"""
This file contains interface functions for data processing
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional


class DataInterface:
    def __init__(
            self,
            cons_file,
            prod_file,
            price_profile: str = "legacy",
            steps_per_day: int = 48,
            tou_hourly_values: Optional[List[float]] = None,
    ):
        # Define file paths
        self.cons_file = cons_file
        self.prod_file = prod_file

        # Read data
        # self.cons_data = pd.read_csv(self.cons_file, parse_dates=['date'], index_col='date')
        # self.prod_data = pd.read_csv(self.prod_file, parse_dates=['date'], index_col='date')
        try:
            self.cons_data = pd.read_csv(
                self.cons_file,
                parse_dates=['date'],
                index_col='date',
                low_memory=False,  # 避免分块类型推断
                dtype=float  # 强制所有数据列为浮点数
            )
            self.prod_data = pd.read_csv(
                self.prod_file,
                parse_dates=['date'],
                index_col='date',
                low_memory=False,
                dtype=float
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read {self.cons_file}: {e}") from e

        # ===== Electricity price profile (TOU) =====
        # - price_profile="legacy": keep current project default (valley/flat/peak with fixed index ranges)
        # - price_profile="lee2020": TOU curve from Lee & Choi (IEEE TII, 2020) Fig.3(a)
        #   Implemented as a 24-hour (hourly) vector that is expanded to `steps_per_day` (e.g., 48 for 30-min).
        #   If you want an exact match to the paper figure, pass `tou_hourly_values` (length=24) explicitly.
        self.price_profile = price_profile
        self.steps_per_day = int(steps_per_day)
        if self.steps_per_day <= 0:
            self.steps_per_day = 48
        self.tou_hourly_values = tou_hourly_values
        self._tou_price_cache = None  # lazily built daily price array (length = steps_per_day)

        # Store daily EV arrival and departure times
        self.ev_schedule = {}
        # Set random seed
        self.np_random = np.random.RandomState(0)
        self.cons_data.index = pd.to_datetime(self.cons_data.index, errors='coerce')
        self.cons_data = self.cons_data[~self.cons_data.index.isna()]  # 删除无效日期行

    def _build_daily_tou_price(self) -> np.ndarray:
        """
        Build a 1-day TOU price array with `self.steps_per_day` steps.

        Returns:
            np.ndarray: shape (steps_per_day,), dtype float32
        """
        steps = self.steps_per_day
        # Step duration in hours (48 -> 0.5h)
        step_h = 24.0 / steps

        if self.price_profile == "legacy":
            # Keep the exact same behavior as the original get_electricity_price() implementation.
            # Prices are normalized values used inside reward shaping (not necessarily real currency).
            time_based_prices = {
                'valley': 0.2,
                'flat': 0.5,
                'peak': 0.8,
            }
            prices = np.zeros(steps, dtype=np.float32)
            for idx in range(steps):
                # Original rule uses 48 steps: valley at [0,20) U [44,48), flat at [8,34), else peak.
                # For non-48 steps, we scale boundaries proportionally.
                if steps == 48:
                    i = idx
                    if 0 <= i < 20 or 44 <= i < 48:
                        prices[idx] = time_based_prices['valley']
                    elif 8 <= i < 34:
                        prices[idx] = time_based_prices['flat']
                    else:
                        prices[idx] = time_based_prices['peak']
                else:
                    # Proportional mapping (keeps shape similar when steps_per_day changes)
                    t = idx * step_h  # hour in [0,24)
                    # Map to the same "legacy" intention: valley at night, flat daytime, peak evening shoulder.
                    if 0.0 <= t < 10.0 or 22.0 <= t < 24.0:
                        prices[idx] = time_based_prices['valley']
                    elif 4.0 <= t < 17.0:
                        prices[idx] = time_based_prices['flat']
                    else:
                        prices[idx] = time_based_prices['peak']
            return prices

        if self.price_profile == "lee2020":
            # Lee & Choi (IEEE TII, 2020) Fig.3(a) shows a 24-hour TOU price profile.
            # The paper figure uses a $ scale roughly in [0.03, 0.18].
            #
            # We represent it as an hourly vector (len=24) and then expand to `steps_per_day`.
            # If you need an exact match to the figure, pass `tou_hourly_values` explicitly.
            if self.tou_hourly_values is not None:
                if len(self.tou_hourly_values) != 24:
                    raise ValueError("tou_hourly_values must be a list of length 24 (hourly TOU prices).")
                hourly = np.array(self.tou_hourly_values, dtype=np.float32)
            else:
                # Default hourly TOU profile EXACTLY matching the points you provided (from your Lee2020 figure):
                # Hour  1- 8: 0.06
                # Hour     9: 0.12
                # Hour 10-11: 0.14
                # Hour    12: 0.12
                # Hour 13-16: 0.14
                # Hour 17-22: 0.12
                # Hour 23-24: 0.06
                # Note: indices 0..23 correspond to hours 1..24.
                hourly = np.array(
                    ([0.06] * 8)
                    + ([0.12] * 1)
                    + ([0.14] * 2)
                    + ([0.12] * 1)
                    + ([0.14] * 4)
                    + ([0.12] * 6)
                    + ([0.06] * 2),
                    dtype=np.float32,
                )

            # Expand hourly -> steps_per_day (e.g. 24 -> 48 by duplication)
            factor = steps / 24.0
            prices = np.zeros(steps, dtype=np.float32)
            for idx in range(steps):
                hour_idx = int(np.floor(idx / factor))
                hour_idx = max(0, min(23, hour_idx))
                prices[idx] = hourly[hour_idx]
            return prices.astype(np.float32)

        # Unknown profile: fallback to legacy behavior
        self.price_profile = "legacy"
        return self._build_daily_tou_price()

    def set_price_profile(self, price_profile: str, tou_hourly_values: Optional[List[float]] = None):
        """Update TOU price profile and clear cached daily prices."""
        self.price_profile = price_profile
        self.tou_hourly_values = tou_hourly_values
        self._tou_price_cache = None

    def seed(self, seed: int) -> None:
        """Seed internal randomness (used by EV schedule generation)."""
        self.np_random = np.random.RandomState(int(seed))

    @classmethod
    def from_storenet_ireland_2020(
            cls,
            house_id: str,
            base_dir: str = "data/storenet_ireland_2020",
            price_profile: str = "legacy",
            steps_per_day: int = 48,
            tou_hourly_values: Optional[List[float]] = None,
    ):
        """
        Convenience constructor for the Ireland energy community dataset (StoreNet, Sci Data 2024).

        Expected files (generated by our converter script):
            {base_dir}/daily_pivot_cons_2020_{house_id}.csv
            {base_dir}/daily_pivot_prod_2020_{house_id}.csv
        """
        hid = str(house_id).strip().upper()
        # 修改这里：添加 'H' 前缀
        if not hid.startswith('H'):
            hid = 'H' + hid
        cons_file = f"{base_dir}/daily_pivot_cons_2020_{hid}.csv"
        prod_file = f"{base_dir}/daily_pivot_prod_2020_{hid}.csv"

        return cls(
            cons_file,
            prod_file,
            price_profile=price_profile,
            steps_per_day=steps_per_day,
            tou_hourly_values=tou_hourly_values,
        )

    # def get_home_load(self, current_date, current_time_index):
    #     # Get current household electricity consumption
    #     return self.cons_data.loc[current_date, self.cons_data.columns[current_time_index]]

    # def get_pv_generation(self, current_date, current_time_index):
    #     # Get current PV system generation
    #     return self.prod_data.loc[current_date, self.cons_data.columns[current_time_index]]

    def get_pv_generation(self, current_date, current_time_index):
        try:
            # 防御日期
            if not isinstance(current_date, str) or len(current_date) < 10 or current_date not in self.prod_data.index:
                return 0.0
            col_idx = int(current_time_index)
            if col_idx >= len(self.prod_data.columns):
                col_idx = len(self.prod_data.columns) - 1
            return self.prod_data.loc[current_date, self.prod_data.columns[col_idx]]
        except Exception:
            return 0.0

    def get_home_load(self, current_date, current_time_index):
        try:
            if not isinstance(current_date, str) or len(current_date) < 10 or current_date not in self.cons_data.index:
                return 0.0
            col_idx = int(current_time_index)
            if col_idx >= len(self.cons_data.columns):
                col_idx = len(self.cons_data.columns) - 1
            return self.cons_data.loc[current_date, self.cons_data.columns[col_idx]]
        except Exception:
            return 0.0

    def get_electricity_price(self, current_date, current_time_index):
        # Build daily TOU price once and index into it.
        if self._tou_price_cache is None:
            self._tou_price_cache = self._build_daily_tou_price()

        idx = int(current_time_index) % int(self.steps_per_day)
        return float(self._tou_price_cache[idx])

    def get_date_time(self, current_date, current_time_index):
        return current_date

    # def is_ev_at_home(self, current_date, current_time_index):
    #     """
    #     Determine if the electric vehicle is at home at the current time.
    #     Generate arrival and departure times only once per day, regenerate the next day.
    #     """
    #     # Check if arrival and departure times have been generated for current date
    #     if current_date not in self.ev_schedule:
    #         self.generate_daily_ev_schedule(current_date)
    #
    #     # If Saturday or Sunday, directly return True indicating EV is at home all day
    #     current_weekday = datetime.strptime(current_date, '%Y-%m-%d').weekday()
    #     if current_weekday >= 5:  # 5 represents Saturday, 6 represents Sunday
    #         return True
    #
    #     t1, t2 = self.ev_schedule[current_date]
    #     current_hour = current_time_index
    #     return not t2 <= current_hour < t1
    def is_ev_at_home(self, current_date, current_time_index):
        # 防御：确保 current_date 是有效的非空字符串
        if not isinstance(current_date, str) or len(current_date) < 10:
            print(
                f"WARNING: is_ev_at_home received invalid date: {current_date} (type={type(current_date)}), returning True")
            return True

        # 尝试解析日期，若失败则返回 True
        try:
            datetime.strptime(current_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            print(f"WARNING: is_ev_at_home cannot parse date: {current_date}, returning True")
            return True

        # 检查是否已生成该日期的 EV 调度
        if current_date not in self.ev_schedule:
            self.generate_daily_ev_schedule(current_date)

        # 周末直接返回 True
        current_weekday = datetime.strptime(current_date, '%Y-%m-%d').weekday()
        if current_weekday >= 5:
            return True

        t1, t2 = self.ev_schedule[current_date]
        current_hour = current_time_index
        return not t2 <= current_hour < t1

    def generate_daily_ev_schedule(self, current_date):
        if not isinstance(current_date, str) or len(current_date) < 10:
            print(f"WARNING: generate_daily_ev_schedule invalid date: {current_date}, setting default")
            self.ev_schedule[current_date] = (0, 48)
            return
        try:
            datetime.strptime(current_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            print(f"WARNING: generate_daily_ev_schedule cannot parse date: {current_date}, setting default")
            self.ev_schedule[current_date] = (0, 48)
            return
        """
        Generate EV arrival and departure times for the current date.
        """
        # If Saturday or Sunday, directly set EV to be at home all day
        current_weekday = datetime.strptime(current_date, '%Y-%m-%d').weekday()
        if current_weekday >= 5:  # 5 represents Saturday, 6 represents Sunday
            self.ev_schedule[current_date] = (0, 24 * 2)  # At home all day
            return

        # Adjust distribution parameters for arrival and departure times
        t1_mean = 20  # Increase mean arrival time (later return home)
        t1_std = 1
        t1_range = (18, 22)  # Expand arrival time range

        t2_mean = 8  # Decrease mean departure time (earlier departure)
        t2_std = 1
        t2_range = (7, 9)  # Narrow departure time range to ensure t2 < t1

        while True:
            t1 = self.np_random.normal(t1_mean, t1_std)
            t2 = self.np_random.normal(t2_mean, t2_std)
            t1 = max(min(t1, t1_range[1]), t1_range[0])
            t2 = max(min(t2, t2_range[1]), t2_range[0])
            if t2 < t1:
                break  # Ensure departure time is earlier than arrival time

        # Store generated times in units of 0.5 (representing half-hour)
        self.ev_schedule[current_date] = (t1 * 2, t2 * 2)

    def is_ev_departing_soon(self, current_date, current_time_idx):
        """Determine if EV is about to depart (within next 2 hours)"""
        return self.get_hours_until_departure(current_date, current_time_idx) <= 2

    def get_hours_until_departure(self, current_date, current_time_idx):
        """Get remaining time until next departure (in hours)"""
        # current_date = datetime.strptime(current_date, '%Y-%m-%d')
        if current_date not in self.ev_schedule:
            self.generate_daily_ev_schedule(current_date)

        departure_time = self.ev_schedule[current_date][1] / 2  # Convert to hours
        current_hour = current_time_idx / 2
        return departure_time - current_hour if current_hour < departure_time else (
                24 + departure_time - current_hour)

    def get_outdoor_temp(self, current_time, current_time_index):
        """
        Outdoor temperature simulation
        """
        try:
            if not isinstance(current_time, str):
                current_time = str(current_time)
            # 如果字符串长度不足10，填充默认日期
            if len(current_time) < 10:
                current_time = "2020-01-01"
            current_datetime = datetime.strptime(current_time, '%Y-%m-%d')
        except (ValueError, TypeError):
            # 静默返回默认温度，不打印警告（避免干扰）
            return 20.0

        day_of_year = current_datetime.timetuple().tm_yday

        # ===== Improved Parameter Settings =====
        # Annual base temperature model
        base_temp = 24 + 10 * np.sin(2 * np.pi * (day_of_year - 200) / 365)  # July peak adjustment

        # Daily temperature variation model (increased day-night temperature difference)
        hour = current_time_index // 2
        minute = 30 * (current_time_index % 2)
        time_of_day = hour + minute / 60
        daily_temp_variation = 8 * np.sin(2 * np.pi * (time_of_day - 14.5) / 24)

        # Summer high temperature
        summer_boost = 0
        if 172 <= day_of_year <= 265:  # Period from 6/21 to 9/22
            summer_boost = 4 * np.sin(np.pi * (day_of_year - 172) / 93)  # Additional summer temperature increase

        # Weather uncertainty
        random_noise = np.random.normal(0, 2.5)

        # ===== Final Temperature Calculation =====
        outdoor_temp = base_temp + daily_temp_variation + summer_boost + random_noise

        # Temperature boundary protection
        outdoor_temp = np.clip(outdoor_temp, -5, 45)

        return round(outdoor_temp, 1)
    # def get_outdoor_temp(self, current_time, current_time_index):
    #     """
    #     Outdoor temperature simulation
    #     """
    #     current_datetime = datetime.strptime(current_time, '%Y-%m-%d')
    #     day_of_year = current_datetime.timetuple().tm_yday
    #
    #     # ===== Improved Parameter Settings =====
    #     # Annual base temperature model
    #     base_temp = 24 + 10 * np.sin(2 * np.pi * (day_of_year - 200) / 365)  # July peak adjustment
    #
    #     # Daily temperature variation model (increased day-night temperature difference)
    #     hour = current_time_index // 2
    #     minute = 30 * (current_time_index % 2)
    #     time_of_day = hour + minute / 60
    #     daily_temp_variation = 8 * np.sin(2 * np.pi * (time_of_day - 14.5) / 24)
    #
    #     # Summer high temperature
    #     summer_boost = 0
    #     if 172 <= day_of_year <= 265:  # Period from 6/21 to 9/22
    #         summer_boost = 4 * np.sin(np.pi * (day_of_year - 172) / 93)  # Additional summer temperature increase
    #
    #     # Weather uncertainty
    #     random_noise = np.random.normal(0, 2.5)
    #
    #     # ===== Final Temperature Calculation =====
    #     outdoor_temp = base_temp + daily_temp_variation + summer_boost + random_noise
    #
    #     # Temperature boundary protection
    #     outdoor_temp = np.clip(outdoor_temp, -5, 45)
    #
    #     return round(outdoor_temp, 1)

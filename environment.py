"""
This file is used to simulate the interaction between electric vehicles and the environment, integrating washing machines and air conditioning equipment
"""
from datetime import datetime, timedelta
from scipy.stats import uniform
import matplotlib.dates as mdates
import torch
import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from interface import DataInterface
import csv  # Add CSV module
import os   # Add OS module
import sys
import pandas as pd
from typing import Optional, List, Tuple, Dict


def build_wash_schedule_action_tuple(
    steps_per_day: int,
    pref: Tuple[int, int],
    wash_duration_hours: float,
) -> Tuple[int, ...]:
    """
    洗衣机离散动作取值：0=HOLD（不改当前预约），1..N 表示「开洗半步索引 = earliest + (k-1)」。
    偏好窗 [pref[0], pref[1]] 整点；须在窗内开洗且 1h 洗涤在 pref[1] 前结束。
    """
    spd = int(steps_per_day)
    earliest = int(pref[0] * spd / 24)
    dur_steps = max(1, int(round(float(wash_duration_hours) * spd / 24)))
    latest = int(pref[1] * spd / 24) - dur_steps
    latest = max(earliest, min(spd - 1, latest))
    n_slots = latest - earliest + 1
    if n_slots < 1:
        raise ValueError(
            f"洗衣机偏好窗过窄：earliest={earliest}, latest={latest}, steps_per_day={spd}"
        )
    return (0,) + tuple(range(1, n_slots + 1))


def _calendar_day_label_cn(date_str: str) -> str:
    """
    返回中文星期 + 工作日/双休日说明，与 interface.DataInterface.is_ev_at_home 一致：
    - 周六、周日：EV 全天在家，可调度；
    - 周一至周五：按随机离/返家时段，非全天可调度。
    """
    if not date_str or len(str(date_str)) < 10:
        return ""
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
        wd = d.weekday()  # 周一=0 … 周日=6
        names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        if wd >= 5:
            return f"{names[wd]}·双休日（EV 全天在家可调度）"
        return f"{names[wd]}·工作日（EV 按离/返家，非全天可调度）"
    except ValueError:
        return ""


class HomeEnergyManagementEnv:
    def __init__(
        self,
        ev_capacity=24,
        ess_capacity=24,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        data_interface: Optional[DataInterface] = None,
        # Episode control (used for StoreNet date split / episode=1 day)
        episode_length_steps: Optional[int] = None,
        episode_schedule: Optional[List[Tuple[str, str]]] = None,  # [(house_id, date_str), ...]
        storenet_base_dir: str = "data/storenet_ireland_2020",
        price_profile: str = "lee2020",
        steps_per_day: int = 48,
        # Battery aging proxy (Version A: throughput / FEC style)
        # These weights are used as an online penalty in reward shaping.
        ess_aging_weight: float = 1.0,
        ev_aging_weight: float = 1.0,
        # Carbon (Ireland grid marginal intensity, 30-min aligned with steps_per_day=48)
        # 重要性建议：舒适度 > 成本 > 老化 > 碳 → carbon_weight 应小于 energy_weight 等
        use_carbon_in_reward: bool = True,
        carbon_intensity_csv_path: Optional[str] = "data/carbon_2020/eirgrid_roi_co2_2020_30min.csv",
        carbon_weight: float = 0.032,
        # Environment stochasticity knobs (for training stability / ablations)
        ac_user_behavior_std: float = 0.2,  # Std of AC user behavior disturbance
        ev_daily_km_mean: float = 50.0,     # Mean daily mileage (km)
        ev_daily_km_sigma: float = 0.2,     # Std of daily mileage (log-normal sigma)
    ):
        self.ev_capacity = ev_capacity  # EV battery capacity
        self.ess_capacity = ess_capacity  # ESS battery capacity
        self.charge_efficiency = charge_efficiency  # Charge efficiency
        self.discharge_efficiency = discharge_efficiency  # Discharge efficiency
        self.ev_min_charge = 12  # Set minimum charge requirement when EV leaves home

        # Initialize penalty coefficients for dynamic adjustment
        self.energy_weight = 0.1  # Grid cost weight
        self.user_satisfaction_weight0 = 0.5  # User dissatisfaction weight
        self.user_satisfaction_weight1 = 0.5  #
        # 洗衣机用户不满权重（加大以相对电价/总回报更重视偏好时段）
        self.user_satisfaction_weight2 = 0.4
        # 洗衣时间偏离偏好窗：每小时偏差惩罚系数（原 multi_agent implicit 5，现为 8）
        self.wash_time_penalty_coef = 8.0
        # 洗衣项内电价惩罚乘子：开洗时刻电价越高惩罚越大，促使在偏好窗内选低价时段
        self.wash_price_penalty_coef = 0.5
        self.violation_weight = 0.05
        self.temp_weight = 0.1
        self.ess_weight = 0.1  # Originally 1
        self.ev_weight = 0.1  # Originally 1
        self.ess_aging_weight = float(ess_aging_weight)
        self.ev_aging_weight = float(ev_aging_weight)
        self.use_carbon_in_reward = bool(use_carbon_in_reward)
        self.carbon_weight = float(carbon_weight)
        self._carbon_intensity_csv_path = carbon_intensity_csv_path
        self._carbon_df: Optional[pd.DataFrame] = None
        self._carbon_fallback_kg_per_kwh: float = 0.25
        self._carbon_kg_per_kwh_max: float = 0.5  # updated when CSV loads; used to normalize observation

        # Stochasticity parameters
        self.ac_user_behavior_std = float(ac_user_behavior_std)
        self.ev_daily_km_mean = float(ev_daily_km_mean)
        self.ev_daily_km_sigma = float(ev_daily_km_sigma)

        self.total_cost = 0

        self.episode_costs = []  # Store total cost for each episode
        self.current_step_cost = 0  # Store cost for current time step
        # 与 calculate_reward 中边际碳核算一致（供评估脚本汇总，不替代 reward 中的加权项）
        self.current_step_emission_kg = 0.0
        self.current_step_grid_import_kwh = 0.0
        self.current_step_grid_export_kwh = 0.0

        # ESS 内储存能量的来源拆分（kWh，与 ess_state 同量纲；近似会计，见 _update_ess_energy_source_bookkeeping）
        self._ess_stored_from_pv_kwh = 0.0
        self._ess_stored_from_grid_kwh = 0.0

        # 须早于 action_space：洗衣机动作为「偏好窗内绝对半步槽位」
        self.steps_per_day = int(steps_per_day)
        self.wash_machine_preferred_time = (6, 20)  # 6:00–20:00 允许洗涤（与论文设定一致可调）
        self.wash_machine_duration = 1.0  # 洗涤时长（小时）

        self.state_space = {
            'home_load': (0, 10),  # Household electricity consumption range
            'pv_generation': (0, 5),  # PV generation range
            'ess_state': (0, ess_capacity),  # ESS battery capacity
            'ev_battery_state': (0, ev_capacity),  # EV battery capacity
            'time_index': (0, max(1, self.steps_per_day)),  # 半步索引，上界随 steps_per_day
            'electricity_price': (0, 2),  # Electricity price range, example values
            'temperature': (0, 40),
            # Indoor temperatures are part of the Markov state for comfort penalties.
            # update_air_conditioner() clips them to [10, 40].
            'indoor_temp': (10, 40),
            'indoor_temp2': (10, 40),
            'wash_machine_state': (0, 1),  # Washing machine state, 0 means off, 1 means running
            # 'start_time': (-1, 48),
            # 'price_trend': (-1, 1),  # Electricity price trend (past 3-hour average vs current)
            'Air_conditioner_power': (0, 5),  # AC power range
            'Air_conditioner_power2': (0, 5),
            'ewh_temp': (40, 70),  # Water heater water temperature
            'ewh_power': (0, 2),  # Water heater power
            # 日历 / 可用性（便于学习季节、周末与 EV 可调度）
            'weekday_sin': (-1, 1),
            'weekday_cos': (-1, 1),
            'is_weekend': (0, 1),
            'day_of_year_sin': (-1, 1),
            'day_of_year_cos': (-1, 1),
            'ev_at_home': (0, 1),
            # 日内周期编码（与 time_index 并存，便于 MLP 学习 TOU 周期）
            'time_index_sin': (-1, 1),
            'time_index_cos': (-1, 1),
            # 洗衣机预约可观测性（马尔可夫补全）：是否有约、计划开洗在窗内相对位置、距开洗还剩多少步（归一化）
            'wash_pending': (0, 1),
            'wash_planned_start_norm': (0, 1),
            'wash_steps_until_start_norm': (0, 1),
            # 当前半步对应的电网边际碳强度（与碳奖励同源 CSV），按历史列 max 归一化到 [0,1]，加速策略学习碳时段
            'carbon_intensity_norm': (0.0, 1.0),
            # ESS 当前电量中来自光伏消纳/充电的份额（近似，见 _update_ess_energy_source_bookkeeping）
            'ess_charge_pv_fraction': (0.0, 1.0),
        }
        self.action_space = {
            'ev_power': (-6.6, -3.3, 0, 3.3, 6.6),  # EV charge/discharge power range

            'battery_power': (-4.4, -2.2, 0, 2.2, 4.4),  # ESS charge/discharge power range

            # 0=HOLD；1..N 表示开洗于半步 earliest+(k-1)（仅允许落在偏好窗内可开洗范围）
            'wash_machine_schedule': build_wash_schedule_action_tuple(
                self.steps_per_day,
                self.wash_machine_preferred_time,
                self.wash_machine_duration,
            ),
            'Air_conditioner_set_temp': (16, 18, 20, 22, 24, 26, 28, 30),  # AC set temperature
            'Air_conditioner_set_temp2': (16, 18, 20, 22, 24, 26, 28, 30),

            'ewh_set_temp': (40, 45, 50, 55, 60, 65, 70)  # Discrete temperature setting actions
        }
        # ===== Episode scheduling =====
        self.episode_length_steps = episode_length_steps
        self.episode_schedule = episode_schedule or None
        self.episode_schedule_cursor = 0
        self.storenet_base_dir = storenet_base_dir
        self.price_profile = price_profile

        self.current_time_index = 0
        # Repo root (environment.py lives at project root) — fixes relative paths when cwd is `model/` or IDE default.
        _project_root = os.path.dirname(os.path.abspath(__file__))
        self._project_root = _project_root
        # Normalize StoreNet folder: relative paths are resolved against repo root (not process cwd).
        if self.storenet_base_dir and not os.path.isabs(self.storenet_base_dir):
            self.storenet_base_dir = os.path.normpath(
                os.path.join(_project_root, self.storenet_base_dir)
            )

        self._load_carbon_intensity_table(_project_root)

        # Data source: explicit DataInterface OR StoreNet episode_schedule (legacy Ausgrid CSVs removed).
        if data_interface is not None:
            self.data_interface = data_interface
        elif self.episode_schedule is not None and len(self.episode_schedule) > 0:
            # reset() will reload per-episode; init uses first scheduled house for a valid interface.
            _house_id, _ = self.episode_schedule[0]
            self.data_interface = DataInterface.from_storenet_ireland_2020(
                house_id=_house_id,
                base_dir=self.storenet_base_dir,
                price_profile=self.price_profile,
                steps_per_day=self.steps_per_day,
            )
        else:
            raise ValueError(
                "HomeEnergyManagementEnv 需要 `data_interface=...` 或非空的 `episode_schedule`（StoreNet）。"
                "已不再加载 Ausgrid 2011–2012 旧数据。快速默认训练请用："
                "`from environment import make_storenet_train_env` 然后 `make_storenet_train_env()`。"
            )

        # 当前日历日（作为 pivot 表行索引），须落在 `cons_data` 已有日期上；不再使用硬编码 2011-07-03。
        if self.episode_schedule is not None and len(self.episode_schedule) > 0:
            self.current_time = HomeEnergyManagementEnv._normalize_date_str(self.episode_schedule[0][1])
        else:
            self.current_time = HomeEnergyManagementEnv._first_cons_date_str(self.data_interface)
        self._seed_value: Optional[int] = None
        self.current_ev_power = 0
        self.current_battery_power = 0

        self.ev_battery_record = []  # Record EV charge level
        self.ess_state_record = []  # Record ESS charge level
        self.home_load_record = []  # Record grid load
        self.pv_generation_record = []  # Record PV generation
        self.electricity_price_record = []  # Record electricity price
        self.ev_at_home_record = []  # Record whether EV is at home
        self.wash_machine_record = []  # Record washing machine state
        self.air_conditioner_power_record = []  # Record AC power
        self.ess_actions = []  # Record ESS charge/discharge power
        self.wash_machine_actions = []  # Record washing machine actions
        self.air_conditioner_actions = []  # Record AC actions

        # ESS 能量来源轨迹（每步末累计量，kWh；与 _ess_stored_from_* 一致）
        self.ess_charge_pv = []
        self.ess_charge_grid = []
        self.ess_discharge_ev = []  # 预留：未按负载拆分配电
        self.ess_discharge_house = []

        # New record variables
        self.records = {
            'timestamps': [],
            'ev_soc': [],
            'ess_soc': [],
            'grid_power': [],
            'energy_cost': [],
            'daily_costs': [],
            'indoor_temp': [],
            'indoor_temp2': [],
            'outdoor_temp': [],
            'current_daily_cost': 0,
            'total_load':[],
            'user_dissatisfaction': []  # User dissatisfaction record
        }

        # New reward record containers
        self.reward_components = {
            'total': [],
            'energy_cost': [],
            'violation_penalty': [],
            'aging_penalty': [],
            'ess_reward': [],
            'ev_reward': [],
            'user_penalty': [],
            'temp_reward': [],
            'carbon_penalty': [],
            # 'sell': []
        }

        # ================== New Water Heater Parameters ==================
        self.ewh_params = {
            # 'ξ_ewh': 0.993,  # Water temperature decay coefficient
            # 'R_prime': 4.18,  # Heat transfer parameter (kJ/°C)
            'h': 10,  # Convective heat transfer coefficient
            'temp_cold': 18,  # Cold water temperature (°C)
            'surface_area': 2,  # Tank surface area (m^2)
            'volume': 100,  # Tank capacity (L)
            'mass': 100,  # Water mass (kg)
            'temp_min': 30.0,  # Minimum acceptable water temperature (°C)
            'temp_max': 100.0,  # Maximum safe water temperature (°C)
            'temp_init': 40,  # Initial water temperature (°C)
            'user_flow_mean': 1.5,  # Average water flow rate (L/h)
            'user_flow_std': 0.3  # Water flow rate standard deviation
        }

        # New record variables
        self.ewh_temp_record = []
        self.ewh_power_record = []
        self.user_flow_record = []

        # Washing machine related parameters（偏好窗与时长已在 action_space 前初始化）
        self.wash_machine_power = 1.5  # Washing machine fixed power (kW)
        self.current_period_start = None  # Current period start time
        self.has_run_in_current_period = False  # Whether has run in current period
        self.time_deviation = 0  # Time offset (hours); used in washing-machine preference penalty
        self.wash_machine_state = 0  # Initial state is off
        self.last_action_time = None  # Record start time
        self.remaining_runtime = 0  # Remaining runtime (30-minute steps); 1h wash => 2 半步
        self.wash_machine_scheduled_start_step: Optional[int] = None  # 到该 step 自动开洗（窗内绝对槽位）
        self._wash_price_step_for_penalty: Optional[int] = None  # 本步若实际开洗，供 calculate_user_dissatisfaction2 取电价

        # AC related parameters
        self.indoor_temp = 25  # Initial indoor temperature
        self.indoor_temp2 = 25  # Second AC
        self.outdoor_temp = 25  # Initial outdoor temperature
        self.temp_change_rate = 0.5  # Indoor temperature change rate
        self.user_temp_preference = 22  # User preferred temperature
        self.user_temp_preference2 = 18  # Second AC

    @staticmethod
    def _normalize_date_str(d) -> str:
        """Normalize pandas index / datetime / str to 'YYYY-MM-DD' for pivot lookups."""
        if isinstance(d, str):
            s = d.strip()
            return s[:10] if len(s) >= 10 else s
        return pd.Timestamp(d).strftime('%Y-%m-%d')

    @staticmethod
    def _first_cons_date_str(di: DataInterface) -> str:
        return HomeEnergyManagementEnv._normalize_date_str(di.cons_data.index[0])

    @staticmethod
    def _last_cons_date_str(di: DataInterface) -> str:
        return HomeEnergyManagementEnv._normalize_date_str(di.cons_data.index[-1])

    def _calendar_observation_dict(self) -> Dict[str, float]:
        """星期/年内位置/是否周末/EV 是否在家，供策略区分季节与可调度时段。"""
        dt = datetime.strptime(self.current_time, '%Y-%m-%d')
        wd = int(dt.weekday())
        doy = int(dt.timetuple().tm_yday)
        sin_wd = float(np.sin(2 * np.pi * wd / 7.0))
        cos_wd = float(np.cos(2 * np.pi * wd / 7.0))
        sin_doy = float(np.sin(2 * np.pi * (doy - 1) / 365.25))
        cos_doy = float(np.cos(2 * np.pi * (doy - 1) / 365.25))
        is_weekend = 1.0 if wd >= 5 else 0.0
        ev_home = 1.0 if self.is_ev_at_home() else 0.0
        return {
            'weekday_sin': sin_wd,
            'weekday_cos': cos_wd,
            'is_weekend': is_weekend,
            'day_of_year_sin': sin_doy,
            'day_of_year_cos': cos_doy,
            'ev_at_home': ev_home,
        }

    def _time_index_observation_dict(self) -> Dict[str, float]:
        """当日半步位置的正余弦编码：2π·time_index/steps_per_day。"""
        spd = max(1, int(self.steps_per_day))
        ti = int(self.current_time_index) % spd
        ang = 2.0 * np.pi * float(ti) / float(spd)
        return {
            'time_index_sin': float(np.sin(ang)),
            'time_index_cos': float(np.cos(ang)),
        }

    def _wash_schedule_observation_dict(self) -> Dict[str, float]:
        """
        预约状态进观测：便于策略区分 HOLD/改约。
        - wash_pending: 是否存在未执行的预约开洗半步
        - wash_planned_start_norm: 计划开洗半步在 [earliest,latest] 内的归一化位置
        - wash_steps_until_start_norm: 距计划开洗还剩多少半步 / steps_per_day，裁剪到 [0,1]
        """
        spd = max(1, int(self.steps_per_day))
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        span = max(1, l - e)
        cur = int(self.current_time_index) % spd
        sch = self.wash_machine_scheduled_start_step
        if sch is None:
            return {
                'wash_pending': 0.0,
                'wash_planned_start_norm': 0.0,
                'wash_steps_until_start_norm': 0.0,
            }
        s = int(sch)
        planned_norm = float(np.clip((float(s - e) / float(span)), 0.0, 1.0))
        steps_left = max(0, s - cur)
        until_norm = float(np.clip(float(steps_left) / float(spd), 0.0, 1.0))
        return {
            'wash_pending': 1.0,
            'wash_planned_start_norm': planned_norm,
            'wash_steps_until_start_norm': until_norm,
        }

    def get_state_vector(self, state_dict):
        """Convert state dictionary to ordered list"""
        ordered_keys = sorted(state_dict.keys())  # Sort alphabetically
        return [state_dict[k] for k in ordered_keys]

    def get_action_mask(self, state, apply_soc_mask: bool = True):
        """
        Return dynamic action mask.
        
        Decoupling rule (for ablation):
        - EV-at-home constraint is ALWAYS enforced (cannot rely on SOC masking).
        - SOC-based charge/discharge feasibility masks are controlled by `apply_soc_mask`
          (so you can set `USE_DYNAMIC_MASK=False` for baseline).
        """
        masks = {
            'battery_power': [True] * len(self.action_space['battery_power']),
            'ev_power': [True] * len(self.action_space['ev_power'])
        }

        delta_t = 0.5

        if apply_soc_mask:
            # 1. ESS battery action mask - dynamic calculation
            # Consider discharge efficiency (assume 95%)
            max_discharge = (state['ess_state'] / delta_t) * self.discharge_efficiency
            max_charge = ((self.ess_capacity - state['ess_state']) / delta_t) / self.charge_efficiency
            
            for idx, action_value in enumerate(self.action_space['battery_power']):
                # Discharge action: can only select actions less than or equal to current dischargeable amount
                if action_value < 0 and abs(action_value) > max_discharge:
                    masks['battery_power'][idx] = False

                # Charge action: can only select charge power less than or equal to current chargeable space
                if action_value > 0 and action_value > max_charge:
                    masks['battery_power'][idx] = False

            # 2. EV battery action mask - also dynamic calculation
            max_ev_discharge = (state['ev_battery_state'] / delta_t) * self.discharge_efficiency
            max_ev_charge = ((self.ev_capacity - state['ev_battery_state']) / delta_t) / self.charge_efficiency
            
            for idx, action_value in enumerate(self.action_space['ev_power']):
                # Discharge limit
                if action_value < 0 and abs(action_value) > max_ev_discharge:
                    masks['ev_power'][idx] = False

                # Charge limit
                if action_value > 0 and action_value > max_ev_charge:
                    masks['ev_power'][idx] = False

        # 3. EV not at home mask
        if not self.is_ev_at_home():
            for idx, action_value in enumerate(self.action_space['ev_power']):
                if action_value != 0:  # Can only select 0 power action
                    masks['ev_power'][idx] = False

        # 4. 洗衣机：HOLD 始终合法；槽位 k>=1 仅当对应开洗半步不早于当前且落在窗内
        ti = int(state.get("time_index", 0))
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        masks["wash_machine_schedule"] = []
        for val in self.action_space["wash_machine_schedule"]:
            v = int(val)
            if v == 0:
                masks["wash_machine_schedule"].append(True)
            else:
                tgt = e + v - 1
                masks["wash_machine_schedule"].append(bool(e <= tgt <= l and tgt >= ti))

        return masks

    def reset(self):
        self.total_cost = 0
        # Initialize episode calendar
        if self.episode_length_steps is not None and self.episode_schedule is not None and len(self.episode_schedule) > 0:
            house_id, date_str = self.episode_schedule[self.episode_schedule_cursor % len(self.episode_schedule)]
            self.episode_schedule_cursor += 1
            # Switch to the corresponding household data for this episode
            self.data_interface = DataInterface.from_storenet_ireland_2020(
                house_id=house_id,
                base_dir=self.storenet_base_dir,
                price_profile=self.price_profile,
                steps_per_day=self.steps_per_day,
            )
            if self._seed_value is not None:
                # Ensure EV arrival/departure randomness reproducible for this run
                self.data_interface.seed(self._seed_value)
            self.current_time = self._normalize_date_str(date_str)
            # 供 visualize() / 调试：本回合对应的 StoreNet 住户与日历日（评估 p3 等为「第 1 个评估 episode」的这一天）
            self._current_episode_house_id = house_id
            self._current_episode_date_str = self.current_time
        else:
            self.current_time = self._first_cons_date_str(self.data_interface)
            self._current_episode_house_id = None
            self._current_episode_date_str = self.current_time
        self.current_time_index = 0
        self.current_ev_power = 0
        self.current_battery_power = 0
        # wash_feature = self._get_wash_machine_features(self.current_time_index)
        self.state = {
            'home_load': self.data_interface.get_home_load(self.current_time, self.current_time_index),
            'pv_generation': self.data_interface.get_pv_generation(self.current_time, self.current_time_index),
            'ess_state': 2.4,
            'ev_battery_state': 12,
            'time_index': self.current_time_index,
            'electricity_price': self.data_interface.get_electricity_price(self.current_time, self.current_time_index),
            'temperature': 20,
            'indoor_temp': self.indoor_temp,
            'indoor_temp2': self.indoor_temp2,
            'wash_machine_state': 0,
            # 'start_time': wash_feature['start_time'],
            # 'price_trend': wash_feature['price_trend'],
            'Air_conditioner_power': 0,
            'Air_conditioner_power2': 0,
            'ewh_temp': self.ewh_params['temp_init'],
            'ewh_power': 0,
            **self._calendar_observation_dict(),
        }
        self.ev_battery_record = []   # Reset records
        self.ess_state_record = []
        self.home_load_record = []
        self.pv_generation_record = []
        self.electricity_price_record = []
        self.ev_at_home_record = []
        self.wash_machine_record = []
        self.air_conditioner_power_record = []
        self.air_conditioner_power_record2 = []
        self.ess_actions = []  # Reset ESS action records
        self.ess_charge_pv = []
        self.ess_charge_grid = []
        self.ess_discharge_ev = []
        self.ess_discharge_house = []

        # Reset records
        self.records = {
            'timestamps': [],
            'ev_soc': [],
            'ess_soc': [],
            'grid_power': [],
            'energy_cost': [],
            'daily_costs': [],
            'indoor_temp': [],
            'indoor_temp2': [],
            'outdoor_temp': [],
            'current_daily_cost': 0,
            'total_load': [],
            'user_dissatisfaction': []
        }

        # Reset reward record containers
        self.reward_components = {
            'total': [],
            'energy_cost': [],
            'violation_penalty': [],
            'aging_penalty': [],
            'ess_reward': [],
            'ev_reward': [],
            'user_penalty': [],
            'temp_reward': [],
            'carbon_penalty': [],
            # 'sell': []
        }

        # Reset washing machine related variables
        self.current_period_start = None  # Current period start time
        self.has_run_in_current_period = False  # Whether has run in current period
        self.time_deviation = 0  # Time offset (hours)
        self.wash_machine_state = 0  # Initial state is off
        self.last_action_time = None  # Record start time
        self.remaining_runtime = 0  # Remaining runtime (30-minute steps)
        self.wash_machine_scheduled_start_step = None
        self._wash_price_step_for_penalty = None

        # Reset AC related variables
        self.indoor_temp = 25
        self.indoor_temp2 = 20
        self.outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)
        # Keep the state dict aligned with the latest indoor temperature variables.
        self.state['indoor_temp'] = self.indoor_temp
        self.state['indoor_temp2'] = self.indoor_temp2
        self.state.update(self._calendar_observation_dict())
        self.state.update(self._time_index_observation_dict())
        self.state.update(self._wash_schedule_observation_dict())
        self.state.update(self._carbon_observation_dict())
        # 初始 SOC 来源未知：按 50/50；后续由逐步 bookkeeping 修正
        self._ess_stored_from_pv_kwh = float(self.state['ess_state']) * 0.5
        self._ess_stored_from_grid_kwh = float(self.state['ess_state']) * 0.5
        self.state['ess_charge_pv_fraction'] = self._ess_charge_pv_fraction()

        # Reset records
        self.ewh_temp_record = []
        self.ewh_power_record = []
        self.user_flow_record = []

        return self.state

    def step(self, state, action):

        # === Action physical clipping to ensure environment physical rationality ===
        # --- ESS battery power clipping ---
        ess_soc = state['ess_state']
        if action['battery_power'] < 0:  # Discharge
            max_discharge = min(abs(action['battery_power']), ess_soc / 0.5 * self.discharge_efficiency)
            action['battery_power'] = -max_discharge
        elif action['battery_power'] > 0:  # Charge
            max_charge = min(action['battery_power'], (self.ess_capacity - ess_soc) / 0.5 / self.charge_efficiency)
            action['battery_power'] = max_charge

        # --- EV power clipping ---
        ev_soc = state['ev_battery_state']
        if action['ev_power'] < 0:  # Discharge
            max_discharge = min(abs(action['ev_power']), ev_soc / 0.5 * self.discharge_efficiency)
            action['ev_power'] = -max_discharge
        elif action['ev_power'] > 0:  # Charge
            max_charge = min(action['ev_power'], (self.ev_capacity - ev_soc) / 0.5 / self.charge_efficiency)
            action['ev_power'] = max_charge


        self.current_ev_power=action['ev_power']  # Store current action
        current_dt = datetime.strptime(self.current_time, '%Y-%m-%d') + \
                     timedelta(minutes=30 * self.current_time_index)

        # Update outdoor temperature
        self.outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)

        # Update AC state
        new_air_conditioner_power, self.indoor_temp = self.update_air_conditioner(action['Air_conditioner_set_temp'], self.indoor_temp)
        self.state['Air_conditioner_power'] = new_air_conditioner_power

        new_air_conditioner_power2, self.indoor_temp2 = self.update_air_conditioner(action['Air_conditioner_set_temp2'], self.indoor_temp2)
        self.state['Air_conditioner_power2'] = new_air_conditioner_power2

        # Update washing machine state
        new_wash_machine_state = self.update_wash_machine2(action['wash_machine_schedule'])
        self.state['wash_machine_state'] = new_wash_machine_state

        # wash_feature = self._get_wash_machine_features(self.current_time_index)

        # Update water heater power
        power = self._fuzzy_heating_control(
            action['ewh_set_temp'],
            state['ewh_temp']
        )

        # Update water heater state
        new_ewh_temp, user_flow = self.update_water_heater(
            power,
            state['ewh_temp']
        )

        # Update EV and ESS battery state
        self.state = {
            'home_load': self.data_interface.get_home_load(self.current_time, self.current_time_index),
            'pv_generation': self.data_interface.get_pv_generation(self.current_time, self.current_time_index),
            'ess_state': self.update_ess(action['battery_power'], state['pv_generation']),
            'ev_battery_state': self.update_ev_battery(action['ev_power']),
            'time_index': self.current_time_index,
            'electricity_price': self.data_interface.get_electricity_price(self.current_time, self.current_time_index),
            'temperature': self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index),
            'indoor_temp': self.indoor_temp,
            'indoor_temp2': self.indoor_temp2,
            'wash_machine_state': new_wash_machine_state,
            # 'start_time': wash_feature['start_time'],
            # 'price_trend': wash_feature['price_trend'],
            'Air_conditioner_power': new_air_conditioner_power,
            'Air_conditioner_power2': new_air_conditioner_power2,
            'ewh_temp': new_ewh_temp,
            'ewh_power': power,
            **self._calendar_observation_dict(),
            **self._time_index_observation_dict(),
            **self._wash_schedule_observation_dict(),
            **self._carbon_observation_dict(),
        }

        reward = self.calculate_reward(state, action)
        self.state['ess_charge_pv_fraction'] = self._ess_charge_pv_fraction()
        # 仿真导出：ESS 内累计能量拆分（kWh，与 ess_state 同量纲）
        self.ess_charge_pv.append(self._ess_stored_from_pv_kwh)
        self.ess_charge_grid.append(self._ess_stored_from_grid_kwh)

        done = self.is_terminal_state()

        self.ev_battery_record.append(self.state['ev_battery_state'])
        self.ess_state_record.append(self.state['ess_state'])
        self.home_load_record.append(self.state['home_load'])
        self.pv_generation_record.append(self.state['pv_generation'])
        self.electricity_price_record.append(self.state['electricity_price'])
        self.ev_at_home_record.append(self.is_ev_at_home())
        self.wash_machine_record.append(self.state['wash_machine_state'])
        self.air_conditioner_power_record.append(self.state['Air_conditioner_power'])
        self.air_conditioner_power_record2.append(self.state['Air_conditioner_power2'])

        # Record ESS action
        self.ess_actions.append(action['battery_power'])

        # Record temperature data
        self.records['indoor_temp'].append(self.indoor_temp)
        self.records['indoor_temp2'].append(self.indoor_temp2)
        self.records['outdoor_temp'].append(self.outdoor_temp)

        # Update records
        self.records['timestamps'].append(current_dt)
        self.records['ev_soc'].append(self.state['ev_battery_state'])
        self.records['ess_soc'].append(self.state['ess_state'])

        self.records['total_load'].append(self.total_load_compute())

        # Record water heater data
        self.ewh_temp_record.append(new_ewh_temp)
        self.ewh_power_record.append(state['ewh_power'])
        self.user_flow_record.append(user_flow)

        self.current_time_index += 1  # Add half hour

        if self.current_time_index >= self.steps_per_day:  # If time index reaches full day, advance one day
            self.current_time_index = 0
            # Convert current date string to datetime object
            current_date = datetime.strptime(self.current_time, '%Y-%m-%d')
            # Add one day
            current_date += timedelta(days=1)
            # Convert datetime object back to string format
            self.current_time = current_date.strftime('%Y-%m-%d')

            # Force reset washing machine state (avoid cross-day residue)
            self.wash_machine_used_today = False

        return self.state, reward, done

    def update_ev_battery(self, ev_charge_discharge):
        # If EV just arrived home, update charge state
        if (not self.data_interface.is_ev_at_home(self.current_time, self.current_time_index - 1)
                and self.is_ev_at_home()):
            # Use charge state before last trip to simulate charge state after arriving home
            ev_state_init = self.update_ev_state_after_trip(self.state['ev_battery_state'])
            if ev_charge_discharge > 0:
                new_soc = ev_state_init + (ev_charge_discharge * self.charge_efficiency) * 0.5
            else:
                new_soc = ev_state_init + (ev_charge_discharge / self.discharge_efficiency) * 0.5
        else:
            if ev_charge_discharge > 0:
                new_soc = self.state['ev_battery_state'] + (ev_charge_discharge * self.charge_efficiency) * 0.5
            else:
                new_soc = self.state['ev_battery_state'] + (ev_charge_discharge / self.discharge_efficiency) * 0.5

        # Force SOC boundary
        # min_soc = self.ev_min_charge * 0.8  # Maintain safety margin
        return np.clip(new_soc, 0, self.ev_capacity)

    def total_load_compute(self):
        ev_power = self.current_ev_power
        ess_power = self.current_battery_power
        pv_load = self.state['pv_generation']
        home_load = self.state['home_load']
        air_cond_power = self.state['Air_conditioner_power']
        air_cond_power2 = self.state['Air_conditioner_power2']
        wash_machine_power = self.state['wash_machine_state'] * self.wash_machine_power
        ewh_power = self.state['ewh_power']
        total_load = (home_load + air_cond_power + air_cond_power2 + wash_machine_power + ewh_power + ev_power
                      + ess_power - pv_load)
        return total_load

    def update_ess(self, ess_charge_discharge, pv_generation):
        # assert charge_power * discharge_power == 0, "ESS cannot charge and discharge simultaneously"
        # diff = pv_generation - self.total_load_compute()

        if ess_charge_discharge > 0:  # Charge action = pv + grid
            new_ess = self.state['ess_state'] + ess_charge_discharge * self.charge_efficiency * 0.5
        else:   # Discharge action = home + grid
            new_ess = self.state['ess_state'] + ess_charge_discharge / self.discharge_efficiency * 0.5

        return np.clip(new_ess, 0, self.ess_capacity)

    def _get_period_start(self, dt):
        if dt.hour >= 0:
            return datetime(dt.year, dt.month, dt.day, 0, 0)
        else:
            prev_day = dt - timedelta(days=1)
            return datetime(prev_day.year, prev_day.month, prev_day.day, 0, 0)

    def _get_period_end(self, dt):
        return datetime(dt.year, dt.month, dt.day, 23, 0)

    def update_time_deviation(self, scheduled_time):
        """与最初 multi_agent 版一致：仅根据计划开始时刻相对偏好窗计算偏差（小时）。"""
        pref_start = self.current_period_start.replace(hour=self.wash_machine_preferred_time[0], minute=0)
        pref_end = self.current_period_start.replace(hour=self.wash_machine_preferred_time[1],
                                                     minute=0) - timedelta(hours=self.wash_machine_duration)
        if scheduled_time < pref_start:
            self.time_deviation = (pref_start - scheduled_time).total_seconds() / 3600
        elif scheduled_time > pref_end:
            self.time_deviation = (scheduled_time - pref_end).total_seconds() / 3600
        else:
            self.time_deviation = 0

    def _wash_abs_earliest_start_step(self) -> int:
        """当日允许开洗的最早半步索引（偏好窗起点）。"""
        return int(self.wash_machine_preferred_time[0] * self.steps_per_day / 24)

    def _wash_abs_latest_start_step(self) -> int:
        """当日允许开洗的最晚半步索引（保证洗涤结束不晚于偏好窗终点）。"""
        dur_steps = max(1, int(round(float(self.wash_machine_duration) * self.steps_per_day / 24)))
        latest = int(self.wash_machine_preferred_time[1] * self.steps_per_day / 24) - dur_steps
        e = self._wash_abs_earliest_start_step()
        return max(e, min(self.steps_per_day - 1, latest))

    def _decode_wash_action_to_start_step(self, action_val: int) -> Optional[int]:
        """动作值 0=HOLD；k>=1 → 目标开洗半步 = earliest + k - 1。"""
        if int(action_val) <= 0:
            return None
        return self._wash_abs_earliest_start_step() + int(action_val) - 1

    def _clamp_wash_start_step(self, target: int, cur_idx: int) -> int:
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        t = int(np.clip(int(target), e, l))
        return int(min(l, max(t, cur_idx)))

    def _dt_at_step_index(self, step_index: int) -> datetime:
        return datetime.strptime(self.current_time, "%Y-%m-%d") + timedelta(
            minutes=30 * int(step_index)
        )

    def update_wash_machine2(self, schedule_action):
        """
        洗衣机调度（偏好窗 6–20 内绝对槽位 + 到点真开洗）：

        - 动作取值由 `build_wash_schedule_action_tuple` 定义：**0 = HOLD**（保持当前预约，不取消）；
          **1..N** = 在窗内选择开洗半步索引 `earliest + (k-1)`，新选择**覆盖**旧预约。
        - 到达预约 `time_index` 时**自动**开洗。
        - **每日至少洗一次**：若仍未洗，则（1）当前半步已达「最晚可开洗」索引，或（2）当日最后半步，
          或（3）episode 最后半步（与 `is_terminal_state` 一致且 episode 不长于一日）时**环境强制**完成洗涤：
          - 若当前尚早于最早可开洗半步 → 预约至 `earliest`；
          - 否则**立即开洗**（若已晚于窗仍强制则视为紧急，偏差惩罚由 `update_time_deviation` 反映）。

        运行时长：1 小时 = 2 个半步（remaining_runtime=2）。
        """
        self._wash_price_step_for_penalty = None

        current_dt = datetime.strptime(self.current_time, '%Y-%m-%d') + timedelta(minutes=30 * self.current_time_index)
        schedule_action = int(schedule_action)

        def _mark_wash_start_price_step() -> None:
            self._wash_price_step_for_penalty = int(self.current_time_index)

        if self.remaining_runtime > 0:
            self.wash_machine_state = 1
            self.remaining_runtime -= 1
            return self.wash_machine_state

        current_period_start = self._get_period_start(current_dt)
        if current_period_start != self.current_period_start:
            self.current_period_start = current_period_start
            self.has_run_in_current_period = False
            self.wash_machine_state = 0
            self.wash_machine_scheduled_start_step = None
            self.time_deviation = 0

        if self.has_run_in_current_period:
            self.wash_machine_state = 0
            self.wash_machine_scheduled_start_step = None
            self.time_deviation = 0
            return self.wash_machine_state

        earliest = self._wash_abs_earliest_start_step()
        latest = self._wash_abs_latest_start_step()
        _spd = int(self.steps_per_day)
        _idx = int(self.current_time_index)
        _ep = getattr(self, "episode_length_steps", None)
        _last_step_of_episode = (
            _ep is not None
            and int(_ep) > 0
            and int(_ep) <= _spd
            and _idx >= int(_ep) - 1
        )
        _need_daily = (not self.has_run_in_current_period) and self.remaining_runtime <= 0
        _force = _need_daily and (
            _idx >= latest
            or _idx >= _spd - 1
            or _last_step_of_episode
        )

        def _start_immediate_wash() -> int:
            self.wash_machine_scheduled_start_step = None
            self.wash_machine_state = 1
            self.has_run_in_current_period = True
            self.remaining_runtime = 2
            self.update_time_deviation(current_dt)
            _mark_wash_start_price_step()
            return self.wash_machine_state

        def _schedule_at_step(target_step: int) -> int:
            ts = self._clamp_wash_start_step(target_step, _idx)
            sdt = self._dt_at_step_index(ts)
            self.update_time_deviation(sdt)
            if ts == _idx:
                return _start_immediate_wash()
            self.wash_machine_scheduled_start_step = int(ts)
            self.wash_machine_state = 0
            return self.wash_machine_state

        if _force:
            if _idx < earliest:
                return _schedule_at_step(earliest)
            return _start_immediate_wash()

        # 已有待执行：先到点自动开洗
        if self.wash_machine_scheduled_start_step is not None:
            if self.current_time_index >= int(self.wash_machine_scheduled_start_step):
                self.update_time_deviation(current_dt)
                self.wash_machine_state = 1
                self.has_run_in_current_period = True
                self.remaining_runtime = 2
                self.wash_machine_scheduled_start_step = None
                _mark_wash_start_price_step()
                return self.wash_machine_state

            if schedule_action == 0:
                self.wash_machine_state = 0
                return self.wash_machine_state

            tgt = self._decode_wash_action_to_start_step(schedule_action)
            if tgt is None:
                self.wash_machine_state = 0
                return self.wash_machine_state
            return _schedule_at_step(tgt)

        # 无待执行
        if schedule_action == 0:
            self.wash_machine_state = 0
            self.time_deviation = 0
            return self.wash_machine_state

        tgt = self._decode_wash_action_to_start_step(schedule_action)
        if tgt is None:
            self.wash_machine_state = 0
            self.time_deviation = 0
            return self.wash_machine_state
        return _schedule_at_step(tgt)

    def update_air_conditioner(self, set_temp, indoor_temp):
        """Update AC power"""
        # Calculate difference between set temperature and current indoor temperature
        temp_diff = set_temp - indoor_temp
        # Fuzzy control rules: determine power based on temperature difference
        if temp_diff > 0:  # Heating mode
            # Define fuzzy control rules for heating mode
            rules = [
                {'range': (0, 0.5), 'power': 0},
                {'range': (0.5, 1), 'power': 0.5},  # 0.5 enables precise control
                {'range': (1, 2), 'power': 1.0},
                {'range': (2, 3), 'power': 1.5},
                {'range': (3, 4), 'power': 2.0},
                {'range': (4, np.inf), 'power': 3.0}
            ]
        else:
            # Define fuzzy control rules for cooling mode
            rules = [
                {'range': (-0.5, 0), 'power': 0},
                {'range': (-1, -0.5), 'power': 0.5},  # 0.5 enables precise control
                {'range': (-2, -1), 'power': 1.0},
                {'range': (-3, -2), 'power': 1.5},
                {'range': (-4, -3), 'power': 2.0},
                {'range': (-np.inf, -4), 'power': 3.0}
            ]

        # Find corresponding power based on temperature difference
        power = 0  # Default power
        for rule in rules:
            if rule['range'][0] <= temp_diff < rule['range'][1]:
                power = rule['power']
                break

        # Dynamically calculate temperature change rate
        max_power = 3.0  # AC maximum power
        efficiency = power / max_power if max_power > 0 else 0  # Calculate efficiency at current power
        temp_change = self.temp_change_rate * efficiency * temp_diff  # Calculate temperature change based on power and temperature difference

        # Simulate random disturbance from user behavior
        # Optional disturbance term for user-driven AC fluctuations.
        # When `ac_user_behavior_std=0`, this becomes deterministic.
        user_behavior = np.random.normal(0, self.ac_user_behavior_std)
        temp_change += user_behavior  # Add random disturbance to temperature change

        # When AC is off (power == 0), indoor temperature gradually approaches outdoor temperature
        if power == 0:
            # Rate at which indoor temperature approaches outdoor temperature can be adjusted
            temp_change += (self.outdoor_temp - indoor_temp) * 0.4 * self.temp_change_rate

        # Update indoor temperature
        indoor_temp += temp_change

        # Ensure indoor temperature is within reasonable range
        indoor_temp = np.clip(indoor_temp, 10, 40)

        # Ensure AC power is within reasonable range
        power = np.clip(power, 0, 3.0)

        return power, indoor_temp

    def update_water_heater(self, power, current_temp):
        """Update water heater state (physical model)"""
        params = self.ewh_params
        delta_t = 0.5  # Half-hour time interval

        # # Generate random water usage (L/h)
        # flow_rate = np.random.uniform(1, 2) if np.random.rand() < 0.3 else 0  # Assume 30% probability of water usage

        # Define peak water usage time periods
        peak_morning = 6 <= self.current_time_index / 2 <= 9  # Morning peak 6-9 AM
        peak_evening = 18 <= self.current_time_index / 2 <= 22  # Evening peak 6-10 PM

        # Set different water usage probabilities and ranges based on time period
        if peak_morning or peak_evening:
            # Peak period water usage probability and amount
            if peak_morning:
                # Morning peak has larger water usage and higher probability
                flow_rate_prob = 0.6  # 60% probability of water usage
                flow_rate_min, flow_rate_max = 3, 5  # 3-5 L/h
            else:
                # Evening peak has slightly smaller water usage and slightly lower probability
                flow_rate_prob = 0.5  # 50% probability of water usage
                flow_rate_min, flow_rate_max = 2, 4  # 2-4 L/h

            # Peak periods have higher water usage probability
            if np.random.rand() < flow_rate_prob:
                flow_rate = np.random.uniform(flow_rate_min, flow_rate_max)
            else:
                flow_rate = 0
        else:
            # Off-peak period water usage probability and amount
            flow_rate_prob = 0.2  # 20% probability of water usage
            if np.random.rand() < flow_rate_prob:
                # Off-peak periods have smaller water usage
                flow_rate = np.random.uniform(0.5, 1.5)  # 0.5-1.5 L/h
            else:
                flow_rate = 0

        # Special handling: occasionally have slightly larger water usage during off-peak periods
        if not (peak_morning or peak_evening) and np.random.rand() < 0.1:
            flow_rate = np.random.uniform(1.5, 2)  # 1.5-2 L/h
        # === Water usage logic end ===

        # Environment parameters
        env_temp = self.state['temperature']
        temp_cold = max(18, env_temp - 2)  # Minimum 18°C cold water
        # temp_cold = 18 # Minimum 18°C cold water

        new_temp = current_temp  # Initialize new temperature

        # Case 1: Temperature change when injecting cold water
        if flow_rate > 0:
            # Calculate volume change corresponding to water usage (assuming total tank volume unchanged)
            used_water_volume = flow_rate * delta_t   # Assume flow rate unit is L/h, convert to L
            # Inject same volume of cold water
            injected_cold_volume = used_water_volume

            # Calculate new temperature using mixing effect
            remaining_hot_volume = params['volume'] - used_water_volume
            new_temp = (current_temp * remaining_hot_volume + temp_cold * injected_cold_volume) / params['volume']

        # Case 2: Temperature rise when heating
        if power > 0:
            hour = self.state['time_index'] // 2
            efficiency = 0.9 if (6 <= hour <= 9 or 18 <= hour <= 22) else 0.8
            # Convert power from kW to W
            power_in_watts = power * 1000 * efficiency

            # Calculate temperature rise from heating
            heat_input = power_in_watts * 3600 * delta_t  # Energy input (J)
            temp_gain = heat_input / (params['mass'] * 4180)  # Temperature rise (°C)
            new_temp += temp_gain

        # Case 3: Natural cooling when not heating
        else:
            # Calculate temperature drop from natural cooling    Cooling coefficient k
            cooling_coefficient = params['h'] * params['surface_area'] / (params['mass'] * 4180)
            new_temp = env_temp + (new_temp - env_temp) * np.exp(-cooling_coefficient * delta_t * 3600)

        # Update parameters
        params['temp_min'] = env_temp
        new_temp = np.clip(new_temp, params['temp_min'], params['temp_max'])

        return new_temp, flow_rate

    def _fuzzy_heating_control(self, set_tem, current_tem):
        # Fuzzy control logic: determine heating power based on difference between target temperature and current temperature
        # temp_diff: target temperature - current temperature
        temp_diff = set_tem - current_tem
        hour = self.state['time_index'] // 2  # Get current hour

        # Dynamically adjust control rules (more aggressive during peak periods)
        if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak period
            rules = [
                {'range': (-np.inf, -3), 'power': 0.0},
                {'range': (-3, 1), 'power': 0.0},
                {'range': (1, 4), 'power': 0.4 + 0.1 * max(0, temp_diff-1)},  # Dynamic proportion
                {'range': (4, 6), 'power': 1.0},
                {'range': (6, np.inf), 'power': 1.2}  # Allow short-term over-power
            ]
        else:  # Off-peak period
            rules = [
                {'range': (-np.inf, -5), 'power': 0.0},
                {'range': (-5, 2), 'power': 0.0},
                {'range': (2, 5), 'power': 0},  # Gentle heating
                {'range': (5, 8), 'power': 0.2},  # Originally 0
                {'range': (8, np.inf), 'power': 0.5}  # Originally 0.5
            ]

        # rules = [
        #     {'range': (-np.inf, -3), 'power': 0.0},
        #     {'range': (-3, 1), 'power': 0.0},
        #     {'range': (1, 4), 'power': 0.4 + 0.1 * max(0, temp_diff - 1)},  # Dynamic proportion
        #     {'range': (4, 6), 'power': 1.0},
        #     {'range': (6, np.inf), 'power': 1.2}  # Allow short-term over-power
        # ]

        for rule in rules:
            if rule['range'][0] <= temp_diff < rule['range'][1]:
                return min(max(rule['power'], 0.0), 1.2)  # Power clamping
        return 0.0  # Default no heating

    def _tou_price_min_max(self) -> Tuple[float, float]:
        """
        ESS/EV 套利项与洗衣机电价项共用的归一化区间。
        优先使用当日 TOU 缓存向量的 min/max（与 lee2020 默认 0.06~0.14、自定义 tou_hourly_values 一致），
        否则按 price_profile 使用 legacy [0.2,0.8] 或 lee2020 [0.06,0.14]。
        """
        di = self.data_interface
        cache = getattr(di, "_tou_price_cache", None)
        if cache is not None:
            arr = np.asarray(cache, dtype=np.float64).ravel()
            if arr.size > 0:
                return float(np.min(arr)), float(np.max(arr))
        profile = getattr(di, "price_profile", "legacy")
        if profile == "lee2020":
            return 0.06, 0.14
        return 0.2, 0.8

    def _load_carbon_intensity_table(self, project_root: str) -> None:
        """Load 30-min ROI carbon CSV (UTC); disables carbon term if missing/invalid."""
        self._carbon_df = None
        if not self.use_carbon_in_reward:
            return
        if not self._carbon_intensity_csv_path:
            self.use_carbon_in_reward = False
            return
        p = self._carbon_intensity_csv_path
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(project_root, p))
        if not os.path.isfile(p):
            print(f"⚠️ Carbon intensity CSV not found ({p}); carbon term disabled.")
            self.use_carbon_in_reward = False
            return
        try:
            df = pd.read_csv(p)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            gcol = None
            for name in (
                "carbon_intensity_gco2_per_kwh",
                "carbon_intensity_gco2_per_kwh_roi",
                "carbon_intensity_gco2_per_kwh_nsw",
            ):
                if name in df.columns:
                    gcol = name
                    break
            if gcol is None:
                raise ValueError("no carbon intensity column (gCO2/kWh) in CSV")
            df["carbon_intensity_kgco2_per_kwh"] = pd.to_numeric(df[gcol], errors="coerce") / 1000.0
            df = df.sort_values("timestamp_utc").reset_index(drop=True)
            self._carbon_df = df
            self._carbon_kg_per_kwh_max = float(df["carbon_intensity_kgco2_per_kwh"].max())
            print(f"✅ Loaded carbon intensity for reward: {p} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Failed to load carbon CSV ({p}): {e}; carbon term disabled.")
            self._carbon_df = None
            self.use_carbon_in_reward = False

    def _simulation_time_utc(self) -> pd.Timestamp:
        """Calendar date + half-hour step interpreted as Europe/Dublin, then UTC."""
        naive = datetime.strptime(self.current_time, "%Y-%m-%d") + timedelta(
            minutes=int(30 * int(self.current_time_index))
        )
        ts = pd.Timestamp(naive)
        try:
            loc = ts.tz_localize("Europe/Dublin", ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            loc = ts.tz_localize("Europe/Dublin", ambiguous=False, nonexistent="shift_forward")
        return loc.tz_convert("UTC")

    def _carbon_intensity_kg_per_kwh_at_now(self) -> float:
        if self._carbon_df is None or len(self._carbon_df) == 0:
            return float(self._carbon_fallback_kg_per_kwh)
        utc_ts = self._simulation_time_utc()
        cdf = self._carbon_df
        mask = cdf["timestamp_utc"] <= utc_ts
        if mask.any():
            return float(cdf.loc[mask, "carbon_intensity_kgco2_per_kwh"].iloc[-1])
        return float(cdf["carbon_intensity_kgco2_per_kwh"].iloc[0])

    def _carbon_observation_dict(self) -> Dict[str, float]:
        """Normalized marginal intensity [0,1] for policy input (same clock as reward carbon lookup)."""
        if not self.use_carbon_in_reward or self._carbon_df is None or len(self._carbon_df) == 0:
            return {"carbon_intensity_norm": 0.5}
        ci = float(self._carbon_intensity_kg_per_kwh_at_now())
        denom = max(float(self._carbon_kg_per_kwh_max), 1e-8)
        return {"carbon_intensity_norm": float(np.clip(ci / denom, 0.0, 1.0))}

    def calculate_reward(self, state, action):
        # Define reward

        # 将绝对电价线性归一化到约 [0,1]，再以 0.5 为「相对谷/峰」分界，适配 lee2020(0.06~0.14) 与 legacy(0.2~0.8)。
        price_raw = float(state.get('electricity_price', 0.0))
        p_min, p_max = self._tou_price_min_max()
        price = (price_raw - p_min) / max(1e-8, (p_max - p_min))
        price = float(np.clip(price, 0.0, 1.0))

        # 1. Constraint penalty
        violation = 0
        # violation += max(0.1*self.ess_capacity-self.state['ess_state'],
        #                 self.state['ess_state']-0.9*self.ess_capacity, 0) ** 2 * 20
        #
        # violation += max(0.1 * self.ev_capacity - self.state['ev_battery_state'],
        #                 self.state['ev_battery_state'] - 0.9 * self.ev_capacity, 0) ** 2 * 20

        # # EV constraint (minimum charge)
        if self.data_interface.is_ev_departing_soon(self.current_time, self.current_time_index):
            ev_violation = max(0, self.ev_min_charge - self.state['ev_battery_state'])
            # violation += np.clip(ev_violation ** 2, 0, 500)  # Add numerical clipping
            violation += ev_violation ** 2

        # 2. Grid cost calculation (half hour)
        ev_charge = max(action['ev_power'], 0)
        ev_discharge = max(-action['ev_power'], 0)
        battery_charge = max(action['battery_power'], 0)
        battery_discharge = max(-action['battery_power'], 0)

        # Calculate total consumption and total generation (kW)
        total_consumption = (
                state['home_load']
                + ev_charge
                + battery_charge
                + state['Air_conditioner_power']
                + state['Air_conditioner_power2']
                + state['wash_machine_state'] * self.wash_machine_power
                + state['ewh_power']
        )

        total_generation = (
                state['pv_generation']
                + ev_discharge
                + battery_discharge
        )
        # Calculate net demand (kW)
        net_demand = total_consumption - total_generation  # This demand is the interaction with the grid

        # Convert to energy (kWh) and split purchase/sale
        purchase_kwh = max(net_demand, 0) * 0.5  # This calculates ideal cost, not actual cost
        sell_kwh = max(-net_demand, 0) * 0.5

        # Calculate energy cost (considering electricity sale price discount)
        energy_cost = (
                purchase_kwh * state['electricity_price']
                - sell_kwh * state['electricity_price'] * 0.9  # Assume sale price is 0.9 of purchase price
        )

        # Net grid energy (kWh) for this step — same duration as purchase_kwh / sell_kwh
        step_h = 24.0 / float(self.steps_per_day)
        grid_import_kwh = max(net_demand, 0.0) * step_h
        grid_export_kwh = max(-net_demand, 0.0) * step_h
        self.current_step_grid_import_kwh = float(grid_import_kwh)
        self.current_step_grid_export_kwh = float(grid_export_kwh)

        carbon_penalty_weighted = 0.0
        emission_kg = 0.0
        if self._carbon_df is not None and len(self._carbon_df) > 0:
            ci_kg = self._carbon_intensity_kg_per_kwh_at_now()
            # kg CO2: import attributed + export credited (same marginal factor)
            emission_kg = ci_kg * (grid_import_kwh - grid_export_kwh)
        self.current_step_emission_kg = float(emission_kg)
        if self.use_carbon_in_reward and self._carbon_df is not None:
            carbon_penalty_weighted = self.carbon_weight * emission_kg

        self.total_cost += energy_cost
        self.records['energy_cost'].append(self.total_cost)

        self.current_step_cost = energy_cost

        # ESS 能量来源记账（self.state 已为 step 后新 SOC；与 net_demand / total_consumption 一致）
        self._update_ess_energy_source_bookkeeping(state, action, float(total_consumption), float(net_demand))

        # 3. Battery aging proxy (Version A: throughput/FEC-like)
        # Use action charge/discharge power as a proxy for equivalent cycling stress.
        # Units:
        #   - battery_power / ev_power are in kW
        #   - step duration in hours is 24/steps_per_day
        #   - throughput in kWh is abs(power) * dt
        #   - FEC proxy ~ throughput / (2 * Q_nom)
        dt_h = 24.0 / float(self.steps_per_day) if getattr(self, "steps_per_day", None) else 0.5
        dt_h = float(dt_h)
        throughput_ess_kwh = abs(float(action.get('battery_power', 0.0))) * dt_h
        throughput_ev_kwh = abs(float(action.get('ev_power', 0.0))) * dt_h
        denom_ess = max(1e-8, 2.0 * float(self.ess_capacity))
        denom_ev = max(1e-8, 2.0 * float(self.ev_capacity))
        fec_inc_ess = throughput_ess_kwh / denom_ess
        fec_inc_ev = throughput_ev_kwh / denom_ev
        aging_penalty = self.ess_aging_weight * fec_inc_ess + self.ev_aging_weight * fec_inc_ev

        # 3. ESS and EV charge/discharge guidance reward

        # ESS reward: encourage low price charging and high price discharging
        ess_reward = 0
        soc = self.state['ess_state'] / self.ess_capacity
        if price < 0.5:
            # ess_reward += -action['battery_power'] * (price - 0.5) * (1-soc) * 10
            ess_reward += -action['battery_power'] * (price - 0.5) * 20
        elif price > 0.5:
            # ess_reward += -action['battery_power'] * (price - 0.5) * soc * 10
            ess_reward += -action['battery_power'] * (price - 0.5) * 20
        else:
            ess_reward += -action['battery_power'] * (soc - 0.8) * 20
        # ess_reward = np.tanh(ess_reward / 10) * 3  # Use tanh to compress amplitude

        # EV reward: encourage low price charging and high price discharging
        ev_reward = 0
        # Get current SOC ratio
        soc_ev = state['ev_battery_state'] / self.ev_capacity
        if price < 0.5:
            # ev_reward += -action['ev_power'] * (price - 0.5) * (1-soc_ev) * 10
            ev_reward += -action['ev_power'] * (price - 0.5) * 20
        elif price > 0.5:
            # ev_reward += -action['ev_power'] * (price - 0.5) * soc_ev * 10
            ev_reward += -action['ev_power'] * (price - 0.5) * 20
        else:
            ev_reward += -action['ev_power'] * (soc_ev - 0.8) * 20
        # ev_reward = np.tanh(ev_reward / 10) * 3  # Use tanh to compress amplitude

        # 4. User dissatisfaction penalty
        user_dissatisfaction_penalty = 0
        user_dissatisfaction_penalty = (self.user_satisfaction_weight0 * self.calculate_user_dissatisfaction0() +
                                        self.user_satisfaction_weight1 * self.calculate_user_dissatisfaction1() +
                                    self.user_satisfaction_weight2 * self.calculate_user_dissatisfaction2(state, action))
        # user_dissatisfaction_penalty = np.clip(user_dissatisfaction_penalty, -20, 20)

        temp_reward = self.calculate_temp_reward(state['ewh_temp'])
        # temp_reward = np.clip(temp_reward, -5, 5)  # Limit temperature reward range

        # 5. Combined reward
        # 优先级经验：舒适度(用户+温度+约束) > 电费 > 老化 > 碳（carbon_weight 宜明显小于 energy_weight）
        reward = (
                - self.energy_weight * energy_cost
                - self.violation_weight * violation    # EV and ESS upper and lower bound constraints
                - aging_penalty
                + self.ess_weight * ess_reward
                + self.ev_weight * ev_reward
                - user_dissatisfaction_penalty   # User dissatisfaction penalty
                + self.temp_weight * temp_reward
                - carbon_penalty_weighted
        )



        # # New exploration reward (prevent premature convergence)
        # if np.random.rand() < 0.1:  # 10% probability add noise
        #     reward += np.random.normal(0, 5)

        # Add records after calculating reward
        reward_breakdown = {
            'total': reward,
            'energy_cost': - self.energy_weight * energy_cost,
            'violation_penalty': - self.violation_weight * violation,
            'aging_penalty': -aging_penalty,
            'ess_reward': self.ess_weight * ess_reward,
            'ev_reward': self.ev_weight * ev_reward,
            'user_penalty': - user_dissatisfaction_penalty,
            'temp_reward': self.temp_weight * temp_reward,
            'carbon_penalty': -carbon_penalty_weighted,
        }

        for key in self.reward_components:
            self.reward_components[key].append(reward_breakdown[key])

        #     # Add numerical stability check
        # if not np.isfinite(reward):
        #     reward = -10  # Fallback handling for abnormal reward values

        return reward

    def _ess_charge_pv_fraction(self) -> float:
        t = self._ess_stored_from_pv_kwh + self._ess_stored_from_grid_kwh
        if t <= 1e-9:
            return 0.5
        return float(np.clip(self._ess_stored_from_pv_kwh / t, 0.0, 1.0))

    def _sync_ess_stored_sources_to_soc(self) -> None:
        """ bookkeeping 与 ess_state 数值漂移时按比例拉回 """
        soc = float(self.state.get('ess_state', 0.0))
        t = self._ess_stored_from_pv_kwh + self._ess_stored_from_grid_kwh
        if soc <= 1e-9:
            self._ess_stored_from_pv_kwh = 0.0
            self._ess_stored_from_grid_kwh = 0.0
            return
        if t <= 1e-9:
            self._ess_stored_from_pv_kwh = soc * 0.5
            self._ess_stored_from_grid_kwh = soc * 0.5
            return
        if abs(t - soc) > 1e-3:
            k = soc / t
            self._ess_stored_from_pv_kwh *= k
            self._ess_stored_from_grid_kwh *= k

    def _update_ess_energy_source_bookkeeping(
        self,
        state: Dict,
        action: Dict,
        total_consumption: float,
        _net_demand: float,
    ) -> None:
        """
        区分 ESS 中能量来自光伏还是电网（论文叙事：序贯归因）。

        **充电（序贯）**：先假定光伏功率优先满足「固定负荷」
        （总用电 − ESS 充电功率，即家储以外设备），
        再令 `pv_surplus = max(0, PV − load_fixed)`；ESS 充电功率中
        优先由 `pv_surplus` 承担，不足部分归因于电网。
        将本步 ΔSOC 按「充电功率中 PV 占比」拆入 _ess_stored_from_pv / _ess_stored_from_grid。

        **放电**：仍按当前池内两桶比例扣减（与充电归因独立）。

        **初始 SOC**：reset 时 50/50；数值漂移由 _sync 对齐。
        """
        prev_ess = float(state['ess_state'])
        new_ess = float(self.state['ess_state'])
        delta_soc = new_ess - prev_ess

        if delta_soc > 1e-9:
            bc = max(0.0, float(action.get('battery_power', 0.0)))
            if bc > 1e-12:
                load_fixed = float(total_consumption) - bc
                pv_kw = float(state.get("pv_generation", 0.0))
                # 光伏先满足固定负荷，剩余功率再供 ESS 充电
                pv_surplus_kw = max(0.0, pv_kw - max(0.0, load_fixed))
                from_pv_power = min(bc, pv_surplus_kw)
                frac_pv = float(from_pv_power / bc)
            else:
                frac_pv = 0.5
            self._ess_stored_from_pv_kwh += float(delta_soc) * frac_pv
            self._ess_stored_from_grid_kwh += float(delta_soc) * (1.0 - frac_pv)
        elif delta_soc < -1e-9:
            chem_out = -delta_soc
            tot = self._ess_stored_from_pv_kwh + self._ess_stored_from_grid_kwh
            if tot > 1e-12:
                fp = self._ess_stored_from_pv_kwh / tot
            else:
                fp = 0.5
            self._ess_stored_from_pv_kwh = max(0.0, self._ess_stored_from_pv_kwh - chem_out * fp)
            self._ess_stored_from_grid_kwh = max(0.0, self._ess_stored_from_grid_kwh - chem_out * (1.0 - fp))

        self._sync_ess_stored_sources_to_soc()

    def calculate_user_dissatisfaction0(self):
        """AC2 discomfort: quadratic penalty outside deadband."""
        return self._quadratic_deadband_discomfort(
            value=float(self.indoor_temp2),
            target=float(self.user_temp_preference2),
            deadband=2.0,
            scale=1.0,
            cap=500.0,
        )

    def calculate_user_dissatisfaction1(self):
        """AC1 discomfort: quadratic penalty outside deadband."""
        return self._quadratic_deadband_discomfort(
            value=float(self.indoor_temp),
            target=float(self.user_temp_preference),
            deadband=2.0,
            scale=1.0,
            cap=500.0,
        )

    def _wash_electricity_price_penalty(self, absolute_price: float) -> float:
        """
        洗衣开洗时刻电价：在当日「合法开洗半步区间」[earliest, latest] 上取绝对电价 min/max，
        将当前开洗电价线性归一化到 [0,1]，再 (norm-0.5)*10。最低价→负贡献（降低不满）、最高价→正贡献。
        与评估侧 `Wash_Window_Price_Score`（窗内相对低价程度）一致。窗内电价近似常数时返回 0。
        """
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        date_str = self.current_time
        prices = [
            float(self.data_interface.get_electricity_price(date_str, s))
            for s in range(e, l + 1)
        ]
        p_lo, p_hi = min(prices), max(prices)
        if p_hi - p_lo < 1e-10:
            return 0.0
        p = float(absolute_price)
        norm = (p - p_lo) / (p_hi - p_lo)
        norm = float(np.clip(norm, 0.0, 1.0))
        return (norm - 0.5) * 10.0

    def calculate_user_dissatisfaction2(self, state, action):
        """
        洗衣机相关惩罚：
        - 电价项：仅在本步**实际开始洗涤**时计入；惩罚为当日偏好窗内合法开洗槽上的**相对电价**（非全日 TOU 极值）。
          使用 `update_wash_machine2` 写入的 `_wash_price_step_for_penalty` 取价。
          仅预约未开洗当步**不计**电价惩罚。
        - 时间项：wash_time_penalty_coef * time_deviation（小时），由 update_time_deviation 维护。
        """
        price_penalty = 0.0
        if self._wash_price_step_for_penalty is not None:
            day_steps = int(getattr(self, "steps_per_day", 48))
            step_idx = int(np.clip(int(self._wash_price_step_for_penalty), 0, max(0, day_steps - 1)))
            price = float(
                self.data_interface.get_electricity_price(self.current_time, step_idx)
            )
            price_penalty = float(self.wash_price_penalty_coef) * self._wash_electricity_price_penalty(price)
        time_penalty = float(self.wash_time_penalty_coef) * float(self.time_deviation)
        return float(time_penalty + price_penalty)

    def calculate_temp_reward(self, current_temp):
        # Continuous EWH discomfort model (deadband + quadratic penalty),
        # then convert to reward by taking the negative value.
        # 训练侧比评估侧 model_evaluation 中的 EWH「舒适带」更严（窄 deadband + 更大 strict），
        # 促使策略在更紧的温控目标下运行；评估仍用较宽的 50–60 / 40–50 映射舒适度得分。
        hour = self.state['time_index'] // 2
        if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak water usage hours
            target, low, high = 55, 54, 56
            strict_factor = 3.0
        else:
            target, low, high = 45, 44, 46
            strict_factor = 1.5
        deadband = 0.5 * float(high - low)
        discomfort = self._quadratic_deadband_discomfort(
            value=float(current_temp),
            target=float(target),
            deadband=deadband,
            scale=strict_factor,
            cap=500.0,
        )
        return -float(discomfort)

    @staticmethod
    def _quadratic_deadband_discomfort(value, target, deadband=2.0, scale=1.0, cap=500.0):
        """Standard comfort model: zero in deadband, quadratic outside."""
        exceed = max(0.0, abs(float(value) - float(target)) - float(deadband))
        return float(min(float(scale) * (exceed ** 2), float(cap)))

    def is_terminal_state(self):
        # Episode termination for StoreNet 1-day setup:
        if self.episode_length_steps is not None and self.episode_length_steps > 0:
            # done is evaluated before time index increment in step()
            return self.current_time_index >= (self.episode_length_steps - 1)

        # Schedule-less rolling episode: end after last calendar day present in pivot cons_data
        return self.current_time > self._last_cons_date_str(self.data_interface)

    def seed(self, seed=None):
        self._seed_value = int(seed) if seed is not None else None
        self.np_random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        if hasattr(self.data_interface, "seed"):
            self.data_interface.seed(seed)

    def is_ev_at_home(self):
        return self.data_interface.is_ev_at_home(self.current_time, self.current_time_index)

    def update_ev_state_after_trip(self, current_ev_soc):
        # Generate daily mileage (log-normal distribution)
        avg_daily_km = self.ev_daily_km_mean  # Average daily mileage (unit: km)
        km_std = self.ev_daily_km_sigma  # Log-normal sigma
        daily_km = np.random.lognormal(mean=np.log(avg_daily_km), sigma=km_std)

        # Assume energy consumption per km is 0.2 kWh
        energy_consumption_per_km = 0.2  # kWh/km

        # Calculate charge state when arriving home
        soc_home = current_ev_soc - daily_km * energy_consumption_per_km

        # Ensure charge state is within reasonable range (0-100%)
        soc_home = np.clip(soc_home, 0, 100)

        return soc_home

    def _find_contiguous_segments(self, bool_list):
        """Detect contiguous time periods where value is True"""
        segments = []
        start_idx = None
        for i, value in enumerate(bool_list):
            if value and start_idx is None:
                start_idx = i
            elif not value and start_idx is not None:
                segments.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:  # Handle last segment
            segments.append((start_idx, len(bool_list) - 1))
        return segments

    def reward_shape(self, progress):
        """
        Dynamically adjust reward function coefficients based on training progress
        :param progress: Training progress, range from 0 to 1
        :return: Weights for each reward component
        """

        # self.energy_weight = 5.0 * (1 - 0.8*progress)  # Linear decay
        # self.temp_weight = 1 / (1 + np.exp(-10*(progress-0.3)))  # S-shaped growth
        self.violation_weight = 3-2*progress  # April 29 attempt this method, try saving model snapshots if ineffective
        # self.user_satisfaction_weight1 = 0.3 + 0.7*progress
        # self.user_satisfaction_weight2 = 0.1 + 0.4*progress
        # self.ess_weight = 3.0 + 2*progress
        # self.ev_weight = 1

    def save_cost_data(self):
        """Save cost data to CSV file"""
        # Create results directory
        results_dir = "model/cost_results"
        os.makedirs(results_dir, exist_ok=True)

        # Create unique filename (include timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(results_dir, f"cost_data_{timestamp}.csv")

        # Write data to CSV file
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(["Timestamp", "Energy Cost"])

            # Write cost data for each time step
            for i, (ts, cost) in enumerate(zip(self.records['timestamps'], self.records['energy_cost'])):
                writer.writerow([ts.strftime('%Y-%m-%d %H:%M:%S'), cost])

        print(f"Cost data saved to: {csv_filename}")

    def save_episode_costs(self):
        """Save total cost for each episode to CSV file"""
        # Create results directory
        results_dir = "model/episode_cost_results"
        os.makedirs(results_dir, exist_ok=True)

        # Create unique filename (include timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"episode_costs_{timestamp}.csv")

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(["Episode", "Total Cost"])

            # Write cost data for each episode
            for episode, cost in enumerate(self.episode_costs):
                writer.writerow([episode + 1, cost])

        print(f"Episode cost data saved to: {filename}")

    def visualize(self):
        # ===== First canvas: EV SOC change plot and electricity price plot =====
        fig = plt.figure(figsize=(20, 5))
        # 说明：本函数绘制的是「当前 env 中已记录的这一条轨迹」。model_evaluation 仅在第 1 个评估 episode 结束后调用一次，
        # 因此 p3.png 对应该次 reset 选中的 schedule 首条（第 1 天），而非 500 天中的某一天随机一天。
        _hid = getattr(self, "_current_episode_house_id", None)
        _ds = getattr(self, "_current_episode_date_str", None)
        if self.records.get("timestamps"):
            t0 = self.records["timestamps"][0]
            day_str = t0.strftime("%Y-%m-%d")
        else:
            day_str = str(_ds) if _ds else "?"
        _day_type = _calendar_day_label_cn(day_str)
        viz_title = f"EV SOC & Price  |  {day_str}"
        if _day_type:
            viz_title += f"  |  {_day_type}"
        if _hid is not None:
            viz_title += f"  |  住户: {_hid}"
        _print_extra = f", 住户: {_hid}" if _hid else ""
        _type_extra = f", {_day_type}" if _day_type else ""
        print(f"[visualize] 保存 p3.png 等环境图 → 日历日: {day_str}{_type_extra}{_print_extra}")

        ax1 = plt.subplot(1, 1, 1)
        ev_soc = np.array(self.ev_battery_record, dtype=np.float32)
        ev_soc[~np.array(self.ev_at_home_record)] = np.nan  # Set away periods to NaN

        mpl_dates = mdates.date2num(self.records['timestamps'])

        # Plot EV SOC curve
        (ev_line,) = ax1.plot(mpl_dates, ev_soc, color='blue', label='电动汽车SOC')
        ax1.set_ylabel('EV SOC (kWh)')
        ax1.set_title(viz_title)

        # Plot electricity price curve (right axis)
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax1_price.set_ylabel('Price ($/kWh)')
        ax1_price.legend(loc='upper right')

        # Segmentally plot filled areas for at-home time periods
        # 绿色阴影：EV 在家且处于可调度时段（与 is_ev_at_home / 动作掩码一致；离家时段 SOC 已置 NaN 不连线）
        home_segments = self._find_contiguous_segments(self.ev_at_home_record)
        for start, end in home_segments:
            segment_dates = mdates.date2num(self.records['timestamps'][start:end + 1])
            ax1.fill_between(segment_dates, 0, 1, color='green', alpha=0.3, transform=ax1.get_xaxis_transform())
        home_legend_patch = Patch(facecolor='green', alpha=0.3, edgecolor='none', label='EV在家（可调度）')
        ax1.legend(handles=[ev_line, home_legend_patch], loc='upper left')

        # Set time axis format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p3.png')
        plt.close(fig)

        # ===== Second canvas: ESS charge/discharge power, PV generation and electricity price change plot =====
        plt.figure(figsize=(20, 10))

        # Ensure horizontal coordinate ranges are consistent for both subplots
        min_date = mpl_dates[0]
        max_date = mpl_dates[-1]

        # ESS charge/discharge power plot
        ax2_1 = plt.subplot(2, 1, 1)
        if len(self.ess_actions) != len(mpl_dates):
            print(
                f"Warning: Length mismatch between ess_actions ({len(self.ess_actions)}) and timestamps ({len(mpl_dates)}).")
            if len(self.ess_actions) < len(mpl_dates):
                self.ess_actions.append(0)
            else:
                self.ess_actions = self.ess_actions[:len(mpl_dates)]
        ess_actions = self.ess_actions

        ess_charge_power = [max(power, 0) for power in ess_actions]
        ess_discharge_power = [min(power, 0) for power in ess_actions]

        # Plot ESS charge/discharge bar chart, adjust bar width
        ax2_1.bar(mpl_dates, ess_charge_power, width=0.015, color='#05B9E2',
                  label='ESS Charging Power')  # Charging color darker, bar width adjusted
        ax2_1.bar(mpl_dates, ess_discharge_power, width=0.015, color='#FFBE7A', label='ESS Discharging Power')

        # Draw horizontal line at 0 scale
        ax2_1.axhline(0, color='black', linewidth=0.8, linestyle='--')

        # Set left axis range
        ax2_1.set_ylim(-5, 5)

        ax2_1.set_ylabel('Power (kW)')
        ax2_1.set_title('ESS Charging/Discharging Power, PV Generation and Electricity Price')
        ax2_1.legend(loc='upper left')

        # Set time axis format and range
        ax2_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax2_1.set_xlim(min_date, max_date)

        # PV generation power plot (right axis)
        ax2_1_pv = ax2_1.twinx()
        # Plot curve with points
        ax2_1_pv.plot(mpl_dates, self.pv_generation_record, color='green', marker='o', linestyle='-',
                      label='PV Generation')
        ax2_1_pv.set_ylabel('PV Generation (kW)')
        ax2_1_pv.legend(loc='upper right')

        # Ensure PV generation curve is above 0 scale
        ax2_1_pv.set_ylim(-1, 1)  # Right axis range from -2 to 2, PV generation displayed above 0

        # Electricity price plot (right axis right side)
        ax2_1_price = ax2_1.twinx()
        ax2_1_price.spines['right'].set_position(('outward', 60))  # Move price axis outward
        ax2_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax2_1_price.set_ylabel('Price ($/kWh)')
        ax2_1_price.legend(loc='lower right')

        # ESS SOC change plot
        ax2_2 = plt.subplot(2, 1, 2)
        time_interval = 0.5 / 24  # 30-minute interval
        bar_width = 0.8 * time_interval

        bars = ax2_2.bar(
            x=mpl_dates,
            height=self.ess_state_record,
            width=bar_width,
            color='#23BAC5',
            edgecolor='none',
            align='edge',
            label='ESS SOC'
        )

        ax2_2.set_xlim(min_date, max_date)

        # Set time axis format
        ax2_2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2_2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        ax2_2.set_ylabel('ESS SOC (kWh)', color='#4EC0E9')
        ax2_2.tick_params(axis='y', labelcolor='#4EC0E9')
        ax2_2.set_title('ESS State of Charge ')

        ax2_2.legend([bars], ['ESS SOC'], loc='upper left')

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p4.png')
        plt.close()

        # ===== Third canvas: AC power change plot and temperature change plot =====
        plt.figure(figsize=(20, 10))

        # AC power and electricity price plot
        ax3_1 = plt.subplot(2, 1, 1)
        # Use step function to plot AC power changes
        ax3_1.step(mpl_dates, self.air_conditioner_power_record, color='#B2DBB9', label='AC Power', where='post')
        ax3_1.set_ylabel('AC Power (kW)')
        ax3_1.set_title('AC Power and Electricity Price')
        ax3_1.legend(loc='upper left')

        ax3_1_price = ax3_1.twinx()
        ax3_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax3_1_price.set_ylabel('Price ($/kWh)')
        ax3_1_price.legend(loc='upper right')

        # Set time axis format
        ax3_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Temperature change plot
        ax3_2 = plt.subplot(2, 1, 2)
        ax3_2.plot(mpl_dates, self.records['indoor_temp'], color='limegreen', label='Indoor Temperature')
        ax3_2.plot(mpl_dates, self.records['outdoor_temp'], color='deepskyblue', label='Outdoor Temperature')

        # Add horizontal lines and filled areas for comfortable temperature range
        ax3_2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax3_2.axhline(24, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax3_2.fill_between(mpl_dates, 20, 24, color='lightblue', alpha=0.3)

        ax3_2.set_ylabel('Temperature (°C)')
        ax3_2.set_title('Indoor and Outdoor Temperature Trends')
        ax3_2.legend(loc='upper left')

        # Set time axis format
        ax3_2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3_2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p5.png')
        plt.close()

        # ===== Fourth canvas: Second AC power change plot and temperature change plot =====
        plt.figure(figsize=(20, 10))

        # AC power and electricity price plot
        ax4_1 = plt.subplot(2, 1, 1)
        # Use step function to plot AC power changes
        ax4_1.step(mpl_dates, self.air_conditioner_power_record2, color='#B2DBB9', label='AC Power', where='post')
        ax4_1.set_ylabel('AC Power (kW)')
        ax4_1.set_title('AC Power and Electricity Price')
        ax4_1.legend(loc='upper left')

        ax4_1_price = ax4_1.twinx()
        ax4_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax4_1_price.set_ylabel('Price ($/kWh)')
        ax4_1_price.legend(loc='upper right')

        # Set time axis format
        ax4_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax4_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Temperature change plot
        ax4_2 = plt.subplot(2, 1, 2)
        ax4_2.plot(mpl_dates, self.records['indoor_temp2'], color='limegreen', label='Indoor Temperature')
        ax4_2.plot(mpl_dates, self.records['outdoor_temp'], color='deepskyblue', label='Outdoor Temperature')

        # Add horizontal lines and filled areas for comfortable temperature range
        ax4_2.axhline(16, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax4_2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax4_2.fill_between(mpl_dates, 16, 20, color='lightblue', alpha=0.3)

        ax4_2.set_ylabel('Temperature (°C)')
        ax4_2.set_title('Indoor and Outdoor Temperature Trends')
        ax4_2.legend(loc='upper left')

        # Set time axis format
        ax4_2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax4_2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p6.png')
        plt.close()

        # ===== Fifth canvas: Washing machine state plot =====
        plt.figure(figsize=(20, 5))

        ax5 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])  # Convert timestamps to matplotlib format

        # Plot washing machine state, align bar chart left to timestamp start point
        time_interval = 0.5 / 24  # 30 minutes converted to days
        bar_width = time_interval  # Set bar width consistent with time interval

        # Adjust bar chart position so left edge aligns with timestamp
        bar_left_edges = mpl_dates

        ax5.bar(bar_left_edges, self.wash_machine_record, width=bar_width, color='#F0A19A',
                label='Washing Machine State', align='edge')
        ax5.set_ylabel('Washing Machine State')
        ax5.set_title('Washing Machine State and Electricity Price')
        # 固定 0~1：避免「全天未开洗」时全 0 柱子导致 y 轴被自动缩成以 0 为中心的小区间（柱高为 0 看起来像没画）。
        ax5.set_ylim(0.0, 1.15)
        _wm_max = max(self.wash_machine_record) if self.wash_machine_record else 0
        if _wm_max < 0.5:
            ax5.text(
                0.02,
                0.92,
                'No wash run on this day (state all 0)',
                transform=ax5.transAxes,
                fontsize=11,
                color='gray',
                ha='left',
                va='top',
            )
        ax5.legend(loc='upper left')

        ax5_price = ax5.twinx()
        # Use step function to plot step-like electricity price curve
        ax5_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax5_price.set_ylabel('Price ($/kWh)')
        ax5_price.legend(loc='upper right')

        # Set time axis format
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax5.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Get recorded date range
        start_date = self.records['timestamps'][0]
        end_date = self.records['timestamps'][-1]

        # Iterate through each day
        current_date = start_date
        while current_date <= end_date:
            # Calculate preferred time period for the day
            preferred_start = current_date.replace(hour=self.wash_machine_preferred_time[0], minute=0)
            preferred_end = current_date.replace(hour=self.wash_machine_preferred_time[1], minute=0)

            # Add vertical shaded regions and dashed lines
            ax5.axvspan(preferred_start, preferred_end, facecolor='#5A9BD5', alpha=0.1)  # Light blue fill
            ax5.axvline(preferred_start, color='#5A9BD5', linestyle='--', linewidth=1)  # Dashed line
            ax5.axvline(preferred_end, color='#5A9BD5', linestyle='--', linewidth=1)

            # Move to next day
            current_date += timedelta(days=1)

        # Set time axis range to recorded timestamp range
        ax5.set_xlim(mpl_dates[0], mpl_dates[-1])

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p7.png')
        plt.close()


        # ===== Sixth canvas: Water heater status =====
        plt.figure(figsize=(20, 10))
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # Create a 2 row 1 column grid layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # Dual axis display power and water usage (top subplot)
        ax6_2 = plt.subplot(gs[0])
        ax6_2.bar(mpl_dates, self.ewh_power_record,
                  width=0.015, color='#1F77B4', label='Power')
        ax6_2.set_ylabel('Power (kW)', color='#1F77B4')
        ax6_2.tick_params(axis='y', labelcolor='#1F77B4')

        ax6_2_flow = ax6_2.twinx()
        ax6_2_flow.plot(mpl_dates, self.user_flow_record,
                        color='#2CA02C', marker='o', label='Water Flow')
        ax6_2_flow.set_ylabel('Flow Rate (L/h)', color='#2CA02C')
        ax6_2_flow.tick_params(axis='y', labelcolor='#2CA02C')

        # Water temperature curve (bottom subplot)
        ax6_1 = plt.subplot(gs[1])
        ax6_1.plot(mpl_dates, self.ewh_temp_record,
                   color='#FF7F0E', label='Water Temperature')
        ax6_1.axhline(40, color='grey', linestyle='--', label='Target Temp')
        ax6_1.set_ylabel('Temperature (°C)')
        ax6_1.set_title('Water Heater Status')

        # Add two comfortable temperature range filled areas
        # Peak period 53-57°C (6-9 AM and 6-10 PM)
        peak_low = 50
        peak_high = 60
        non_peak_low = 40
        non_peak_high = 50

        # Create a unified temperature range for filling
        all_low = [peak_low if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_low for ts in
                   self.records['timestamps']]
        all_high = [peak_high if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_high for ts in
                    self.records['timestamps']]

        # Plot unified filled area
        ax6_1.fill_between(mpl_dates, all_low, all_high,
                           color='lightgreen', alpha=0.3, label='Comfort Zone')

        # Plot special markers for peak periods
        peak_mask = [(6 <= ts.hour <= 9) or (18 <= ts.hour <= 22) for ts in self.records['timestamps']]
        ax6_1.fill_between(mpl_dates, peak_low, peak_high,
                           where=peak_mask, color='lightcoral', alpha=0.3, label='Peak Comfort Zone')

        # Unify time axis format
        for ax in [ax6_1, ax6_2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Add legend
        ax6_1.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p8.png')
        plt.close()

        # ===== Seventh canvas: Household total load change plot =====
        plt.figure(figsize=(20, 5))
        ax7 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])  # Convert timestamps to matplotlib format

        # Plot household total load curve
        ax7.plot(mpl_dates, self.records['total_load'], color='purple', label='Total Load')
        ax7.set_ylabel('Load (kW)')
        ax7.set_title('Household Total Load Over Time')
        ax7.legend(loc='upper left')

        # Set time axis format
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax7.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Get current plot y-axis range
        ymin, ymax = ax7.get_ylim()

        # Set grid
        ax7.grid(alpha=0.3)

        # Fill red area (upper part)
        ax7.fill_between(mpl_dates, self.records['total_load'], 0, where=(np.array(self.records['total_load']) > 0),
                         color='red', alpha=0.3)

        # Fill green area (lower part)
        ax7.fill_between(mpl_dates, self.records['total_load'], 0, where=(np.array(self.records['total_load']) < 0),
                         color='green', alpha=0.3)

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p9.png')
        plt.close()

        # ===== Eighth canvas: Cost over time plot =====
        plt.figure(figsize=(20, 5))

        ax8 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # Plot cost curve
        ax8.plot(mpl_dates, self.records['energy_cost'], color='purple', label='Energy Cost')
        ax8.set_ylabel('Cost ($)')
        ax8.set_title('Energy Cost Over Time')
        ax8.legend(loc='upper left')

        # Set time axis format
        ax8.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax8.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        plt.savefig('figures/environment_plots/p10.png')
        plt.close()

        # New: save cost data to file
        # self.save_cost_data()

    def plot_reward_components(self):
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
        plt.figure(figsize=(20, 8))
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # Create color mapping
        colors = {
            'energy_cost': '#FF6B6B',
            'violation_penalty': '#4ECDC4',
            'aging_penalty': '#6C5CE7',
            'ess_reward': '#D95319',
            'ev_reward': '#96CEB4',
            'user_penalty': '#FFEEAD',
            'temp_reward': '#D4A5A5',
            'carbon_penalty': '#95A5A6',
            # 'sell': '#925EBO'
        }

        # Plot stacked area chart
        components = ['energy_cost', 'violation_penalty', 'aging_penalty', 'ess_reward',
                      'ev_reward', 'user_penalty', 'temp_reward', 'carbon_penalty']
        comp_cn = {
            'energy_cost': '能源成本',
            'violation_penalty': '约束违反惩罚',
            'aging_penalty': '电池老化惩罚',
            'ess_reward': '储能奖励',
            'ev_reward': '电动汽车奖励',
            'user_penalty': '用户满意度惩罚',
            'temp_reward': '温度奖励',
            'carbon_penalty': '碳排放惩罚',
        }

        # Cumulative values for stacking
        cumulative = np.zeros(len(mpl_dates))

        for comp in components:
            values = np.array(self.reward_components[comp])
            plt.fill_between(mpl_dates, cumulative, cumulative + values,
                             label=comp_cn.get(comp, comp),
                             color=colors[comp], alpha=0.8)
            cumulative += values

        # Plot total reward line
        plt.plot(mpl_dates, self.reward_components['total'],
                 color='#2C3E50', linewidth=2, label='总奖励')

        # Format settings
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.xticks(rotation=45)

        plt.ylabel('奖励值')
        plt.title('奖励分量分解')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig('figures/environment_plots/p2.png')
        plt.close()
        # plt.show()

    def save_simulation_data(self, filename=None):
        """Save simulation data to CSV file for subsequent plotting and analysis"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.csv"
        
        # Ensure data directory exists
        os.makedirs('simulation_data', exist_ok=True)
        filepath = os.path.join('simulation_data', filename)
        
        # Prepare data dictionary
        data_dict = {
            'timestamp': self.records['timestamps'],
            'ev_soc': self.ev_battery_record,
            'ess_soc': self.ess_state_record,
            'home_load': self.home_load_record,
            'pv_generation': self.pv_generation_record,
            'electricity_price': self.electricity_price_record,
            'ev_at_home': self.ev_at_home_record,
            'wash_machine_state': self.wash_machine_record,
            'air_conditioner_power': self.air_conditioner_power_record,
            'ess_actions': self.ess_actions,
            'wash_machine_actions': self.wash_machine_actions,
            'air_conditioner_actions': self.air_conditioner_actions,
            'ess_charge_pv': self.ess_charge_pv,
            'ess_charge_grid': self.ess_charge_grid,
            'ess_discharge_ev': self.ess_discharge_ev,
            'ess_discharge_house': self.ess_discharge_house,
            'indoor_temp': self.records['indoor_temp'],
            'indoor_temp2': self.records['indoor_temp2'],
            'outdoor_temp': self.records['outdoor_temp'],
            'total_load': self.records['total_load'],
            'energy_cost': self.records['energy_cost'],
            'user_dissatisfaction': self.records['user_dissatisfaction'],
            'ewh_temp': self.ewh_temp_record,
            'ewh_power': self.ewh_power_record,
            'user_flow': self.user_flow_record,
            'air_conditioner_power2': self.air_conditioner_power_record2,
            'daily_costs': self.records['daily_costs']
        }
        
        # Handle length mismatch issues.
        # Some record buffers can be empty (len==0) depending on episode/device usage,
        # so we must not access value[-1].
        list_values = [v for v in data_dict.values() if isinstance(v, list)]
        list_lengths = [len(v) for v in list_values]
        max_length = max(list_lengths) if list_lengths else 0
        
        # Ensure all lists have consistent length
        for key, value in data_dict.items():
            if isinstance(value, list):
                if len(value) < max_length:
                    missing = max_length - len(value)
                    if len(value) == 0:
                        # If the buffer is empty, fill with None.
                        data_dict[key] = [None] * missing
                    else:
                        # Fill with last value (carry-forward).
                        data_dict[key] = value + [value[-1]] * missing
                elif len(value) > max_length:
                    # Truncate to maximum length
                    data_dict[key] = value[:max_length]
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Save to CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Simulation data saved to: {filepath}")
        
        # Also save reward component data
        reward_filename = filename.replace('.csv', '_rewards.csv')
        reward_filepath = os.path.join('simulation_data', reward_filename)
        
        reward_data = {}
        for key, values in self.reward_components.items():
            if isinstance(values, list):
                if len(values) < max_length:
                    missing = max_length - len(values)
                    if len(values) == 0:
                        reward_data[key] = [None] * missing
                    else:
                        reward_data[key] = values + [values[-1]] * missing
                else:
                    reward_data[key] = values[:max_length]
        
        reward_df = pd.DataFrame(reward_data)
        reward_df.to_csv(reward_filepath, index=False, encoding='utf-8-sig')
        print(f"Reward component data saved to: {reward_filepath}")
        
        return filepath


def make_storenet_train_env(
    seed: int = 42,
    train_days: int = 200,
    val_days: int = 30,
    test_days: int = 100,
    **env_kwargs,
) -> HomeEnergyManagementEnv:
    """
    默认训练用环境：StoreNet Ireland 2020，划分方式与 PPO_3rd / DroQ 主脚本一致。
    需已生成 `data/storenet_ireland_2020/daily_pivot_*_H*.csv`。
    """
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    data_append = os.path.join(root, "data")
    if data_append not in sys.path:
        sys.path.append(data_append)
    from storenet_splits import StoreNetSplitConfig, make_storenet_splits_2020, flatten_episode_schedule

    base_dir = os.path.join(root, "data", "storenet_ireland_2020")
    train_houses = [f"H{i}" for i in range(1, 16)]
    val_houses = [f"H{i}" for i in range(1, 16)]
    test_houses = [f"H{i}" for i in range(16, 21)]
    split_cfg = StoreNetSplitConfig(seed=seed, train_days=train_days, val_days=val_days, test_days=test_days)
    splits = make_storenet_splits_2020(
        base_dir=base_dir,
        houses_train=train_houses,
        houses_val=val_houses,
        houses_test=test_houses,
        config=split_cfg,
    )
    schedule_train = flatten_episode_schedule(splits, house_ids=train_houses, split_name="train", seed=seed)
    return HomeEnergyManagementEnv(
        episode_length_steps=48,
        episode_schedule=schedule_train,
        storenet_base_dir=base_dir,
        price_profile="lee2020",
        steps_per_day=48,
        **env_kwargs,
    )


import os
import pandas as pd
import numpy as np
from parameters import index_columns
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from scipy.signal import argrelextrema
from statsmodels.tsa.stl._stl import STL


class TimeSeriesAnalyzer:
    """
    A class for analyzing time series characteristics.
    """

    def __init__(self):
        self.index_columns = index_columns

    def feature_extract(self, data: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """
        Extract features from the time series data.

        For each column, this method compute:
            - Stationarity using the ADF test.
            - Candidate periods via FFT analysis.
            - For each candidate period, seasonal and trend strengths via STL decomposition.
        """
        ts_characteristics_df = pd.DataFrame(columns=self.index_columns)
        n_samples = data.shape[0]
        # copy_df = df.copy()
        # series_length = [df.shape[0]]
        #
        for col in data.columns:
            series = pd.to_numeric(data[col], errors='coerce')

            # 1. Stationarity: Compute the ADF test results
            adf_p_value, is_stationary = self.compute_stationarity(series)
            print("adf_p_value: ", adf_p_value)
            # 2. Trend and Seasonality: Use FFT to get candidate periods
            candidate_periods, amplitudes = self.compute_fft_transform(series, min_amplitude=0)
            filtered_periods = self.filter_candidate_periods(candidate_periods, amplitudes)

            # 3. For each candidate period, compute seasonal and trend strengths
            seasonal_results = []
            for period in filtered_periods:
                strengths = self.compute_strengths_for_period(series, period)
                if strengths is not None:
                    seasonal_results.append(strengths)

            # Ensure we have at least three seasonal/trend results; append defaults if needed.
            while len(seasonal_results) < 3:
                seasonal_results.append({
                    "period": 0,
                    "seasonal_strength": -1,
                    "trend_strength": -1,
                })

            # Sort results by seasonal strength descending
            seasonal_results.sort(key=lambda r: r["seasonal_strength"], reverse=True)
            top_result = seasonal_results[0]
            seasonal_flag = top_result["seasonal_strength"] >= 0.9
            trend_flag = top_result["trend_strength"] >= 0.85

            # Flatten the top 3 seasonal/trend results into a feature list
            seasonal_features = []
            for result in seasonal_results[:3]:
                seasonal_features.extend([
                    result["period"],
                    result["seasonal_strength"],
                    result["trend_strength"],
                ])

            # Assemble final feature vector for this column:
            # [file_name, series_length, seasonal features, seasonal_flag, trend_flag, adf_p_value, is_stationary]
            feature_vector = (
                    [file_name, n_samples] +
                    seasonal_features +
                    [seasonal_flag, trend_flag, adf_p_value, is_stationary]
            )

            print("feature_vector: ", feature_vector)
            ts_characteristics_df.loc[len(ts_characteristics_df)] = feature_vector

        return ts_characteristics_df

    def compute_strengths_for_period(self, series: pd.Series, period: int) -> dict or None:
        """
        Compute the seasonal and trend strengths for a given candidate period using STL decomposition.
        """
        n_samples = len(series)
        # Define a threshold to decide if STL decomposition is applicable
        threshold = max(n_samples // 3, 12)
        if period >= threshold:
            return None

        try:
            trend, seasonal, resid = self.stl_decompose(series, period)
            seasonal_strength = self.compute_seasonal_strength(series, trend, seasonal, resid)
            trend_strength = self.compute_trend_strength(series, trend, seasonal, resid)
            return {"period": period,
                    "seasonal_strength": seasonal_strength,
                    "trend_strength": trend_strength}
        except Exception as e:
            print(f"STL decomposition failed for period {period}: {e}")
            return None

    @staticmethod
    def stl_decompose(series: pd.Series, period: int) -> tuple:
        """
        Perform STL decomposition on the series using a specified period.
        """
        stl_fit = STL(series, period=period).fit()
        return stl_fit.trend, stl_fit.seasonal, stl_fit.resid

    @staticmethod
    def compute_trend_strength(
            series: pd.Series, trend: pd.Series, seasonal: pd.Series, resid: pd.Series
    ) -> float:
        """
        Compute trend strength using the variance of the de-seasonalized series.

        Trend Strength = max(0, 1 - (variance of residuals / variance of (original - seasonal)))
        """
        de_seasonalized = series - seasonal
        var_den = de_seasonalized.var()
        if var_den == 0:
            return 0.0
        return max(0, 1 - resid.var() / var_den)

    @staticmethod
    def compute_seasonal_strength(
            series: pd.Series, trend: pd.Series, seasonal: pd.Series, resid: pd.Series
    ) -> float:
        """
        Compute seasonal strength using the variance of the de-trended series.

        Seasonal Strength = max(0, 1 - (variance of residuals / variance of (original - trend)))
        """
        de_trended = series - trend
        var_den = de_trended.var()
        if var_den == 0:
            return 0.0
        return max(0, 1 - resid.var() / var_den)

    def filter_candidate_periods(self, candidate_periods: np.ndarray, amplitudes: np.ndarray) -> list:
        """
        Filter candidate periods by adjusting them to common cycle lengths and removing duplicates.
        """
        sorted_indices = np.argsort(amplitudes)[::-1]
        candidate_periods_sorted = [round(candidate_periods[idx]) for idx in sorted_indices]
        filtered_periods = []
        for period in candidate_periods_sorted:
            adjusted = self.adjust_period(period)
            if adjusted not in filtered_periods and adjusted >= 4:
                filtered_periods.append(adjusted)
        return filtered_periods

    @staticmethod
    def compute_stationarity(series: pd.Series, significant_level: float = 0.05) -> tuple:
        """
        # ============= Stationary testing =================
        # ADF Test - ADF (Augmented Dickey-Fuller) test is a statistical test used to determine whether a time series is stationary or non-stationary.
        # Null Hypothesis (H₀): The time series has a unit root (i.e., the time series is non-stationary).
        # Alternative Hypothesis (H₁): The time series does not have a unit root (i.e., the time series is stationary).
        # p-value < 0.05: reject the null hypothesis, suggesting that the series does not have a unit root and is stationary.
        # p-value >= 0.05): we fail to reject the null hypothesis, indicating that the series has a unit root and is likely non-stationary.
        """
        try:
            adf_result = adfuller(series.dropna().values, autolag="AIC")
            p_value = adf_result[1]
            is_stationary = p_value < significant_level
        except Exception as e:
            print(f"ADF test failed for series: {e}")
            p_value = None
            is_stationary = None
        return p_value, is_stationary

    @staticmethod
    def compute_fft_transform(series: pd.Series, min_amplitude: float = 0.2) -> tuple:
        """
        Compute the Fourier Transform of a time series and extract candidate periods.
        """
        n = len(series)
        print(series, n)
        fft_vals = np.fft.fft(series.dropna().values)
        amplitudes = np.abs(fft_vals) / n

        # Only consider positive frequencies (first half) and scale accordingly
        amplitudes = amplitudes[: n // 2] * 2

        # Identify local maxima in the amplitude spectrum
        local_max_indices = argrelextrema(amplitudes, np.greater)[0]
        local_max_amplitudes = amplitudes[local_max_indices]

        # Filter by minimum amplitude
        valid_mask = local_max_amplitudes >= min_amplitude
        valid_indices = local_max_indices[valid_mask]
        valid_amplitudes = local_max_amplitudes[valid_mask]

        candidate_periods = n / valid_indices
        return candidate_periods, valid_amplitudes

    @staticmethod
    def adjust_period(period_value: int) -> int:
        if abs(period_value - 4) <= 1:
            period_value = 4
        if abs(period_value - 7) <= 1:
            period_value = 7
        if abs(period_value - 12) <= 2:
            period_value = 12
        if abs(period_value - 24) <= 3:
            period_value = 24
        if abs(period_value - 48) <= 1 or (4 >= (48 - period_value) >= 0):
            period_value = 48
        if abs(period_value - 52) <= 2:
            period_value = 52
        if abs(period_value - 96) <= 10:
            period_value = 96
        if abs(period_value - 144) <= 10:
            period_value = 144
        if abs(period_value - 168) <= 10:
            period_value = 168
        if abs(period_value - 336) <= 50:
            period_value = 336
        if abs(period_value - 672) <= 20:
            period_value = 672
        if abs(period_value - 720) <= 20:
            period_value = 720
        if abs(period_value - 1008) <= 100:
            period_value = 1008
        if abs(period_value - 1440) <= 200:
            period_value = 1440
        if abs(period_value - 8766) <= 500:
            period_value = 8766
        if abs(period_value - 10080) <= 500:
            period_value = 10080
        if abs(period_value - 21600) <= 2000:
            period_value = 21600
        if abs(period_value - 43200) <= 2000:
            period_value = 43200
        return period_value


# =================== Test Case ================
if __name__ == "__main__":
    file_path = '/home/ktp_user/PhD-Code/TimeSeries-AdvancedMethods-Hub/dataset/ETT-small/ETTh2.csv'
    df = pd.read_csv(file_path)
    file_name = "ETTh2"

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    ts_analyzer = TimeSeriesAnalyzer()
    ts_characteristics_df = ts_analyzer.feature_extract(df, file_name)
    print(ts_characteristics_df)



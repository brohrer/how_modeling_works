#! /usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402


SEED = 67
START_YEAR = 1900
END_YEAR = 2018
DAYS_PER_YEAR = 365.25
BASELINE_TEMP = 11

COMPONENTS_FIG = "components.png"
COMPONENTS_ZOOM_FIG = "components_zoom.png"
TEMPS_FIG = "temps.png"
TEMPS_ZOOM_FIG = "temps_zoom.png"
ANNUAL_TEMPS_FIG = "annual_temps.png"


def simulate_annual() -> np.array:
    """
    Modify the simulated daily temperatures to get annual medians.
    """
    temps = simulate()
    n_years = END_YEAR - START_YEAR + 1
    annual_temps = np.zeros(n_years)
    for i_year in range(n_years):
        i_start_day = int(np.floor(i_year * DAYS_PER_YEAR))
        i_end_day = int(np.floor((i_year + 1) * DAYS_PER_YEAR))
        annual_temps[i_year] = np.nanmedian(temps[i_start_day: i_end_day])
    return annual_temps


def simulate() -> np.ndarray:
    """
    Create some imaginary longitudinal temperature data.

    Units are degrees Celsius.
    Missing values are represented by nans.
    """
    # Initialize the random number generator so that the data
    # generated is the same each time.
    np.random.seed(SEED)

    n_days = int((END_YEAR - START_YEAR + 1) * DAYS_PER_YEAR)

    warming_trend = create_warming_trend(n_days)
    solar_cycle = create_solar_cycle(n_days)
    annual_trend = create_annual_trend(n_days)
    meanders = create_meanders(n_days)
    jitter = create_jitter(n_days)

    temps = (
        BASELINE_TEMP
        + warming_trend
        + solar_cycle
        + annual_trend
        + meanders
        + jitter
    )

    temps = add_gaps(temps)
    return temps


def create_warming_trend(
        n_days: int,
        pivot_year: int = 90,
        time_constant_year: int = 30,
) -> np.ndarray:
    """
    Simulate a warming trend as an exponential curve.

    pivot_year is the year at which the exponential is at 1.
    At pivot_year + time_constant_year it is at e.
    """
    pivot = pivot_year * DAYS_PER_YEAR
    time_constant = time_constant_year * DAYS_PER_YEAR
    i_days = np.cumsum(np.ones(n_days))
    warming_trend = np.exp((i_days - pivot) / time_constant)
    return warming_trend


def create_solar_cycle(
    n_days: int = None,
    typical_duration: float = 10,
    typical_amplitude: float = 2,
    variation: float = .5,
) -> np.ndarray:
    """
    Approximate the El Nino / La Nina solar radiation cycles.
    """
    # Solar cycle
    solar_cycle = np.zeros(n_days)
    i_next = 0
    done = False
    while not done:
        cycle_length = np.random.normal(loc=typical_duration)
        n_days_cycle = int(cycle_length * DAYS_PER_YEAR)
        i_days_cycle = np.cumsum(np.ones(n_days_cycle))
        cycle_amplitude = np.maximum(
            typical_amplitude / 4,
            np.random.normal(loc=typical_amplitude, scale=variation))
        cycle = (cycle_amplitude / 2) * (
            np.sin(2 * np.pi * (i_days_cycle / n_days_cycle)))

        if i_next + n_days_cycle < n_days:
            solar_cycle[i_next: i_next + n_days_cycle] = cycle
            i_next += n_days_cycle
        else:
            solar_cycle[i_next:] = cycle[:(n_days - i_next)]
            done = True
    return solar_cycle


def create_annual_trend(
    n_days: int,
    offset: int = 30,
    seasonal_amplitude: float = 25,
) -> np.ndarray:
    """
    Generate the annual trend in temperatures.

    The first day is January 1.
    The coldest day is January 1 + offset.
    The hottest day is July 1 + offset.
    There is a difference between summer and winter of seasonal_amplitude.
    """
    i_days = np.cumsum(np.ones(n_days))
    annual_trend = (seasonal_amplitude / 2) * (
        -np.cos(2 * np.pi * (i_days - offset) / DAYS_PER_YEAR))
    return annual_trend


def create_meanders(
    n_days: int,
    typical_meander_amplitude: float = 4,
    typical_meander_interval: int = 4,
    typical_meander_length: int = 6,
) -> np.ndarray:
    """
    Create brief temperature meanders. Warm spells. Cold snaps.
    """
    meanders = np.zeros(n_days)
    i_next = 0
    done = False
    while not done:
        n_days_meander = np.random.normal(loc=typical_meander_length)
        n_days_meander = int(np.maximum(2, n_days_meander))

        i_days_meander = np.cumsum(np.ones(n_days_meander))
        meander_amplitude = np.random.normal(scale=typical_meander_amplitude)
        meander = meander_amplitude * (
            np.sin(np.pi * (i_days_meander / n_days_meander))**2)

        if i_next + n_days_meander < n_days:
            meanders[i_next: i_next + n_days_meander] = meander

            meander_interval = np.random.normal(loc=typical_meander_interval)
            meander_interval = int(np.maximum(1, meander_interval))
            i_next += meander_interval

            if i_next >= n_days:
                done = True
        else:
            meanders[i_next:] += meander[:(n_days - i_next)]
            done = True
    return meanders


def create_jitter(
    n_days: int,
    typical_jitter_amplitude: float = 2,
) -> np.ndarray:
    """
    Random jitter, day-by-day.
    """
    jitter = np.random.normal(scale=typical_jitter_amplitude, size=n_days)
    return jitter


def create_gaps(
    temps: np.ndarray,
    gap_candidate_period: int = 70,
    n_gaps: int = None,
    typical_length: float = None,
    variation: float = None,
) -> np.ndarray:
    """
    Add periods of missing values in the temperature data.

    All gaps will occur within the first
    gap_candidate_years of the data.
    """
    for i_gap in range(n_gaps):
        start = int(np.random.random_sample()
                    * gap_candidate_period
                    * DAYS_PER_YEAR)
        gap_length = int(np.random.normal(
            loc=typical_length,
            scale=variation,
        ))
        gap_length = np.maximum(typical_length // 4, gap_length)
        temps[start: start + gap_length] = np.nan

    return temps


def add_gaps(
    temps: np.ndarray,
    n_long_gaps: int = 4,
    n_med_gaps: int = 12,
    n_short_gaps: int = 50,
    typical_long_gap_length: float = 60,  # days
    typical_med_gap_length: float = 14,  # days
    typical_short_gap_length: float = 2,  # days
    long_gap_variation: float = 20,  # days
    med_gap_variation: float = 4,  # days
    short_gap_variation: float = 1,  # days
) -> np.ndarray:
    """
    Add in segments of missing data of varying durations.
    """
    temps = create_gaps(
        temps,
        n_gaps=n_long_gaps,
        typical_length=typical_long_gap_length,
        variation=long_gap_variation,
    )
    temps = create_gaps(
        temps,
        n_gaps=n_med_gaps,
        typical_length=typical_med_gap_length,
        variation=med_gap_variation,
    )
    temps = create_gaps(
        temps,
        n_gaps=n_short_gaps,
        typical_length=typical_short_gap_length,
        variation=short_gap_variation,
    )
    return temps


def visualize(
    dpi: int = 300,
) -> None:
    """
    Express the data in pictures.
    """
    temps = simulate()

    n_days = int((END_YEAR - START_YEAR + 1) * DAYS_PER_YEAR)
    warming_trend = create_warming_trend(n_days)
    solar_cycle = create_solar_cycle(n_days)
    annual_trend = create_annual_trend(n_days)
    meanders = create_meanders(n_days)
    jitter = create_jitter(n_days)

    # Choose a random year to zoom into.
    i_year = int(np.random.random_sample() * (END_YEAR - START_YEAR))
    year_label = str(START_YEAR + i_year)
    start_day = int(i_year * DAYS_PER_YEAR)
    end_day = int((i_year + 1) * DAYS_PER_YEAR)

    def plot_temp(
        temp_data: np.ndarray,
        ylabel: str = "temps",
        linewidth: float = 0,
        marker: str = ".",
        markersize: float = 1,
        color: str = "black",
        alpha: float = .1,
    ) -> None:
        """
        Show time series temperature data in an easy-to-grasp manner.
        """
        plt.plot(
            temp_data,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            color=color,
            alpha=alpha,
        )
        plt.ylabel(ylabel)
        return

    plt.figure(num=111)
    plot_temp(temps)
    plt.savefig(TEMPS_FIG, dpi=dpi)

    plt.figure(num=11111)
    plot_temp(
        temps[start_day: end_day],
        ylabel="temps " + year_label,
        alpha=1,
    )
    plt.savefig(TEMPS_ZOOM_FIG, dpi=dpi)

    plt.figure(num=222)
    plt.subplot(5, 1, 1)
    plot_temp(warming_trend, ylabel="warming_trend")

    plt.subplot(5, 1, 2)
    plot_temp(solar_cycle, ylabel="solar_cycle")

    plt.subplot(5, 1, 3)
    plot_temp(annual_trend, ylabel="annual_trend")

    plt.subplot(5, 1, 4)
    plot_temp(meanders, ylabel="meanders")

    plt.subplot(5, 1, 5)
    plot_temp(jitter, ylabel="jitter")

    plt.savefig(COMPONENTS_FIG, dpi=dpi)

    plt.figure(num=22222)
    plt.subplot(5, 1, 1)
    plot_temp(
        warming_trend[start_day: end_day],
        ylabel="warming_trend " + year_label,
        alpha=1,
    )

    plt.subplot(5, 1, 2)
    plot_temp(
        solar_cycle[start_day: end_day],
        ylabel="solar_cycle " + year_label,
        alpha=1,
    )

    plt.subplot(5, 1, 3)
    plot_temp(
        annual_trend[start_day: end_day],
        ylabel="annual_trend " + year_label,
        alpha=1,
    )

    plt.subplot(5, 1, 4)
    plot_temp(
        meanders[start_day: end_day],
        ylabel="meanders " + year_label,
        alpha=1,
    )

    plt.subplot(5, 1, 5)
    plot_temp(
        jitter[start_day: end_day],
        ylabel="jitter " + year_label,
        alpha=1,
    )

    plt.savefig(COMPONENTS_ZOOM_FIG, dpi=dpi)


def test() -> None:
    print(" ".join([
        "Inspect",
        COMPONENTS_FIG,
        "and",
        COMPONENTS_ZOOM_FIG,
        "and check for plausibility.",
    ]))
    visualize()
    return


if __name__ == "__main__":
    test()

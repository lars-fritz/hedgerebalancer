import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Set a random seed for reproducibility (for unique runs each time) ---
# In Streamlit, we might want to fix the seed or allow user to change for consistent runs
np.random.seed() # Using a fixed seed for consistent results across runs

st.set_page_config(layout="wide")

st.title("Geometric Brownian Motion Simulation with Automated LP Rebalancing and Hedging")

st.sidebar.header("Simulation Parameters")

# --- 1. Define Simulation Parameters for GBM (User Inputs) ---
p0_simulation_input = st.sidebar.number_input(
    "Initial Price (P0_SIMULATION)",
    min_value=1.0,
    max_value=1000.0,
    value=39.0,
    step=0.1,
    format="%.2f",
    key="p0_sim"
)
DAILY_VOLATILITY_FACTOR = st.sidebar.number_input(
    "Daily Volatility Factor (Sigma Factor)",
    min_value=0.0001,
    max_value=0.1,
    value=0.002,
    step=0.0001,
    format="%.4f",
    key="vol_factor"
)
SIGMA_DAILY = p0_simulation_input * DAILY_VOLATILITY_FACTOR # Daily volatility (sigma)

TIME_STEP_SECONDS = st.sidebar.number_input(
    "Time Step (seconds)",
    min_value=1,
    max_value=3600,
    value=1,
    step=1,
    key="time_step"
)
TOTAL_SIMULATION_SECONDS = st.sidebar.number_input(
    "Total Simulation Duration (seconds)",
    min_value=3600,
    max_value=24 * 3600 * 7, # Max 7 days
    value=24 * 3600,
    step=3600,
    key="total_sim_seconds"
)
# FUNDING_RATE_PER_HOUR = 0.00001 # Funding rate (0.00001 per hour) - Removed

st.sidebar.header("Liquidity Position & Hedge Parameters")

INITIAL_TOTAL_USDC_VALUE = st.sidebar.number_input(
    "Initial Total USDC Value",
    min_value=100.0,
    max_value=100000.0,
    value=1000.0,
    step=100.0,
    format="%.2f",
    key="initial_usdc"
)
HEDGE_RATIO_R = st.sidebar.slider(
    "Hedge Ratio (r)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
    format="%.2f",
    key="hedge_ratio"
)
RELATIVE_RANGE = st.sidebar.number_input(
    "Relative LP Range (e.g., 0.05 for +/- 5%)",
    min_value=0.01,
    max_value=0.5,
    value=0.05,
    step=0.01,
    format="%.2f",
    key="relative_range"
)

HEDGE_ACTIVATION_THRESHOLD_PERCENTAGE = st.sidebar.number_input(
    "Hedge Activation Threshold Percentage (e.g., 0.01 for 1%)",
    min_value=0.0,
    max_value=0.1,
    value=0.0, # Currently set to 0.0 as per last request
    step=0.001,
    format="%.3f",
    key="hedge_activation_threshold"
)

# Derived parameters from inputs
P_UP_THRESHOLD = p0_simulation_input * (1 + HEDGE_ACTIVATION_THRESHOLD_PERCENTAGE)
P_DOWN_THRESHOLD = p0_simulation_input * (1 - HEDGE_ACTIVATION_THRESHOLD_PERCENTAGE)


# --- Helper functions (from previous Python script) ---

def calculate_xA(p, L_val, p_max_val):
    if p <= 0: return 0.0
    if p_max_val <= 0: raise ValueError("p_max must be greater than 0 for x_A calculation.")
    return L_val * (1 / np.sqrt(p) - 1 / np.sqrt(p_max_val))

def calculate_xB(p, L_val, p_min_val):
    if p <= 0: return 0.0
    if p_min_val <= 0: raise ValueError("p_min must be greater than 0 for x_B calculation.")
    return L_val * (np.sqrt(p) - np.sqrt(p_min_val))

def calculate_lp_position(usdc_deposit, p0_price, relative_range_param):
    p_min_lp = p0_price * (1 - relative_range_param)
    p_max_lp = p0_price * (1 + relative_range_param)
    L_derived = 0.0
    denominator_L = (2 * np.sqrt(p0_price) - np.sqrt(p_min_lp) - p0_price / np.sqrt(p_max_lp))
    if denominator_L != 0:
        L_derived = usdc_deposit / denominator_L
    return L_derived, calculate_xA(p0_price, L_derived, p_max_lp), calculate_xB(p0_price, L_derived, p_min_lp), p_min_lp, p_max_lp

def calculate_delta_xA_sell(p, p_ref, L_val_inner):
    if p <= 0 or p_ref <= 0: return 0.0
    return L_val_inner * (1 / np.sqrt(p_ref) - 1 / np.sqrt(p))

def calculate_delta_xB_sell(p, p_ref, L_val_inner):
    if p <= 0 or p_ref <= 0: return 0.0
    return L_val_inner * (np.sqrt(p_ref) - np.sqrt(p))

def calculate_il(L_val, p0_val, p_max_val, p_min_val, p_current):
    def calculate_IL_A(p, p_ref, L_val_inner):
        if p <= p_ref: return 0.0
        return (p - np.sqrt(p_ref * p)) * calculate_delta_xA_sell(p, p_ref, L_val_inner)
    def calculate_IL_B_in_USDC(p, p_ref, L_val_inner):
        if p >= p_ref: return 0.0
        return (1 - np.sqrt(p / p_ref)) * calculate_delta_xB_sell(p, p_ref, L_val_inner)
    
    il_at_current_price = 0.0
    if p_min_val <= p_current <= p_max_val:
        if p_current >= p0_val:
            il_at_current_price = calculate_IL_A(p_current, p0_val, L_val)
        else:
            il_at_current_price = calculate_IL_B_in_USDC(p_current, p0_val, L_val)
    
    max_il_up = calculate_IL_A(p_max_val, p0_val, L_val) if p_max_val > p0_val else 0.0
    max_il_down = calculate_IL_B_in_USDC(p_min_val, p0_val, L_val) if p_min_val < p0_val else 0.0
    
    return il_at_current_price, max_il_up, max_il_down

def calculate_rebalance_triggers(p0_val, p_min_val, p_max_val, p_up_threshold_val, p_down_threshold_val, L_val):
    p_rebalance_up = 0.0
    if p_max_val > p_up_threshold_val and p0_val > 0 and p_max_val > 0 and p_up_threshold_val > 0:
        numerator_term = np.sqrt(p0_val) * np.sqrt(p_max_val) * (p_max_val - p_up_threshold_val)
        denominator_term = (np.sqrt(p0_val) * p_max_val - np.sqrt(p0_val) * np.sqrt(p0_val * p_max_val) +
                            np.sqrt(p_max_val) * np.sqrt(p0_val * p_max_val) - np.sqrt(p_max_val) * p_up_threshold_val)
        if denominator_term != 0:
            p_rebalance_up = (numerator_term / denominator_term)**2
    
    p_rebalance_down = 0.0
    if p_min_val < p_down_threshold_val and p_down_threshold_val != 0 and p0_val != 0 and p_min_val != 0:
        numerator_term_down = (1 / np.sqrt(p0_val)) * (1 / np.sqrt(p_min_val)) * ( (1 / p_min_val) - (1 / p_down_threshold_val) )
        denominator_term_down_part1 = (1 / np.sqrt(p0_val)) * ( (-1 / p_min_val) + (1 / np.sqrt(p0_val * p_min_val)) )
        denominator_term_down_part2 = (1 / np.sqrt(p_min_val)) * ( (-1 / np.sqrt(p0_val * p_min_val)) + (1 / p_down_threshold_val) )
        denominator_term_down = denominator_term_down_part1 + denominator_term_down_part2
        if denominator_term_down != 0:
            p_rebalance_down = (numerator_term_down / denominator_term_down)**(-2)
    return p_rebalance_up, p_rebalance_down

def initialize_total_position(total_usdc_value, ratio_r, p0_simulation_val, relative_range_val, p_up_threshold_val, p_down_threshold_val):
    usdc_for_lp = total_usdc_value * (1 - ratio_r)
    usdc_for_hedge = total_usdc_value * ratio_r
    L_lp, hype_amount_lp, usdc_amount_lp, p_min_lp, p_max_lp = calculate_lp_position(usdc_for_lp, p0_simulation_val, relative_range_val)
    _, max_il_up, max_il_down = calculate_il(L_lp, p0_simulation_val, p_max_lp, p_min_lp, p0_simulation_val)
    
    lever_upward_move = 0.0
    if usdc_for_hedge > 0 and max_il_up > 0:
        lever_upward_move = max(1.0, max_il_up / usdc_for_hedge)
    
    lever_downward_move = 0.0
    if usdc_for_hedge > 0 and max_il_down > 0:
        lever_downward_move = max(1.0, max_il_down / usdc_for_hedge)

    x_A_hedge_target = 0.0
    if p_max_lp > p_up_threshold_val and p_up_threshold_val > 0:
        denominator_xa = (p_max_lp - p_up_threshold_val)
        if denominator_xa != 0:
            x_A_hedge_target = ((p_max_lp - np.sqrt(p0_simulation_val * p_max_lp)) / denominator_xa) * \
                               calculate_delta_xA_sell(p_max_lp, p0_simulation_val, L_lp)
    
    x_B_hedge_target = 0.0
    if p_min_lp < p_down_threshold_val and p_down_threshold_val > 0:
        denominator_xb = (1/p_min_lp - 1/p_down_threshold_val)
        if denominator_xb != 0:
            x_B_hedge_target = ((1/p_min_lp - 1/np.sqrt(p0_simulation_val * p_min_lp)) / denominator_xb) * \
                               calculate_delta_xB_sell(p_min_lp, p0_simulation_val, L_lp)

    return L_lp, hype_amount_lp, usdc_amount_lp, p_min_lp, p_max_lp, usdc_for_hedge, \
           max_il_up, max_il_down, lever_upward_move, lever_downward_move, \
           x_A_hedge_target, x_B_hedge_target

# --- Main Simulation Function ---
def run_simulation(p0_sim, daily_vol_factor, time_step_s, total_sim_s,
                   initial_total_usdc, hedge_ratio, relative_range_param,
                   hedge_activation_threshold_perc):

    sigma_daily = p0_sim * daily_vol_factor
    num_steps = int(total_sim_s / time_step_s)
    dt_days = time_step_s / (24 * 3600)

    price_path_simulated = np.zeros(num_steps + 1)
    price_path_simulated[0] = p0_sim
    time_stamps_simulated = pd.to_datetime('2023-01-01 00:00:00') + pd.to_timedelta(np.arange(num_steps + 1), unit='s')

    for t in range(num_steps):
        Z = np.random.normal(0, 1)
        price_path_simulated[t+1] = price_path_simulated[t] * np.exp(-0.5 * sigma_daily**2 * dt_days + sigma_daily * np.sqrt(dt_days) * Z)
        if price_path_simulated[t+1] <= 0:
            price_path_simulated[t+1] = 1e-6

    # Rebalancing Logic Variables
    current_total_usdc_value = initial_total_usdc
    current_p0_lp = p0_sim
    current_p_up_threshold = current_p0_lp * (1 + hedge_activation_threshold_perc)
    current_p_down_threshold = current_p0_lp * (1 - hedge_activation_threshold_perc)

    all_impermanent_losses_segment = []
    all_total_hedge_pnl_segment = []
    all_net_pnl_segment = []
    cumulative_total_value_path = []
    all_current_position_unrealized_value = []

    rebalance_times = []
    rebalance_prices = []
    rebalance_new_values = []

    (current_L, current_hype_amount, current_usdc_amount,
     current_p_min_lp, current_p_max_lp, current_usdc_for_hedge,
     current_max_il_up, current_max_il_down, current_lever_up, current_lever_down,
     current_x_A_hedge_target, current_x_B_hedge_target) = \
        initialize_total_position(current_total_usdc_value, hedge_ratio, current_p0_lp, relative_range_param,
                                  current_p_up_threshold, current_p_down_threshold)
    
    current_p_rebalance_up, current_p_rebalance_down = calculate_rebalance_triggers(
        current_p0_lp, current_p_min_lp, current_p_max_lp, current_p_up_threshold, current_p_down_threshold, current_L
    )

    segment_impermanent_loss = 0.0
    segment_hedge_pnl = 0.0

    all_impermanent_losses_segment.append(0.0)
    all_total_hedge_pnl_segment.append(0.0)
    all_net_pnl_segment.append(0.0)
    cumulative_total_value_path.append(initial_total_usdc)
    all_current_position_unrealized_value.append(initial_total_usdc)


    for i in range(num_steps):
        p_previous = price_path_simulated[i]
        p_current = price_path_simulated[i+1]
        current_time = time_stamps_simulated[i+1]

        rebalance_triggered = False
        rebalance_direction = None

        if current_p_rebalance_up > 0 and p_current >= current_p_rebalance_up and p_previous < current_p_rebalance_up:
            rebalance_triggered = True
            rebalance_direction = "Up"
        elif current_p_rebalance_down > 0 and p_current <= current_p_rebalance_down and p_previous > current_p_rebalance_down:
            rebalance_triggered = True
            rebalance_direction = "Down"
        
        il_at_step, _, _ = calculate_il(current_L, current_p0_lp, current_p_max_lp, current_p_min_lp, p_current)
        segment_impermanent_loss = il_at_step

        step_hedge_pnl = 0.0
        if p_current > current_p_up_threshold:
            step_hedge_pnl += current_x_A_hedge_target * (p_current - current_p_up_threshold)
        elif p_current < current_p_down_threshold:
            if current_p_down_threshold != 0:
                hype_shorted_for_down_hedge = current_x_B_hedge_target / current_p_down_threshold
                step_hedge_pnl += hype_shorted_for_down_hedge * (current_p_down_threshold - p_current)

        segment_hedge_pnl = step_hedge_pnl
        net_pnl_current_segment = segment_hedge_pnl - segment_impermanent_loss
        current_position_unrealized_value_at_step = current_total_usdc_value + net_pnl_current_segment

        if rebalance_triggered:
            current_total_usdc_value = current_total_usdc_value + net_pnl_current_segment
            rebalance_times.append(current_time)
            rebalance_prices.append(p_current)
            rebalance_new_values.append(current_total_usdc_value)

            current_p0_lp = p_current
            current_p_up_threshold = current_p0_lp * (1 + hedge_activation_threshold_perc)
            current_p_down_threshold = current_p0_lp * (1 - hedge_activation_threshold_perc)

            (current_L, current_hype_amount, current_usdc_amount,
             current_p_min_lp, current_p_max_lp, current_usdc_for_hedge,
             current_max_il_up, current_max_il_down, current_lever_up, current_lever_down,
             current_x_A_hedge_target, current_x_B_hedge_target) = \
                initialize_total_position(current_total_usdc_value, hedge_ratio, current_p0_lp, relative_range_param,
                                          current_p_up_threshold, current_p_down_threshold)
            
            current_p_rebalance_up, current_p_rebalance_down = calculate_rebalance_triggers(
                current_p0_lp, current_p_min_lp, current_p_max_lp, current_p_up_threshold, current_p_down_threshold, current_L
            )

            segment_impermanent_loss = 0.0
            segment_hedge_pnl = 0.0
            current_position_unrealized_value_at_step = current_total_usdc_value

        all_impermanent_losses_segment.append(segment_impermanent_loss)
        all_total_hedge_pnl_segment.append(segment_hedge_pnl)
        all_net_pnl_segment.append(net_pnl_current_segment)
        cumulative_total_value_path.append(current_total_usdc_value)
        all_current_position_unrealized_value.append(current_position_unrealized_value_at_step)

    return (price_path_simulated, time_stamps_simulated, all_impermanent_losses_segment,
            all_total_hedge_pnl_segment, all_net_pnl_segment, cumulative_total_value_path,
            all_current_position_unrealized_value, rebalance_times, rebalance_prices,
            rebalance_new_values, current_total_usdc_value, initial_total_usdc,
            p0_sim, sigma_daily, current_p_min_lp, current_p_max_lp,
            current_p_up_threshold, current_p_down_threshold)


# Run the simulation when the button is pressed
if st.sidebar.button("Run Simulation"):
    (price_path_simulated, time_stamps_simulated, all_impermanent_losses_segment,
     all_total_hedge_pnl_segment, all_net_pnl_segment, cumulative_total_value_path,
     all_current_position_unrealized_value, rebalance_times, rebalance_prices,
     rebalance_new_values, final_total_usdc_value, initial_total_usdc,
     p0_sim, sigma_daily, p_min_position, p_max_position,
     p_up_threshold, p_down_threshold) = \
        run_simulation(p0_simulation_input, DAILY_VOLATILITY_FACTOR, TIME_STEP_SECONDS, TOTAL_SIMULATION_SECONDS,
                       INITIAL_TOTAL_USDC_VALUE, HEDGE_RATIO_R, RELATIVE_RANGE,
                       HEDGE_ACTIVATION_THRESHOLD_PERCENTAGE)

    st.subheader("Simulation Results")
    st.write(f"**Initial Total USDC Value:** {initial_total_usdc:.4f} USDC")
    st.write(f"**Final Total USDC Value (after all rebalances):** {final_total_usdc_value:.4f} USDC")
    st.write(f"**Total Number of Rebalances:** {len(rebalance_times)}")

    # Plot 1: Simulated Price Path with LP Range and Rebalances
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(time_stamps_simulated, price_path_simulated, label='Simulated Price Path', color='blue')
    ax1.axhline(y=p0_sim, color='green', linestyle='--', label=f'Initial P0: {p0_sim:.2f}')
    ax1.axhline(y=p_min_position, color='red', linestyle=':', label=f'LP Range Min: {p_min_position:.2f}')
    ax1.axhline(y=p_max_position, color='purple', linestyle=':', label=f'LP Range Max: {p_max_position:.2f}')
    ax1.axhline(y=p_up_threshold, color='orange', linestyle='-.', label=f'Hedge Up Threshold: {p_up_threshold:.2f}', alpha=0.7)
    ax1.axhline(y=p_down_threshold, color='cyan', linestyle='-.', label=f'Hedge Down Threshold: {p_down_threshold:.2f}', alpha=0.7)

    for r_time, r_price, r_value in zip(rebalance_times, rebalance_prices, rebalance_new_values):
        ax1.axvline(x=r_time, color='red', linestyle=':', linewidth=1, alpha=0.7)
        ax1.plot(r_time, r_price, 'o', color='red', markersize=6, label=f'Rebalance @ {r_price:.2f}' if r_time == rebalance_times[0] else "")

    ax1.set_title('Simulated Price Path with Rebalance Events')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price (HYPE/USDC)')
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    # Plot 2: Segment-Specific PnL
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(time_stamps_simulated, all_impermanent_losses_segment, label='Impermanent Loss (IL - Current Segment)', color='orange', linestyle='-')
    ax2.plot(time_stamps_simulated, all_total_hedge_pnl_segment, label='Total Hedge PnL (Current Segment)', color='green', linestyle='--')
    ax2.plot(time_stamps_simulated, all_net_pnl_segment, label='Net PnL (Hedge - IL - Current Segment)', color='blue', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_title('Segment-Specific PnL: Impermanent Loss, Hedge PnL, and Net PnL Over Time (in USDC)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value (USDC)')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Plot 3: Total Position Value Over Time (Cumulative & Unrealized)
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    ax3.plot(time_stamps_simulated, cumulative_total_value_path, label='Total Position Value (Cumulative, Realized)', color='purple', linewidth=2)
    ax3.plot(time_stamps_simulated, all_current_position_unrealized_value, label='Current Position Value (Unrealized)', color='darkred', linestyle='-.', linewidth=2, alpha=0.7)
    ax3.axhline(y=initial_total_usdc, color='green', linestyle='--', label=f'Initial Deposit: {initial_total_usdc:.2f}')
    ax3.set_title('Total Position Value Over Time (Cumulative & Unrealized, with Rebalancing)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Total Value (USDC)')
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

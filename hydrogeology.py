"""
Jacob (Cooper-Jacob) Straight-Line Method
------------------------------------------
Pumping test analysis for a confined aquifer.

Given:
    Q  = constant discharge rate
    r  = distance from pumping well to observation well
    t  = time since pumping started (minutes)
    s  = drawdown in the observation well (cm)

Theory (Cooper-Jacob approximation of the Theis equation, valid once
u = r^2*S / (4*T*t) < 0.01, i.e. at "late" time):

    s = (2.303 Q)/(4 pi T) * log10( 2.25 T t / (r^2 S) )

This is a straight line when s is plotted against log10(t):

    s = m * log10(t) + c

where m = slope (drawdown change per log cycle of time).

From the slope and the time-intercept t0 (the time at which the fitted
line crosses s = 0):

    T = 2.303 * Q / (4 * pi * m)          -> Transmissivity
    S = 2.25 * T * t0 / r**2              -> Storativity (Storage coeff.)

Units used below: Q in m3/min, t in minutes, s (drawdown) converted to
metres before use in the T formula, r in metres -> T comes out in m2/min.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. INPUT DATA
# ---------------------------------------------------------------
Q_m3hr = 50.0          # constant discharge, m3/hr
r = 50.0               # distance to observation well, m

t = np.array([4, 6, 8, 10, 12, 16, 18, 24, 30, 36, 45, 51, 85, 100,
              160, 230, 320, 450, 680, 800, 900, 1000])        # minutes
s = np.array([3, 5, 6, 9, 11, 14, 15, 18, 20, 21, 24, 26, 30, 32,
              38, 42, 48, 51, 60, 63, 60, 60])                 # cm

Q = Q_m3hr / 60.0       # convert to m3/min

# ---------------------------------------------------------------
# 2. SELECT THE STRAIGHT-LINE ("LATE-TIME") PORTION
#    Jacob's approximation is only valid once u < 0.01, so the very
#    early points (well-storage/near-well effects) are curved and
#    must be dropped. Here t = 4, 6, 8 min curve away from the trend,
#    and t = 900, 1000 min break the increasing trend (data anomaly/
#    noise) so they are also excluded from the fit.
# ---------------------------------------------------------------
mask = (t >= 10) & (t <= 800)
logt_fit = np.log10(t[mask])
s_fit = s[mask]

# ---------------------------------------------------------------
# 3. LEAST-SQUARES FIT OF THE STRAIGHT LINE  s = m*log10(t) + c
#    (equivalent to fitting the best line through the plotted points
#     by hand on semi-log graph paper)
# ---------------------------------------------------------------
m_slope, c_intercept = np.polyfit(logt_fit, s_fit, 1)   # m in cm/log-cycle

delta_s_cm = m_slope                # drawdown per log cycle (cm)
delta_s_m = delta_s_cm / 100.0      # convert to metres

# time intercept t0 : where the fitted line crosses s = 0
t0 = 10 ** (-c_intercept / m_slope)   # minutes

# ---------------------------------------------------------------
# 4. TRANSMISSIVITY AND STORATIVITY
# ---------------------------------------------------------------
T = 2.303 * Q / (4 * np.pi * delta_s_m)     # m2/min
S = 2.25 * T * t0 / r**2                    # dimensionless

# ---------------------------------------------------------------
# 5. REPORT RESULTS
# ---------------------------------------------------------------
print("=========  Cooper-Jacob Straight-Line Method  =========")
print(f"Discharge, Q                 = {Q_m3hr:.2f} m3/hr  ({Q:.4f} m3/min)")
print(f"Observation well distance, r = {r:.1f} m")
print(f"Slope, Delta_s               = {delta_s_cm:.2f} cm/log-cycle "
      f"({delta_s_m:.4f} m/log-cycle)")
print(f"Time intercept, t0           = {t0:.2f} min")
print("---------------------------------------------------------")
print(f"Transmissivity, T            = {T:.4f} m2/min")
print(f"                              = {T*60:.2f} m2/hr")
print(f"                              = {T*1440:.2f} m2/day")
print(f"Storativity, S                = {S:.5f}  ({S:.3e})")
print("===========================================================")

# ---------------------------------------------------------------
# 6. SEMI-LOG PLOT (s vs log t)  -- the "graph paper" plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xscale('log')

ax.scatter(t, s, color='black', zorder=5, label='Observed drawdown')
ax.scatter(t[mask], s[mask], facecolors='none', edgecolors='red', s=140,
           linewidths=1.5, zorder=6, label='Points used for straight line')

tt = np.logspace(0, np.log10(1200), 200)
ss = m_slope * np.log10(tt) + c_intercept
ax.plot(tt, ss, 'b-', lw=2, label='Cooper-Jacob straight-line fit')

ax.axhline(0, color='gray', lw=0.8)
ax.axvline(t0, color='purple', ls=':', lw=1.5)
ax.plot(t0, 0, 'p', color='purple', markersize=10, zorder=7)
ax.annotate(f'$t_0$ = {t0:.2f} min', xy=(t0, 0), xytext=(t0*2.2, -14),
            arrowprops=dict(arrowstyle='->', color='purple'), color='purple')

ax.set_xlabel('Time, t (minutes) - log scale')
ax.set_ylabel('Drawdown, s (cm)')
ax.set_title('Cooper-Jacob Straight-Line Method\n'
             f'Confined Aquifer (Q = {Q_m3hr:.0f} m3/hr, r = {r:.0f} m)')
ax.grid(True, which='both', ls=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(1, 1200)

plt.tight_layout()
plt.savefig('jacob_semilog_plot.png', dpi=200)
plt.show()

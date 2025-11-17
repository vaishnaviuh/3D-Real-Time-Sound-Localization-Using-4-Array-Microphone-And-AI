from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_2d_azimuth_distance(
    azimuth_deg: float, 
    distance_m: Optional[float], 
    mic_positions: Optional[np.ndarray] = None
) -> None:
    r = distance_m if distance_m is not None else 1.0
    theta = np.deg2rad(azimuth_deg)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fig, ax = plt.subplots(figsize=(7, 7))
    circle = plt.Circle((0, 0), r, color="lightgray", fill=False, linestyle="--")
    ax.add_artist(circle)
    
    # Plot microphone positions (4 mics on XY plane, z=0)
    if mic_positions is not None:
        mic_x = mic_positions[:, 0]
        mic_y = mic_positions[:, 1]
        ax.scatter(mic_x, mic_y, c="green", marker="^", s=200, label="Microphones", zorder=5)
        # Label microphones
        for i, (mx, my) in enumerate(zip(mic_x, mic_y)):
            ax.annotate(f"M{i+1}", (mx, my), xytext=(5, 5), textcoords="offset points", 
                       fontsize=9, color="green", weight="bold")
    
    # Plot detected sound source
    ax.quiver(0, 0, x, y, angles="xy", scale_units="xy", scale=1, color="C0", width=0.005, zorder=3)
    ax.scatter([x], [y], c="red", marker="*", s=300, label="Sound Source", zorder=4)
    
    # Add parameter text box
    param_text = f"Azimuth: {azimuth_deg:.1f}°\nDistance: {r:.2f} m"
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment="top", 
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            family="monospace")
    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"2D Sound Localization Map")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    # Add margins so the map looks larger and less cramped
    margin = max(0.5, 0.5 * r)
    lim = r + margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.tight_layout()
    plt.show()


def plot_3d_direction(
    azimuth_deg: float, 
    elevation_deg: float, 
    distance_m: Optional[float],
    mic_positions: Optional[np.ndarray] = None
) -> None:
    r = distance_m if distance_m is not None else 1.0
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot microphone positions
    if mic_positions is not None:
        mic_x = mic_positions[:, 0]
        mic_y = mic_positions[:, 1]
        mic_z = mic_positions[:, 2]
        ax.scatter(mic_x, mic_y, mic_z, c="green", marker="^", s=200, label="Microphones", zorder=5)
        # Label microphones
        for i, (mx, my, mz) in enumerate(zip(mic_x, mic_y, mic_z)):
            ax.text(mx, my, mz, f" M{i+1}", fontsize=9, color="green", weight="bold")
    
    # Plot detected sound source
    ax.quiver(0, 0, 0, x, y, z, length=1.0, color="C0", normalize=True, arrow_length_ratio=0.2, zorder=3)
    ax.scatter([x], [y], [z], c="red", marker="*", s=300, label="Sound Source", zorder=4)
    
    # Add parameter text box
    param_text = f"Azimuth: {azimuth_deg:.1f}°\nElevation: {elevation_deg:.1f}°\nDistance: {r:.2f} m"
    ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes, 
              fontsize=11, verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
              family="monospace")
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Sound Localization Map")
    lim = max(1.0, r)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_combined_2d_3d(
    azimuth_deg: float,
    elevation_deg: float,
    distance_m: Optional[float],
    mic_positions: Optional[np.ndarray] = None,
) -> None:
    # Prepare derived values once
    r = distance_m if distance_m is not None else 1.0
    theta = np.deg2rad(azimuth_deg)
    x2 = r * np.cos(theta)
    y2 = r * np.sin(theta)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    x3 = r * np.cos(el) * np.cos(az)
    y3 = r * np.cos(el) * np.sin(az)
    z3 = r * np.sin(el)

    fig = plt.figure(figsize=(16, 8))
    # Left: 2D
    ax2d = fig.add_subplot(1, 2, 1)
    circle = plt.Circle((0, 0), r, color="lightgray", fill=False, linestyle="--")
    ax2d.add_artist(circle)
    # Mics in 2D
    max_mic_radius = 0.0
    if mic_positions is not None:
        mic_x = mic_positions[:, 0]
        mic_y = mic_positions[:, 1]
        ax2d.scatter(mic_x, mic_y, c="green", marker="^", s=200, label="Microphones", zorder=5)
        for i, (mx, my) in enumerate(zip(mic_x, mic_y)):
            ax2d.annotate(f"M{i+1}", (mx, my), xytext=(5, 5), textcoords="offset points",
                          fontsize=9, color="green", weight="bold")
        max_mic_radius = float(np.max(np.linalg.norm(np.c_[mic_x, mic_y], axis=1)))
    # Detected in 2D
    ax2d.quiver(0, 0, x2, y2, angles="xy", scale_units="xy", scale=1, color="C0", width=0.005, zorder=3)
    ax2d.scatter([x2], [y2], c="red", marker="*", s=300, label="Sound Source", zorder=4)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("X (m)")
    ax2d.set_ylabel("Y (m)")
    ax2d.set_title("2D Sound Localization Map")
    ax2d.grid(True, linestyle=":", alpha=0.5)
    ax2d.legend(loc="upper right")
    base_extent = max(r, max_mic_radius)
    margin = max(0.5, 0.5 * base_extent)
    lim = base_extent + margin
    ax2d.set_xlim(-lim, lim)
    ax2d.set_ylim(-lim, lim)
    ax2d.text(0.02, 0.98, f"Azimuth: {azimuth_deg:.1f}°\nDistance: {r:.2f} m",
              transform=ax2d.transAxes, fontsize=11, verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
              family="monospace")

    # Right: 3D
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    if mic_positions is not None:
        mic_x = mic_positions[:, 0]
        mic_y = mic_positions[:, 1]
        mic_z = mic_positions[:, 2]
        ax3d.scatter(mic_x, mic_y, mic_z, c="green", marker="^", s=200, label="Microphones", zorder=5)
        for i, (mx, my, mz) in enumerate(zip(mic_x, mic_y, mic_z)):
            ax3d.text(mx, my, mz, f" M{i+1}", fontsize=9, color="green", weight="bold")
    ax3d.quiver(0, 0, 0, x3, y3, z3, length=1.0, color="C0", normalize=True, arrow_length_ratio=0.2, zorder=3)
    ax3d.scatter([x3], [y3], [z3], c="red", marker="*", s=300, label="Sound Source", zorder=4)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Sound Localization Map")
    lim3 = max(1.0, r)
    ax3d.set_xlim(-lim3, lim3)
    ax3d.set_ylim(-lim3, lim3)
    ax3d.set_zlim(-lim3, lim3)
    ax3d.legend(loc="upper left")
    ax3d.text2D(0.02, 0.98, f"Azimuth: {azimuth_deg:.1f}°\nElevation: {elevation_deg:.1f}°\nDistance: {r:.2f} m",
                transform=ax3d.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                family="monospace")

    plt.tight_layout()
    plt.show()


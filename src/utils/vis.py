import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

def animate(
        rollout: torch.Tensor | np.ndarray,
        vectors: torch.Tensor | np.ndarray | None = None,
        vector_positions: torch.Tensor | np.ndarray | None = None,
        ground_truth: torch.Tensor | np.ndarray = None,
        ref_frame: tuple | list | np.ndarray | None = None,
        interval: int = 100,
        start_idx: int = 0,
        n_skip_ahead_timesteps: int = 1,
    ) -> animation.FuncAnimation:
    """
    Animates up to three sub-plots:
      • Particles (scatter)
      • Vector field (quiver) – *optional*
      • Ground-truth particles (scatter)
    """

    base_title_fs = plt.rcParams.get('axes.titlesize', plt.rcParams['font.size'])
    if isinstance(base_title_fs, str):                    # e.g. 'medium', 'large'
        base_title_fs = plt.rcParams['font.size']
    title_fs = 2 * float(base_title_fs)

    def _to_numpy(arr):
        try:
            return arr.cpu().numpy()
        except AttributeError:
            return np.asarray(arr)

    rollout_np      = _to_numpy(rollout)
    ground_truth_np = _to_numpy(ground_truth)

    has_vectors = vectors is not None and vector_positions is not None
    if has_vectors:
        vectors_np          = _to_numpy(vectors)
        vector_positions_np = _to_numpy(vector_positions)
        max_norm   = np.linalg.norm(vectors_np.reshape(-1, 2), axis=1).max()
        vectors_np = vectors_np / max_norm
        max_norm   = float(max_norm)
        assert vectors_np.shape[0] == rollout_np.shape[0], \
            f"Mismatch in T: {vectors_np.shape[0]} vs {rollout_np.shape[0]}"

    assert ground_truth_np.shape[0] == rollout_np.shape[0], \
        f"Mismatch in T: {ground_truth_np.shape[0]} vs {rollout_np.shape[0]}"

    pos_sources = [rollout_np.reshape(-1, 2), ground_truth_np.reshape(-1, 2)]
    if has_vectors:
        pos_sources.append(vector_positions_np)
    pos_all = np.concatenate(pos_sources, axis=0)

    margin = 0.1
    if ref_frame is not None:                 # ★ override limits if user supplied them
        (min_x, max_x), (min_y, max_y) = np.asarray(ref_frame)
        margin = 0.0                      # exact limits requested, no padding
    else:
        pos_sources = [rollout_np.reshape(-1, 2), ground_truth_np.reshape(-1, 2)]
        if has_vectors:
            pos_sources.append(vector_positions_np)
        pos_all = np.concatenate(pos_sources, axis=0)
        min_x, max_x = pos_all[:, 0].min(), pos_all[:, 0].max()
        min_y, max_y = pos_all[:, 1].min(), pos_all[:, 1].max()

    width = max(max_x - min_x, max_y - min_y)

    if has_vectors:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 6))
        ax2 = fig.add_subplot(133)
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No vectors supplied',
                  ha='center', va='center', transform=ax2.transAxes)
    
    # doubled font-size for timestep indication ↓↓↓
    time_text = fig.text(0.5, 0.005, '', ha='center', va='bottom', fontsize=24)

    # ------------------------ subplot 1: particles --------------------------
    scatter_plot = ax1.scatter(rollout_np[0,:,0], rollout_np[0,:,1], s=40)
    ax1.set_title('Rollout',       fontsize=title_fs)    # ★ doubled
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    ax1.grid(False); ax1.set_xlim(min_x-margin, max_x+margin); ax1.set_ylim(min_y-margin, max_y+margin)

    # ------------------------ subplot 2: vectors (optional) ----------------
    if has_vectors:
        quiver_plot = ax2.quiver(vector_positions_np[:,0],
                                 vector_positions_np[:,1],
                                 vectors_np[0,:,0], vectors_np[0,:,1],
                                 angles='xy', scale_units='xy',
                                 scale=1.0/(0.1*width))
        ax2.set_title('Vector field', fontsize=title_fs)
        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        ax2.grid(False); ax2.set_xlim(min_x-margin, max_x+margin); ax2.set_ylim(min_y-margin, max_y+margin)
    else:
        quiver_plot = ax2  # placeholder so we can still return it

    # ------------------------ subplot 3: ground-truth ----------------------
    GT_plot = ax3.scatter(ground_truth_np[0,:,0], ground_truth_np[0,:,1], s=40)
    ax3.set_title('Ground Truth',  fontsize=title_fs)
    ax3.set_xlabel('X'); ax3.set_ylabel('Y')
    ax3.grid(False); ax3.set_xlim(min_x-margin, max_x+margin); ax3.set_ylim(min_y-margin, max_y+margin)

    # -----------------------------------------------------------------------
    def init():
        scatter_plot.set_offsets(rollout_np[0])
        GT_plot.set_offsets(ground_truth_np[0])
        if has_vectors:
            quiver_plot.set_offsets(vector_positions_np)
            quiver_plot.set_UVC(vectors_np[0,:,0], vectors_np[0,:,1])
        time_text.set_text(f't={start_idx}')
        return scatter_plot, quiver_plot, GT_plot

    def update(frame):
        scatter_plot.set_offsets(rollout_np[frame])
        GT_plot.set_offsets(ground_truth_np[frame])
        if has_vectors:
            quiver_plot.set_UVC(vectors_np[frame,:,0], vectors_np[frame,:,1])
        time_text.set_text(f't={n_skip_ahead_timesteps * frame + start_idx}')
        return scatter_plot, quiver_plot, GT_plot

    ani = animation.FuncAnimation(fig, update,
                                  frames=rollout_np.shape[0],
                                  init_func=init,
                                  interval=interval, blit=True)
    return ani

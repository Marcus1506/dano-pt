import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

def animate(
        rollout: torch.Tensor | np.ndarray,
        vectors: torch.Tensor | np.ndarray | None = None,
        vector_positions: torch.Tensor | np.ndarray | None = None,
        ground_truth: torch.Tensor | np.ndarray | None = None,
        ref_frame: tuple | list | np.ndarray | None = None,
        interval: int = 100,
        start_idx: int = 0,
        n_skip_ahead_timesteps: int = 1,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
    ) -> animation.FuncAnimation:
    """
    Animates up to three sub-plots (in columns):
      • Rollout particles (scatter)
      • Vector field (quiver) – optional
      • Ground-truth particles (scatter) – optional
    
      n_skip_ahead_timesteps only affects the displayed time index; frames are not skipped.

    Shapes:
      rollout:         (T, N, 2)
      ground_truth:    (T, N, 2)
      vectors:         (T, N, 2)
      vector_positions:(N, 2) or (T, N, 2)
    """

    # ----------------------------- helpers ---------------------------------
    def _to_numpy(arr):
        if arr is None:
            return None
        try:
            return arr.detach().cpu().numpy()
        except AttributeError:
            return np.asarray(arr)

    # ----------------------------- inputs ----------------------------------
    rollout_np      = _to_numpy(rollout)
    GT_np           = _to_numpy(ground_truth)
    has_gt          = GT_np is not None

    if rollout_np.ndim != 3 or rollout_np.shape[-1] != 2:
        raise ValueError(f"rollout must be (T,N,2); got {rollout_np.shape}")

    T, N, _ = rollout_np.shape

    has_vectors = (vectors is not None) and (vector_positions is not None)
    if has_vectors:
        vectors_np          = _to_numpy(vectors)
        vector_positions_np = _to_numpy(vector_positions)

        if vectors_np.ndim != 3 or vectors_np.shape[-1] != 2:
            raise ValueError(f"vectors must be (T,N,2); got {vectors_np.shape}")
        if vectors_np.shape[0] != T:
            raise ValueError(f"T mismatch between rollout ({T}) and vectors ({vectors_np.shape[0]})")

        # vector positions can be (N,2) or (T,N,2)
        if vector_positions_np.ndim == 2:
            if vector_positions_np.shape != (vectors_np.shape[1], 2):
                raise ValueError(f"static vector_positions must be (N,2); got {vector_positions_np.shape}, expected {(vectors_np.shape[1],2)}")
            vp_timevarying = False
        elif vector_positions_np.ndim == 3:
            if vector_positions_np.shape[:2] != vectors_np.shape[:2] or vector_positions_np.shape[-1] != 2:
                raise ValueError(f"time-varying vector_positions must be (T,N,2); got {vector_positions_np.shape}, vectors {vectors_np.shape}")
            vp_timevarying = True
        else:
            raise ValueError("vector_positions must be (N,2) or (T,N,2)")

        # Safe normalization
        max_norm = np.linalg.norm(vectors_np.reshape(-1, 2), axis=1).max()
        denom = max(max_norm, 1e-12)
        vectors_np = vectors_np / denom
    else:
        vectors_np = None
        vector_positions_np = None
        vp_timevarying = False

    if has_gt:
        if GT_np.ndim != 3 or GT_np.shape != rollout_np.shape:
            raise ValueError(f"ground_truth must be (T,N,2) and match rollout; got {GT_np.shape}")

    # ----------------------- frame subsampling ------------------------------
    frame_indices = range(T)

    # ------------------------- axis limits ---------------------------------
    if ref_frame is not None:
        ref_frame = np.asarray(ref_frame)
        if ref_frame.shape != (2, 2):
            raise ValueError("ref_frame must be ((min_x,max_x),(min_y,max_y))")
        (min_x, max_x), (min_y, max_y) = ref_frame
        margin = 0.0
    else:
        margin = 0.1
        pos_list = [rollout_np.reshape(-1, 2)]
        if has_gt:
            pos_list.append(GT_np.reshape(-1, 2))
        if has_vectors:
            if vp_timevarying:
                pos_list.append(vector_positions_np.reshape(-1, 2))
            else:
                pos_list.append(vector_positions_np)
        pos_all = np.concatenate(pos_list, axis=0)
        min_x, max_x = float(pos_all[:, 0].min()), float(pos_all[:, 0].max())
        min_y, max_y = float(pos_all[:, 1].min()), float(pos_all[:, 1].max())

    width = max(max_x - min_x, max_y - min_y)
    width = width if width > 0 else 1.0

    # --------------------------- figure/axes --------------------------------
    ncols = 1 + (1 if has_vectors else 0) + (1 if has_gt else 0)
    if figsize is None:
        figsize = (6 * ncols, 6)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=dpi)
    try:
        fig.set_constrained_layout(False)  # in case you have it on globally
    except Exception:
        pass
    fig.subplots_adjust(bottom=0.18)       # increase to make room for bottom time label

    if ncols == 1:
        axes = [axes]
    else:
        axes = list(np.atleast_1d(axes))

    # order: Rollout | [Vectors] | [Ground Truth]
    ax_rollout = axes[0]
    ax_vectors = axes[1] if has_vectors else None
    ax_gt      = axes[2] if (has_vectors and has_gt) else (axes[1] if (not has_vectors and has_gt) else None)

    base_title_fs = plt.rcParams.get('axes.titlesize', plt.rcParams['font.size'])
    if isinstance(base_title_fs, str):
        base_title_fs = plt.rcParams['font.size']
    title_fs = 2 * float(base_title_fs)

    # footer axis that spans the whole figure width; place time label centered there
    footer_ax = fig.add_axes([0.0, 0.0, 1.0, 0.15], frameon=False)
    footer_ax.set_axis_off()
    time_text = footer_ax.text(
        0.5, 0.5, '', ha='center', va='center',
        fontsize=18, animated=True, transform=footer_ax.transAxes,
        clip_on=False,
    )
    time_text.set_zorder(10_000)

    # Rollout subplot
    sc_rollout = ax_rollout.scatter(rollout_np[0, :, 0], rollout_np[0, :, 1], s=40)
    ax_rollout.set_title('Rollout', fontsize=title_fs)
    ax_rollout.set_xlabel('X'); ax_rollout.set_ylabel('Y')
    ax_rollout.grid(False)
    ax_rollout.set_xlim(min_x - margin, max_x + margin)
    ax_rollout.set_ylim(min_y - margin, max_y + margin)

    # Vectors subplot
    if has_vectors:
        if vp_timevarying:
            x0 = vector_positions_np[0, :, 0]
            y0 = vector_positions_np[0, :, 1]
        else:
            x0 = vector_positions_np[:, 0]
            y0 = vector_positions_np[:, 1]

        qv = ax_vectors.quiver(
            x0, y0,
            vectors_np[0, :, 0], vectors_np[0, :, 1],
            angles='xy', scale_units='xy', scale=1.0 / (0.1 * width)
        )
        ax_vectors.set_title('Vector field', fontsize=title_fs)
        ax_vectors.set_xlabel('X'); ax_vectors.set_ylabel('Y')
        ax_vectors.grid(False)
        ax_vectors.set_xlim(min_x - margin, max_x + margin)
        ax_vectors.set_ylim(min_y - margin, max_y + margin)
    else:
        qv = None

    # Ground-truth subplot
    if has_gt:
        sc_gt = ax_gt.scatter(GT_np[0, :, 0], GT_np[0, :, 1], s=40)
        ax_gt.set_title('Ground Truth', fontsize=title_fs)
        ax_gt.set_xlabel('X'); ax_gt.set_ylabel('Y')
        ax_gt.grid(False)
        ax_gt.set_xlim(min_x - margin, max_x + margin)
        ax_gt.set_ylim(min_y - margin, max_y + margin)
    else:
        sc_gt = None

    # --------------------------- anim funcs ---------------------------------
    def init():
        sc_rollout.set_offsets(rollout_np[0])
        artists = [sc_rollout]
        if has_vectors:
            if vp_timevarying:
                qv.set_offsets(vector_positions_np[0])
            else:
                qv.set_offsets(np.c_[x0, y0])
            qv.set_UVC(vectors_np[0, :, 0], vectors_np[0, :, 1])
            artists.append(qv)
        if has_gt:
            sc_gt.set_offsets(GT_np[0])
            artists.append(sc_gt)
        time_text.set_text(f't={start_idx}')
        artists.append(time_text)
        return tuple(artists)

    def update(t):
        # t is an absolute frame index from frame_indices
        sc_rollout.set_offsets(rollout_np[t])
        artists = [sc_rollout]
        if has_vectors:
            if vp_timevarying:
                qv.set_offsets(vector_positions_np[t])
            qv.set_UVC(vectors_np[t, :, 0], vectors_np[t, :, 1])
            artists.append(qv)
        if has_gt:
            sc_gt.set_offsets(GT_np[t])
            artists.append(sc_gt)
        time_text.set_text(f't={start_idx + t * n_skip_ahead_timesteps}')
        artists.append(time_text)
        return tuple(artists)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        interval=interval,
        blit=False,
        cache_frame_data=False,
    )
    return ani

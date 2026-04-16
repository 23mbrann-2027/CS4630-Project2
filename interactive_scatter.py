import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

clusters_raw_k2   = np.load("clusters_raw_k2.npy")
clusters_pca_2    = np.load("clusters_pca_2.npy")
clusters_pca_5    = np.load("clusters_pca_5.npy")
clusters_pca_10   = np.load("clusters_pca_10.npy")
X_2d              = np.load("X_2d.npy")
y                 = np.load("y_labels.npy")

PLOT_N = 10000
idx_plot = np.random.choice(len(X_2d), size=PLOT_N, replace=False)

titles      = ["Raw 28-dim (proj to 2D)", "PCA-2", "PCA-5", "PCA-10"]
all_clusters = [clusters_raw_k2, clusters_pca_2, clusters_pca_5, clusters_pca_10]
colors_map  = {0: "steelblue", 1: "tomato"}

fig = make_subplots(rows=1, cols=4, subplot_titles=titles)

for col_idx, (title, lbls) in enumerate(zip(titles, all_clusters), start=1):
    for cls in [0, 1]:
        mask = lbls[idx_plot] == cls
        pts  = X_2d[idx_plot][mask]
        true_labels = y[idx_plot][mask]

        fig.add_trace(
            go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="markers",
                marker=dict(size=4, color=colors_map[cls], opacity=0.5),
                name=f"Cluster {cls}",
                legendgroup=f"Cluster {cls}",
                showlegend=(col_idx == 1),
                customdata=np.stack([true_labels, lbls[idx_plot][mask]], axis=1),
                hovertemplate=(
                    "<b>Cluster:</b> %{customdata[1]}<br>"
                    "<b>True label:</b> %{customdata[0]:.0f} "
                    "(0=background, 1=signal)<br>"
                    "<b>PC1:</b> %{x:.3f}<br>"
                    "<b>PC2:</b> %{y:.3f}<extra></extra>"
                )
            ),
            row=1, col=col_idx
        )

fig.update_layout(
    title_text="K-Means Clustering: Raw vs PCA-Reduced",
    height=500,
    width=1400,
)
fig.update_xaxes(title_text="PC1")
fig.update_yaxes(title_text="PC2")

fig.write_html("cluster_scatter_interactive.html")
fig.show()
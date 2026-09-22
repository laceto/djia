import sqlite3
import sys

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FEATURES = [
    "bpm", "swing_score",
    "dark", "hypnotic", "euphoric", "aggressive", "industrial", "minimal",
]
N_CLUSTERS = 5

conn = sqlite3.connect("db/djia.db")
df = pd.read_sql_query(
    """
    SELECT
        t.id, t.file_name, t.artist, t.title, t.duration,
        f.bpm, f.key, f.camelot_key, f.key_confidence, f.swing_score,
        m.dark, m.hypnotic, m.euphoric, m.aggressive, m.industrial, m.minimal,
        (SELECT COUNT(*) FROM segments s WHERE s.track_id = t.id) AS segment_count
    FROM tracks t
    LEFT JOIN features f ON f.track_id = t.id
    LEFT JOIN mood m ON m.track_id = t.id
    ORDER BY t.id
    """,
    conn,
)
conn.close()

clusterable = df.dropna(subset=FEATURES).copy()
X = StandardScaler().fit_transform(clusterable[FEATURES])
clusterable["cluster"] = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit_predict(X)

print(clusterable[["id", "file_name", "artist", "title"] + FEATURES + ["cluster"]])
print()
print(clusterable.groupby("cluster")[FEATURES].mean().round(3))
print()
print(clusterable["cluster"].value_counts().sort_index())

# --- Profile clusters across ALL collected features, not just the ones used to build them ---
ALL_FEATURES = FEATURES + ["key_confidence", "segment_count", "duration"]
all_clusterable = df.dropna(subset=ALL_FEATURES).copy()
all_clusterable["cluster"] = clusterable["cluster"].reindex(all_clusterable.index)
all_clusterable = all_clusterable.dropna(subset=["cluster"])

cluster_profile = all_clusterable.groupby("cluster")[ALL_FEATURES].mean()
print()
print("Cluster profile across all features:")
print(cluster_profile.round(3))

# z-score each feature so differently-scaled features (e.g. bpm vs swing_score) plot on
# the same axis, making cluster-to-cluster movement per feature visible.
profile_z = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()

plt.figure(figsize=(10, 6))
for cluster_id, row in profile_z.iterrows():
    plt.plot(ALL_FEATURES, row.values, marker="o", label=f"cluster {int(cluster_id)}")
plt.xticks(rotation=45, ha="right")
plt.ylabel("z-score (relative to overall mean)")
plt.title("Cluster movement across all features")
plt.legend()
plt.tight_layout()
plt.savefig("examples/cluster_profiles.png")
print("\nsaved plot to examples/cluster_profiles.png")

# --- Full track listing per cluster ---
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
for cluster_id in sorted(clusterable["cluster"].unique()):
    members = clusterable.loc[clusterable["cluster"] == cluster_id, ["id", "file_name", "artist", "title"]]
    print(f"\n=== Cluster {cluster_id} ({len(members)} tracks) ===")
    print(members.to_string(index=False))

# --- Hierarchical clustering (Ward linkage) on the same standardized FEATURES ---
Z = linkage(X, method="ward")

coph_corr, _ = cophenet(Z, pdist(X))
print(f"\nCophenetic correlation (how well the dendrogram preserves true distances): {coph_corr:.3f}")

# Merge-height jumps in the last few steps: a big jump right before N clusters means
# those N groups are well-separated; small/uniform jumps mean the split is arbitrary.
last_merges = Z[-15:, 2]
print("\nMerge-height jumps (bigger jump = more natural cut at that cluster count):")
for m in range(1, len(last_merges)):
    k = len(last_merges) - m  # clusters remaining right after this merge
    print(f"  {k:>3} clusters : jump {last_merges[m] - last_merges[m - 1]:.2f}")

# Cut into N_CLUSTERS for a direct comparison against the KMeans partition above.
clusterable["hclust"] = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

print("\nKMeans (rows) vs Hierarchical (cols) cluster crosstab:")
print(pd.crosstab(clusterable["cluster"], clusterable["hclust"]))

print("\nHierarchical cluster profile (means):")
print(clusterable.groupby("hclust")[FEATURES].mean().round(3))
print(clusterable["hclust"].value_counts().sort_index())

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90.0)
plt.title("Hierarchical clustering dendrogram (last 30 merges)")
plt.xlabel("merged cluster size (or track index)")
plt.ylabel("Ward linkage distance")
plt.tight_layout()
plt.savefig("examples/dendrogram.png")
print("\nsaved dendrogram to examples/dendrogram.png")

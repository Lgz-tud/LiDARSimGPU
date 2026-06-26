# Benchmark: run slidaRtools pipeline and export intermediate results for comparison
# Must run from slidaRtools renv environment:
#   cd D:/Model/slidaRtools
#   Rscript D:/pyCode/lidar-voxel/benchmark/run_slidartools.R

library(slidaRtools)
library(data.table)

outdir <- "D:/pyCode/lidar-voxel/benchmark/r_output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# --- Load demo data ---
load("D:/Model/slidaRtools/DemoWorkflow/Demo_Data_1ha_Traunstein.rda")

# --- Prepare tree attributes (same as Demo_Workflow) ---
inv.dt[, D := DBH]
inv.dt[, H := (60 * D) / (0.5 + D)]
inv.dt[, CL := 0.4 * H]
inv.dt[, CD := 15 * D^0.8]
inv.dt[, CS := 4]
inv.dt[, LAD := 0.3]

# Export prepared inventory
fwrite(inv.dt, file.path(outdir, "inv_prepared.csv"))
cat("Inventory exported:", nrow(inv.dt), "trees\n")

# --- Voxel forest (simple version, same parameters) ---
vxf.dt <- make_voxelforest(inv.dt, minx = 0, maxx = 100, miny = 0, maxy = 100)
fwrite(vxf.dt, file.path(outdir, "voxelforest.csv"))
cat("Voxelforest exported:", nrow(vxf.dt), "voxels\n")
cat("  Z range:", range(vxf.dt$Z), "\n")
cat("  Unique X-Y columns:", nrow(unique(vxf.dt[, .(X, Y)])), "\n")

# --- Voxel forest (parallel version) ---
vxf2.dt <- make_voxelforest_parallel(inv.dt, minx = 0, maxx = 100, miny = 0, maxy = 100, res = 50)
fwrite(vxf2.dt, file.path(outdir, "voxelforest_parallel.csv"))
cat("Voxelforest (parallel) exported:", nrow(vxf2.dt), "voxels\n")

# --- LiDAR simulation ---
sim.lid.dt <- make_lidarscan(vxf.dt, P0.AGR = 0.2, P0.GR = 0.2, k = 0.2)
fwrite(sim.lid.dt, file.path(outdir, "sim_pointcloud.csv"))
cat("Simulated point cloud exported:", nrow(sim.lid.dt), "points\n")
cat("  Z range:", range(sim.lid.dt$Z), "\n")
cat("  Ground points (Z==0):", sum(sim.lid.dt$Z == 0), "\n")
cat("  Canopy points (Z>0):", sum(sim.lid.dt$Z > 0), "\n")

# --- Summary stats for comparison ---
summary_dt <- data.table(
  metric = c("n_voxels", "n_voxels_ground", "n_voxels_canopy",
             "vxf_z_max", "vxf_z_mean",
             "n_lidar_points", "n_lidar_ground", "n_lidar_canopy",
             "lidar_z_max", "lidar_z_mean"),
  value = c(nrow(vxf.dt), sum(vxf.dt$Z == 0), sum(vxf.dt$Z > 0),
            max(vxf.dt$Z), mean(vxf.dt$Z),
            nrow(sim.lid.dt), sum(sim.lid.dt$Z == 0), sum(sim.lid.dt$Z > 0),
            max(sim.lid.dt$Z), mean(sim.lid.dt$Z))
)
fwrite(summary_dt, file.path(outdir, "summary_stats.csv"))
cat("\nSummary stats:\n")
print(summary_dt)

cat("\nDone. R output saved to:", outdir, "\n")

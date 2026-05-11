
import sys
sys.path.insert(0, "./")
import scripts.PY_files.eval_models_scripts.plot_results as plot_results

plot_animations_1step = False
plot_animations_rollout = True

ds_geo_dir = r".\GNN_training\one_wave\nc_files\wave_28_ts_600_g4_sigmamin_15.nc"
raw_dir = r"./GNN_training\one_wave\startup_tests\results_withenergy\raw_preds"
plot_dir = r"./GNN_training\one_wave\startup_tests\results_withenergy\plots"
anim_dir = r"./GNN_training\one_wave\startup_tests\results_withenergy\animations"

plot_results.plot_results(ds_geo_dir,raw_dir,plot_dir,anim_dir, generations = 4, plot_animations_1step = False, plot_animations_rollout = True)


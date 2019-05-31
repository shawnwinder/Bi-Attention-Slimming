# cars196 
echo "compute cars196 network statistics on resnet50"
python src/param_statistics.py --config config/cars196/params_statistic_baseline.yaml | tee cars196_statistics_baseline.out
python src/param_statistics.py --config config/cars196/slimming_pruned-0.3.yaml | tee cars196_statistics_0.3.out
python src/param_statistics.py --config config/cars196/slimming_pruned-0.4.yaml | tee cars196_statistics_0.4.out
python src/param_statistics.py --config config/cars196/slimming_pruned-0.5.yaml | tee cars196_statistics_0.5.out
python src/param_statistics.py --config config/cars196/slimming_pruned-0.6.yaml | tee cars196_statistics_0.6.out
python src/param_statistics.py --config config/cars196/slimming_pruned-0.7.yaml | tee cars196_statistics_0.7.out

# birds200
echo "compute birds200 network statistics on resnet50"
python src/param_statistics.py --config config/birds200/params_statistic_baseline.yaml | tee birds200_statistics_baseline.out
python src/param_statistics.py --config config/birds200/slimming_pruned-0.3.yaml | tee birds200_statistics_0.3.out
python src/param_statistics.py --config config/birds200/slimming_pruned-0.4.yaml | tee birds200_statistics_0.4.out
python src/param_statistics.py --config config/birds200/slimming_pruned-0.5.yaml | tee birds200_statistics_0.5.out
python src/param_statistics.py --config config/birds200/slimming_pruned-0.6.yaml | tee birds200_statistics_0.6.out
python src/param_statistics.py --config config/birds200/slimming_pruned-0.7.yaml | tee birds200_statistics_0.7.out

# aircrafts100
echo "compute aircrafts100 network statistics on resnet50"
python src/param_statistics.py --config config/aircrafts100/params_statistic_baseline.yaml | tee aircrafts100_statistics_baseline.out
python src/param_statistics.py --config config/aircrafts100/slimming_pruned-0.3.yaml | tee aircrafts100_statistics_0.3.out
python src/param_statistics.py --config config/aircrafts100/slimming_pruned-0.4.yaml | tee aircrafts100_statistics_0.4.out
python src/param_statistics.py --config config/aircrafts100/slimming_pruned-0.5.yaml | tee aircrafts100_statistics_0.5.out
python src/param_statistics.py --config config/aircrafts100/slimming_pruned-0.6.yaml | tee aircrafts100_statistics_0.6.out
python src/param_statistics.py --config config/aircrafts100/slimming_pruned-0.7.yaml | tee aircrafts100_statistics_0.7.out





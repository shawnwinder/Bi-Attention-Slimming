# # train baseline model
# echo "train baseline model..."
# python src/main.py --config config/birds200/slimming_baseline.yaml
# echo "done..."
# 
# # train sparse model with scale 1e-5
# echo ""
# echo ""
# echo ""
# echo ""
# echo ""
# echo "train sparse model with scale 1e-5..."
# python src/main.py --config config/birds200/slimming_sparse-1e-5.yaml
# echo "done..."

echo "train pruned model with pruning percent 30%..."
# python src/main.py --config config/birds200/slimming_pruned-0.3.yaml
python src/main.py --config config/birds200/slimming_pruned-0.4.yaml
echo "done..."

echo ""
echo ""
echo ""
echo ""
echo ""
echo "train pruned model with pruning percent 50%..."
python src/main.py --config config/birds200/slimming_pruned-0.5.yaml
echo "done..."

echo ""
echo ""
echo ""
echo ""
echo ""
echo "train pruned model with pruning percent 70%..."
# python src/main.py --config config/birds200/slimming_pruned-0.7.yaml
python src/main.py --config config/birds200/slimming_pruned-0.6.yaml
echo "done..."

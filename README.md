# DETPP-approach-Kikoriki
The team project as topic: Time interval modelling with DETPP approach

Основа кода взята из https://github.com/ivan-chai/hotpp-benchmark

Скрипты запусков из hotpp-benchmark

Запуск с трансформером + no BCE:

python -m hotpp.train \
  --config-path "$(pwd)/experiments/amazon/configs" \
  --config-name detection \
  name=amazon_beauty_detpp_simpletr_k48_no_bce_bs1 \
  detection_k=48 \
  num_classes=16 \
  ++module.loss.presence_loss_weight=0.0 \
  ++data_module.batch_size=1 \
  data_module.train_path="$(pwd)/experiments/amazon/data/train.parquet" \
  data_module.val_path="$(pwd)/experiments/amazon/data/val.parquet" \
  data_module.test_path="$(pwd)/experiments/amazon/data/test.parquet" \
  trainer.accelerator=cpu trainer.devices=1 trainer.precision=32 \
  data_module.num_workers=3 \
  hydra.job.chdir=false \
  logger._target_=pytorch_lightning.loggers.CSVLogger \
  logger.save_dir=lightning_logs \
  logger.name=amazon_beauty_detpp_simpletr_k48_no_bce_bs1 \
  '~logger.project' \
  module.seq_encoder._target_=hotpp.nn.TransformerEncoder \
  '~module.seq_encoder.rnn_partial' \
  '~module.seq_encoder.max_inference_context' \
  '~module.seq_encoder.inference_context_step' \
  +module.seq_encoder.max_context=128 \
  +module.seq_encoder.timestamps_field=timestamps \
  module.seq_encoder.max_time_delta=12 \
  +module.seq_encoder.transformer_partial._target_=hotpp.nn.SimpleTransformer \
  +module.seq_encoder.transformer_partial._partial_=true \
  +module.seq_encoder.transformer_partial.n_positions=256 \
  +module.seq_encoder.transformer_partial.n_embd=32 \
  +module.seq_encoder.transformer_partial.n_layer=2 \
  +module.seq_encoder.transformer_partial.n_head=2 \
  +module.seq_encoder.transformer_partial.dropout=0.1 \
  +module.seq_encoder.transformer_partial.causal=true

  Запуск с BCE + GRU:

  python -m hotpp.train \
  --config-path "$(pwd)/experiments/amazon/configs" \
  --config-name detection \
  name=amazon_beauty_detpp_gru_k48_with_bce_bs1 \
  detection_k=48 \
  num_classes=16 \
  ++module.loss.presence_loss_weight=1.0 \
  ++data_module.batch_size=1 \
  data_module.train_path="$(pwd)/experiments/amazon/data/train.parquet" \
  data_module.val_path="$(pwd)/experiments/amazon/data/val.parquet" \
  data_module.test_path="$(pwd)/experiments/amazon/data/test.parquet" \
  trainer.accelerator=cpu trainer.devices=1 trainer.precision=32 \
  data_module.num_workers=3 \
  hydra.job.chdir=false \
  logger._target_=pytorch_lightning.loggers.CSVLogger \
  logger.save_dir=lightning_logs \
  logger.name=amazon_beauty_detpp_gru_k48_with_bce_bs1 \
  '~logger.project'
  
Запуск Detpp на DetectionLoss на 3 головах на Amazon_beauty:

DS=data/Amazon_Beauty/GTS-q0.9-val_last-test_random

NUM_CLASSES=$(python - << 'PY'
import json
DS="data/Amazon_Beauty/GTS-q0.9-val_last-test_random"
print(len(json.load(open(f"{DS}/item_map.json"))))
PY
)

python -m hotpp.train \
  --config-dir experiments/amazon/configs \
  --config-name detection \
  name=amazon_beauty_detpp_gru_k3_nextitem \
  detection_k=3 \
  num_classes=$NUM_CLASSES \
  data_module.train_path=$DS/train.parquet \
  data_module.val_path=$DS/val.parquet \
  data_module.test_path=$DS/test3.parquet \
  +data_module.val_params.global_target_fields='[target_labels,target_timestamps]' \
  +data_module.test_params.global_target_fields='[target_labels,target_timestamps]' \
  ++module.val_metric._target_=hotpp.metrics.next_item.NextItemMetric \
  ++module.test_metric._target_=hotpp.metrics.next_item.NextItemMetric \
  ++module.val_metric.topk='[1,5,10,20,50,100]' \
  ++module.test_metric.topk='[1,5,10,20,50,100]' \
  trainer.accelerator=cpu trainer.devices=1 trainer.precision=32 \
  logger._target_=pytorch_lightning.loggers.CSVLogger \
  logger.save_dir=lightning_logs \
  logger.name=amazon_beauty_detpp_gru_k3_nextitem \
  '~logger.project' \
  hydra.job.chdir=false

Запуск Detpp на NextItemLoss на 1 голове на Amazon_beauty:

DS="data/Amazon_Beauty/GTS-q0.9-val_last-test_random"

NUM_CLASSES=$(python - << 'PY'
import json
DS="data/Amazon_Beauty/GTS-q0.9-val_last-test_random"
print(len(json.load(open(f"{DS}/item_map.json"))))
PY
)

python -m hotpp.train \
  --config-path ../experiments/amazon/configs \
  --config-name detection \
  name=amazon_beauty_nextitem_rankonly_gru_bs4 \
  detection_k=1 \
  num_classes=$NUM_CLASSES \
  ++data_module.batch_size=4 \
  data_module.train_path=$DS/train.parquet \
  data_module.val_path=$DS/val.parquet \
  data_module.test_path=$DS/test2.parquet \
  +data_module.val_params.global_target_fields='[target_labels,target_timestamps]' \
  +data_module.test_params.global_target_fields='[target_labels,target_timestamps]' \
  module.loss._target_=hotpp.losses.NextItemLoss \
  +module.loss.losses.labels._target_=hotpp.losses.CrossEntropyLoss \
  +module.loss.losses.labels.num_classes=${num_classes} \
  +module.loss.losses.timestamps._target_=hotpp.losses.TimeMAELoss \
  +module.loss.losses.timestamps.delta=start \
  +module.loss.losses.timestamps.max_delta=12 \
  ~module.loss.k \
  ~module.loss.horizon \
  ~module.loss.loss_subset \
  ~module.loss.prefetch_factor \
  ~module.loss.match_weights \
  ~module.loss.next_item_adapter \
  ~module.loss.next_item_loss \
  ++module.val_metric._target_=hotpp.metrics.next_item_ranking.NextItemRankingMetric \
  ++module.test_metric._target_=hotpp.metrics.next_item_ranking.NextItemRankingMetric \
  "++module.val_metric.topk=[1,5,10,20,50,100]" \
  "++module.test_metric.topk=[1,5,10,20,50,100]" \
  trainer.model_selection.metric='val/next-item-ndcg@20' \
  trainer.model_selection.mode=max \
  trainer.accelerator=cpu trainer.devices=1 trainer.precision=32 \
  data_module.num_workers=0 \
  hydra.job.chdir=false \
  logger._target_=pytorch_lightning.loggers.CSVLogger \
  logger.save_dir=lightning_logs \
  logger.name=amazon_beauty_nextitem_rankonly_gru_bs4 \
  '~logger.project'





  

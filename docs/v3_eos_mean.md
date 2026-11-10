# EOS + content mean ablation

Use configs/fingerprint_v3/qwen3_8b_eos_mean.json with the existing fingerprint_v3.py train command and the same frozen official CSVs and external CSV. Use a new output run name. This starts from the pretrained base, not an existing attribution adapter.

The last valid token (explicit EOS from encode) and the masked mean of final-layer content tokens are independently LayerNorm-normalized, concatenated and classified by Linear(2H,512), GELU, Dropout(0.1), Linear(512,18). All tokenizer special IDs and padding are excluded from the mean. A special-only input falls back to EOS. The decoder is evaluated once, without requesting every layer's hidden states. This ablation jointly changes pooling and head capacity; it does not isolate pooling alone.

The complete head remains named score, so PEFT modules_to_save and the existing head learning rate apply. The saved fold config reconstructs the custom class before loading the adapter. Old configs without pooling continue using the original EOS linear classifier. Keep this code installed for inference; the custom adapter is not directly usable with a generic Qwen sequence classifier.

Compare neural balanced accuracy to 0.5759259 (8B, external 2 epochs, official 4). Keep seed, datasets, sequence length and optimizer settings identical. This is a hypothesis, not a promised score improvement.

Run: python -m pytest tests/test_v3_decoder.py -q

No 8B/GPU quality or memory benchmark was performed here. This change does not add quantized or multi-GPU inference. Kaggle 16GB deployment still needs a separate implementation and measurement.

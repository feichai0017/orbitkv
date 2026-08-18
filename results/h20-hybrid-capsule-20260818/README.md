# H20 Hybrid Capsule Crossover

OrbitKV persisted GPT-OSS 20B state as two authenticated components: Full KV
for the complete prefix and SWA KV for only the final 128 tokens. All nine
paired runs matched cold output digests.

The host-file Capsule path was slower at 1K (+100.71%) and 4K (+27.87%), but
faster at 16K (-19.74%). The optimizer should therefore choose cold prefill for
short prefixes and Capsule restore only beyond the measured crossover.

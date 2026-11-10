# Lab 4: Depth cues in models

```
mkdir lab4 && cd lab4
wget https://utkarshmall13.github.io/cv8508-labs/lab4/setup.sh
sh setup.sh
. .venv/bin/activate
export HF_HOME=$PWD/models OPENROUTER_API_KEY=...     # the key is given in class
```

`lab4_tools.py` has the three depth models (`depth_direct`, `depth_generative`, `mllm_depth`), scoring at point pairs
(`pair_scores_map`, `height_prior_map`), `or_image` (generate or edit an image with a multimodal LLM), `or_key_status`
(budget left on the shared key) and the vanishing-point tools (`annotate_segments`, `fit_vp`, `draw_vp_figure`).
The handout is `Lab4_Depth_Cues_in_Models.pdf`.

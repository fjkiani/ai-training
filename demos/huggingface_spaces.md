# Deploying Gradio Demos to HuggingFace Spaces

Each domain's Gradio demo can be deployed to HuggingFace Spaces (free tier) and embedded in the portfolio via iframe.

## Prerequisites

1. Create a free [HuggingFace](https://huggingface.co) account
2. Generate an access token at https://huggingface.co/settings/tokens (scope: `write`)
3. Install the HF CLI: `pip install huggingface_hub`

## Deploying a Demo

For each domain, create a new Space:

```bash
# 1. Create a new Space (one-time)
huggingface-cli repo create ai-medical --type space --space-sdk gradio
huggingface-cli repo create ai-geospatial --type space --space-sdk gradio
huggingface-cli repo create ai-audio --type space --space-sdk gradio
huggingface-cli repo create ai-video --type space --space-sdk gradio

# 2. Clone the Space repo locally
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-medical
cd ai-medical

# 3. Copy the demo files
cp -r /path/to/ai-training/domains/medical/* .
cp -r /path/to/ai-training/shared .

# 4. Create app.py (Gradio entry point)
# Rename demo.py's build_demo() call to app.py:
cat > app.py << 'EOF'
from domains.medical.demo import build_demo
demo = build_demo()
demo.launch()
EOF

# 5. Create requirements.txt
cp /path/to/ai-training/requirements.txt .

# 6. Commit and push
git add .
git commit -m "Deploy medical imaging demo"
git push

# 7. The demo will be live at:
# https://YOUR_USERNAME-ai-medical.hf.space
```

## Embedding in the Portfolio

Once deployed, embed each Space in the portfolio using an iframe:

```html
<iframe
  src="https://YOUR_USERNAME-ai-medical.hf.space"
  frameborder="0"
  width="100%"
  height="600"
></iframe>
```

## Notes

- HuggingFace Spaces free tier: 2 vCPU, 16GB RAM (sufficient for all demos)
- Spaces sleep after inactivity — first load may take ~30s to wake
- Model checkpoints must be committed to the Space repo (or uploaded as HF model artifacts)
- For the video demo, CLIP model weights (~600MB) download on first run

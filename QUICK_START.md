# Quick Start Guide for KushalAdhyaru

## 🚀 Deploy in 5 Minutes

### Option 1: Automated Script (Easiest)

```bash
# Run this one command:
./deploy_to_hf_spaces.sh KushalAdhyaru negotiate-env
```

When prompted for credentials:
- Username: `KushalAdhyaru`
- Password: Your HuggingFace Access Token (get from https://huggingface.co/settings/tokens)

### Option 2: Manual Steps

See `DEPLOY_INSTRUCTIONS.md` for detailed step-by-step guide.

---

## ✅ After Deployment

Your environment will be live at:
```
https://kushaladhyaru-negotiate-env.hf.space
```

Test it:
```bash
curl https://kushaladhyaru-negotiate-env.hf.space/health
```

---

## 🎓 Train Your Model

1. **Upload to Google Colab**:
   - Open Google Colab: https://colab.research.google.com
   - Upload `colab_training.ipynb`
   - Make sure you have Colab Pro for A100 GPU

2. **Run Training**:
   - The notebook is already configured with your Space URL
   - Just run all cells!
   - Training takes ~1 hour on A100

3. **Save Model**:
   - Model automatically saves to: `KushalAdhyaru/negotiate-env-qwen-1.5b`

---

## 📋 Hackathon Checklist

- [ ] Deploy environment to HF Spaces
- [ ] Make dataset public: `mayukareddy/SyntheticSaasDataset`
- [ ] Train model in Colab Pro
- [ ] Upload trained model to HF Hub
- [ ] Push code to GitHub: `kushal511/negotiate-env`
- [ ] Create demo video/screenshots
- [ ] Submit to hackathon portal

---

## 🔗 Your URLs

| Resource | URL |
|----------|-----|
| HF Space | https://huggingface.co/spaces/KushalAdhyaru/negotiate-env |
| API Endpoint | https://kushaladhyaru-negotiate-env.hf.space |
| Trained Model | https://huggingface.co/KushalAdhyaru/negotiate-env-qwen-1.5b |
| GitHub Repo | https://github.com/kushal511/negotiate-env |
| Dataset | https://huggingface.co/datasets/mayukareddy/SyntheticSaasDataset |

---

## 🆘 Need Help?

1. Check `DEPLOY_INSTRUCTIONS.md` for detailed steps
2. Check Space logs: https://huggingface.co/spaces/KushalAdhyaru/negotiate-env (Logs tab)
3. Test locally first: `uvicorn negotiate_env.server.app:app --port 7860`

---

## 🎯 What You Have

✅ Complete OpenEnv environment (NegotiateEnv)
✅ 200 scenarios from HuggingFace dataset
✅ 4 opponent strategies with hidden information
✅ Full reward system (terminal + shaping + penalties)
✅ Constraint drift injection
✅ Docker deployment ready
✅ TRL GRPO training script
✅ Unsloth training script (T4 fallback)
✅ Colab notebook configured
✅ Baseline agents (random + rule-based)
✅ Evaluation scripts with metrics
✅ Demo script with transcripts
✅ All documentation

**You're ready to deploy and train! 🚀**

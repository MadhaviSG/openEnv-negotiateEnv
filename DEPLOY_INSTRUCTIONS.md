# Deploy NegotiateEnv to HuggingFace Spaces
## Customized for: KushalAdhyaru

Follow these exact steps to deploy your environment:

---

## Step 1: Create the Space on HuggingFace

1. Go to: https://huggingface.co/new-space
2. Fill in the form:
   - **Owner**: KushalAdhyaru
   - **Space name**: `negotiate-env`
   - **License**: MIT
   - **Select the Space SDK**: **Docker** ⚠️ IMPORTANT - Must be Docker!
   - **Space hardware**: CPU Basic (free)
   - **Visibility**: Public
3. Click **"Create Space"**

---

## Step 2: Clone Your New Space

Open your terminal in the `openEnv-negotiateEnv` directory and run:

```bash
git clone https://huggingface.co/spaces/KushalAdhyaru/negotiate-env hf_space
cd hf_space
```

---

## Step 3: Copy Files to Space

```bash
# Copy the environment package
cp -r ../negotiate_env .

# Copy deployment files
cp ../Dockerfile .
cp ../requirements.txt .
cp ../pyproject.toml .
cp ../.dockerignore .

# Copy the Space README
cp ../SPACE_README.md README.md
```

---

## Step 4: Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Deploy NegotiateEnv environment server"

# Push to HuggingFace
git push
```

If it asks for credentials:
- Username: `KushalAdhyaru`
- Password: Use your HuggingFace **Access Token** (not your password)
  - Get token at: https://huggingface.co/settings/tokens
  - Click "New token" → "Write" access → Copy the token

---

## Step 5: Watch the Build

1. Go to: https://huggingface.co/spaces/KushalAdhyaru/negotiate-env
2. Click the **"Logs"** tab
3. Wait for the build to complete (2-3 minutes)
4. Look for: `Running on http://0.0.0.0:7860`
5. Status should show **"Running"** with a green dot

---

## Step 6: Test Your Deployment

```bash
# Test health endpoint
curl https://kushaladhyaru-negotiate-env.hf.space/health

# Should return: {"status":"healthy"}
```

Or test in Python:

```python
import requests

# Test health
response = requests.get("https://kushaladhyaru-negotiate-env.hf.space/health")
print(response.json())  # Should print: {'status': 'healthy'}

# Test reset
response = requests.post("https://kushaladhyaru-negotiate-env.hf.space/reset", json={})
obs = response.json()
print(f"Scenario: {obs['context'][:100]}...")
print(f"Max turns: {obs['max_turns']}")
```

---

## Step 7: Use in Colab Training

Your environment URL is:
```
https://kushaladhyaru-negotiate-env.hf.space
```

In your Colab notebook (`colab_training.ipynb`):
1. Upload the notebook to Google Colab
2. Find the cell with `ENV_URL = "..."`
3. Change it to:
   ```python
   ENV_URL = "https://kushaladhyaru-negotiate-env.hf.space"
   ```
4. Run the training!

---

## Troubleshooting

### Build fails
- Check the "Logs" tab for error messages
- Make sure you selected "Docker" as the SDK (not Gradio or Streamlit)
- Verify all files were copied correctly

### Can't push to HuggingFace
- Make sure you're using an Access Token, not your password
- Token needs "Write" access
- Get token at: https://huggingface.co/settings/tokens

### Health check returns 404
- Wait a bit longer - build might still be in progress
- Check the Space status is "Running" (green dot)
- Verify the URL is correct (lowercase username)

---

## Quick Deploy Script (Alternative)

Instead of manual steps 2-4, you can use the automated script:

```bash
./deploy_to_hf_spaces.sh KushalAdhyaru negotiate-env
```

This will do steps 2-4 automatically!

---

## Your URLs After Deployment

- **Space Page**: https://huggingface.co/spaces/KushalAdhyaru/negotiate-env
- **API Endpoint**: https://kushaladhyaru-negotiate-env.hf.space
- **Health Check**: https://kushaladhyaru-negotiate-env.hf.space/health

---

## Next Steps After Deployment

1. ✅ Test the health endpoint
2. ✅ Upload `colab_training.ipynb` to Google Colab
3. ✅ Set `ENV_URL` to your Space URL
4. ✅ Run training with Colab Pro (A100 GPU)
5. ✅ Save trained model to HuggingFace Hub
6. ✅ Submit to hackathon!

---

Need help? Check the Space logs or let me know where you're stuck!

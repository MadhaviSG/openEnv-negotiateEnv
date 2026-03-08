# Push to HuggingFace - Final Step

## ✅ Steps 3-4 Complete!

All files are committed and ready to push. Now you need to authenticate and push.

---

## Step 5: Get Your HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Fill in:
   - **Name**: `negotiate-env-deploy`
   - **Type**: **Write** (important!)
4. Click **"Generate"**
5. **Copy the token** (starts with `hf_...`)

---

## Step 6: Push to HuggingFace

### Option 1: Using the Script (Easiest)

```bash
./push_to_hf.sh
```

It will ask for your token - paste it when prompted.

### Option 2: Manual Commands

```bash
cd hf_space

# Set up authentication (replace YOUR_TOKEN with your actual token)
git remote set-url origin https://KushalAdhyaru:YOUR_TOKEN@huggingface.co/spaces/KushalAdhyaru/negotiate-env

# Push
git push
```

---

## Step 7: Watch the Build

1. Go to: https://huggingface.co/spaces/KushalAdhyaru/negotiate-env
2. Click the **"Logs"** tab
3. Wait for the build (2-3 minutes)
4. Look for: `Running on http://0.0.0.0:7860`
5. Status should show **"Running"** ✅

---

## Step 8: Test Your Deployment

Once the Space shows "Running", test it:

```bash
curl https://kushaladhyaru-negotiate-env.hf.space/health
```

Expected response:
```json
{"status":"healthy"}
```

Or test with Python:

```python
import requests

# Test health
response = requests.get("https://kushaladhyaru-negotiate-env.hf.space/health")
print(response.json())  # Should print: {'status': 'healthy'}

# Test reset
response = requests.post("https://kushaladhyaru-negotiate-env.hf.space/reset", json={})
obs = response.json()
print(f"✅ Environment is working!")
print(f"Scenario: {obs['context'][:100]}...")
```

---

## ✅ What's Been Done

- ✅ Step 1: Created Space on HuggingFace
- ✅ Step 2: Cloned Space repository
- ✅ Step 3: Copied all files to hf_space/
- ✅ Step 4: Committed all changes
- ⏳ Step 5: **YOU DO THIS** - Get HuggingFace token
- ⏳ Step 6: **YOU DO THIS** - Push to HuggingFace
- ⏳ Step 7: Wait for build to complete
- ⏳ Step 8: Test the deployment

---

## 🎯 After Deployment

Your environment will be live at:
```
https://kushaladhyaru-negotiate-env.hf.space
```

Use this URL in your Colab training notebook!

---

## Troubleshooting

### "Authentication failed"
- Make sure you're using an Access Token, not your password
- Token must have "Write" access
- Get token at: https://huggingface.co/settings/tokens

### Build fails
- Check the "Logs" tab on your Space page
- Look for error messages
- Common issues:
  - Missing dependencies (already handled)
  - Port not 7860 (already correct)
  - Dockerfile syntax (already correct)

### Can't access the URL
- Wait for build to complete (check Logs tab)
- Make sure status shows "Running" with green dot
- Try again in 1-2 minutes

---

## Next: Train Your Model

Once your Space is running:

1. Upload `colab_training.ipynb` to Google Colab
2. The notebook already has your Space URL configured
3. Run all cells to train your model
4. Model will save to HuggingFace Hub automatically

---

Need help? Check the Space logs or let me know!

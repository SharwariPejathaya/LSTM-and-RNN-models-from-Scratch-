# 🚀 PUSH NLP SLOT-INTENT PROJECT TO GITHUB - ANACONDA PROMPT

## YOUR PROJECT: Slot Filling and Intent Classification

### Step 1: Open Anaconda Prompt
- Press `Win + S` (Windows Search)
- Type "Anaconda Prompt"
- Click on "Anaconda Prompt"

### Step 2: Navigate to Your Project Folder
```bash
cd Desktop\nlp_assignment2
```

### Step 3: Add README and Requirements Files

You need to add the README.md and requirements.txt files I created to your project folder.

**Option A - Download and copy manually:**
1. Download `README.md` and `requirements.txt` from the files I'm providing
2. Copy them into your `Desktop\nlp_assignment2` folder

**Option B - Create them directly:**
Create these files in your project folder (content provided in separate files)

### Step 4: Initialize Git Repository
```bash
git init
```

### Step 5: Configure Git (Replace with YOUR info)
```bash
git config --global user.name "Sharwari Pejathaya"
git config --global user.email "your.email@example.com"
```

### Step 6: Create .gitignore File

Create a file named `.gitignore` in your project folder with this content:
```
__pycache__/
*.pyc
.ipynb_checkpoints/
*.log
.DS_Store
```

**Or use the .gitignore file I'm providing!**

### Step 7: Add All Files
```bash
git add .
```

### Step 8: Check Status (Optional - see what will be committed)
```bash
git status
```

### Step 9: Create Initial Commit
```bash
git commit -m "Initial commit: Slot Filling and Intent Classification on ATIS and SLURP

- RNN and LSTM implementations for intent classification and slot filling
- Independent, Slot→Intent, Intent→Slot, and Joint multi-task models
- Achieved 98.5%% slot accuracy on ATIS and 78.7%% intent accuracy on SLURP
- Comprehensive analysis of multi-task learning effectiveness
- Complete model weights and experiment results"
```

### Step 10: Rename Branch to Main
```bash
git branch -M main
```

---

## CREATE GITHUB REPOSITORY

### Step 11: Go to GitHub
1. Open browser: https://github.com/new
2. **Repository name**: `nlp-slot-intent-classification` (or `atis-slurp-slot-intent`)
3. **Description**: `Multi-task learning for slot filling and intent classification on ATIS and SLURP datasets using RNN and LSTM`
4. Choose **Public** or **Private**
5. **DO NOT** check "Add a README file"
6. **DO NOT** check "Add .gitignore"
7. Click **"Create repository"**

---

## PUSH TO GITHUB

### Step 12: Add Remote (Replace YOUR_USERNAME)
```bash
git remote add origin https://github.com/YOUR_USERNAME/nlp-slot-intent-classification.git
```

**Example:** If your username is "SharwariPejathaya":
```bash
git remote add origin https://github.com/SharwariPejathaya/nlp-slot-intent-classification.git
```

### Step 13: Verify Remote
```bash
git remote -v
```

### Step 14: Push to GitHub
```bash
git push -u origin main
```

---

## 🔑 AUTHENTICATION

**Username:** Your GitHub username (e.g., SharwariPejathaya)

**Password:** Your **Personal Access Token** (NOT your GitHub password!)

### Get Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Name: `NLP Slot Intent Project`
4. Expiration: 90 days or No expiration
5. Select scopes: ✅ **repo** (full repository access)
6. Click **"Generate token"**
7. **COPY THE TOKEN** (looks like: `ghp_xxxxxxxxxxxxxxxxxxxx`)
8. Paste it when prompted for password

---

## ✅ VERIFY SUCCESS

Visit: `https://github.com/YOUR_USERNAME/nlp-slot-intent-classification`

You should see:
- ✅ README.md with comprehensive project overview
- ✅ Assignment_2_Report.pdf
- ✅ All your notebooks (.ipynb files)
- ✅ All model weights (.pth files)
- ✅ Experiment results (.json files)
- ✅ requirements.txt

---

## 🎯 COMPLETE COMMAND SEQUENCE

```bash
# Navigate to project
cd Desktop\nlp_assignment2

# Initialize Git
git init

# Configure Git (CHANGE TO YOUR INFO!)
git config --global user.name "Sharwari Pejathaya"
git config --global user.email "your.email@example.com"

# Add all files
git add .

# Commit
git commit -m "Initial commit: Slot Filling and Intent Classification on ATIS and SLURP"

# Set branch to main
git branch -M main

# Add remote (CHANGE YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/nlp-slot-intent-classification.git

# Push
git push -u origin main
```

---

## ⚠️ IMPORTANT: LARGE FILES

Your .pth model files might be large. If Git complains about file size:

### Option 1: Remove model files from Git (keep locally)
```bash
git rm --cached *.pth
echo "*.pth" >> .gitignore
git add .gitignore
git commit -m "Remove large model files, add to .gitignore"
git push
```

### Option 2: Use Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
git push
```

---

## 🆘 TROUBLESHOOTING

### Error: "file too large"
- Use Git LFS (see above) or remove .pth files from repo

### Error: "Authentication failed"  
- Use Personal Access Token, NOT password

### Error: "Repository not found"
- Make sure you created the repo on GitHub
- Check the URL: `git remote -v`

---

## 📱 AFTER SUCCESSFUL PUSH

### Add Topics to Repository:
1. Go to your GitHub repository
2. Click "Add topics" (under About)
3. Add: `nlp`, `machine-learning`, `rnn`, `lstm`, `slot-filling`, `intent-classification`, `multi-task-learning`, `pytorch`, `atis`, `slurp`, `dialogue-systems`

---

## 🔄 FUTURE UPDATES

```bash
cd Desktop\nlp_assignment2
git add .
git commit -m "Description of changes"
git push
```

---

**Good luck! Your NLP project will be live on GitHub! 🎉**

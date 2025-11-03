#!/bin/bash

# Stochastic Convolution Neural Networks - Push to GitHub
# This script initializes git, creates a new GitHub repository, and pushes the project

set -e  # Exit on any error

# Configuration
REPO_NAME="stochastic-convolution-neural-networks"
REPO_DESCRIPTION="Implementation of stochastic convolution techniques in neural networks for improved regularization"
REPO_TOPICS="deep-learning,pytorch,stochastic-neural-networks,convolutional-neural-networks,regularization"

echo "🚀 Starting GitHub push for $REPO_NAME"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed. Please install it first:"
    echo "   - Ubuntu/Debian: sudo apt install gh"
    echo "   - macOS: brew install gh"
    echo "   - Or download from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated with GitHub
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub. Please run: gh auth login"
    exit 1
fi

# Initialize git repository
echo "📝 Initializing git repository..."
git init
git add .

# Create .gitignore for deep learning project
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# ML/DL specific
models/
checkpoints/
logs/
tensorboard/
*.h5
*.pb
*.pkl
*.joblib
*.model
*.pt
*.pth
*.onnx

# Large datasets
data/
datasets/
*.zip
*.tar.gz

# Results and outputs
results/
outputs/
predictions/
plots/
*.png
*.jpg
*.jpeg
*.gif
*.pdf

# Temporary files
tmp/
temp/
*.tmp
*.temp
*~

# Log files
*.log
training.log
EOF

git add .gitignore
git commit -m "Initial commit: Stochastic Convolution Neural Networks

- Implementation of stochastic convolution techniques for neural networks
- Randomized convolution operations for improved regularization
- PyTorch-based stochastic neural network architecture
- Ensemble-like effects through controlled randomization
- Comprehensive analysis of stochastic vs. standard convolution

Demonstrates advanced deep learning concepts: stochastic regularization,
probabilistic neural computation, and ensemble learning within single models."

# Create GitHub repository
echo "📦 Creating GitHub repository..."
if gh repo create "$REPO_NAME" --description "$REPO_DESCRIPTION" --public --source=. --remote=origin --push; then
    echo "✅ Repository created and pushed successfully!"
    echo "🌐 Repository URL: https://github.com/$(gh api user -q '.login')/$REPO_NAME"
else
    echo "❌ Failed to create repository. It may already exist or you may not have permission."
    echo "   Try a different name or check your GitHub authentication."
    exit 1
fi

# Add topics (tags)
echo "🏷️  Adding repository topics..."
gh repo edit "$REPO_NAME" --add-topic "$REPO_TOPICS"

echo "🎉 Project successfully pushed to GitHub!"
echo "   Repository: https://github.com/$(gh api user -q '.login')/$REPO_NAME"
echo ""
echo "📖 Don't forget to:"
echo "   - Update the README with your contact information"
echo "   - Test the notebook: jupyter notebook stochasticrandomconvolution.ipynb"
echo "   - Add requirements.txt for PyTorch dependencies"
echo "   - Consider adding experimental results or model checkpoints"
echo "   - Document the stochastic parameters and their effects"

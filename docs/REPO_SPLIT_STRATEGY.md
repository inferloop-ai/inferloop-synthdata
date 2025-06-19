# Repository Split Strategy for inferloop-synthdata

## Overview
This document outlines the strategy to split the inferloop-synthdata monorepo into separate repositories for each synthetic data type. Since each subdirectory is completely independent, the process is straightforward.

## Repository Structure

### Current Monorepo Structure:
- `audio/` - Audio synthetic data generation
- `codeapidsl/` - Code/API/DSL synthetic data generation
- `image/` - Image synthetic data generation
- `tabular/` - Tabular data generation
- `textnlp/` - Text/NLP synthetic data generation
- `tsiot/` - Time Series IoT data generation (Go)
- `video/` - Video synthetic data generation
- `docs/structured-documents-synthetic-data/` - Structured documents generation

### Target Repository Names:
- `inferloop-synthdata-audio`
- `inferloop-synthdata-codeapidsl`
- `inferloop-synthdata-image`
- `inferloop-synthdata-tabular`
- `inferloop-synthdata-textnlp`
- `inferloop-synthdata-tsiot`
- `inferloop-synthdata-video`
- `inferloop-synthdata-structured-docs`

## Split Process

### Step 1: Create branches with git subtree split
This preserves the full git history for each module.

```bash
# Run from the root of inferloop-synthdata repository
git subtree split --prefix=audio -b audio-split
git subtree split --prefix=codeapidsl -b codeapidsl-split
git subtree split --prefix=image -b image-split
git subtree split --prefix=tabular -b tabular-split
git subtree split --prefix=textnlp -b textnlp-split
git subtree split --prefix=tsiot -b tsiot-split
git subtree split --prefix=video -b video-split
git subtree split --prefix=docs/structured-documents-synthetic-data -b structured-docs-split
```

### Step 2: Create new repositories
For each module, create a new repository and push the split branch:

```bash
# Example for audio module
mkdir inferloop-synthdata-audio
cd inferloop-synthdata-audio
git init
git pull ../inferloop-synthdata audio-split
git remote add origin https://github.com/inferloop-ai/inferloop-synthdata-audio.git
git push -u origin main

# Repeat for other modules
```

### Step 3: Add root files
Copy necessary root files to each new repository:

```bash
# From within each new repository
cp ../inferloop-synthdata/LICENSE .
cp ../inferloop-synthdata/.gitignore .
git add LICENSE .gitignore
git commit -m "Add root files from monorepo"
git push
```

### Step 4: Create repository-specific README
Each repository should have its own README:

```bash
# Create a basic README for each module
echo "# InferLoop SynthData - Audio

Synthetic audio data generation module.

Previously part of the [inferloop-synthdata](https://github.com/inferloop-ai/inferloop-synthdata) monorepo.

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

See documentation in the \`docs/\` directory.
" > README.md

git add README.md
git commit -m "Add module-specific README"
git push
```

## Automation Script

Create a script to automate the entire process:

```bash
#!/bin/bash
# split-all-repos.sh

# Array of modules and their paths
declare -A modules=(
    ["audio"]="audio"
    ["codeapidsl"]="codeapidsl"
    ["image"]="image"
    ["tabular"]="tabular"
    ["textnlp"]="textnlp"
    ["tsiot"]="tsiot"
    ["video"]="video"
    ["structured-docs"]="docs/structured-documents-synthetic-data"
)

# Step 1: Create all split branches
echo "Creating split branches..."
for module in "${!modules[@]}"; do
    path="${modules[$module]}"
    echo "Splitting $module from $path..."
    git subtree split --prefix="$path" -b "${module}-split"
done

# Step 2: Create and push new repositories
for module in "${!modules[@]}"; do
    repo_name="inferloop-synthdata-${module}"
    echo "Creating repository $repo_name..."
    
    # Create new repo directory
    mkdir "../$repo_name"
    cd "../$repo_name"
    
    # Initialize and pull the split branch
    git init
    git pull ../inferloop-synthdata "${module}-split"
    
    # Add root files
    cp ../inferloop-synthdata/LICENSE .
    cp ../inferloop-synthdata/.gitignore .
    git add LICENSE .gitignore
    git commit -m "Add root files from monorepo"
    
    # Create basic README
    echo "# InferLoop SynthData - ${module^}

Synthetic ${module} data generation module.

Previously part of the [inferloop-synthdata](https://github.com/inferloop-ai/inferloop-synthdata) monorepo.
" > README.md
    git add README.md
    git commit -m "Add module-specific README"
    
    # Add remote and push (uncomment when ready)
    # git remote add origin "https://github.com/inferloop-ai/$repo_name.git"
    # git push -u origin main
    
    cd ../inferloop-synthdata
done

echo "Split complete!"
```

## Post-Split Tasks

### 1. Update the Original Repository
Add a notice to the monorepo README:

```markdown
# ⚠️ Repository Split Notice

This monorepo has been split into separate repositories for better modularity:

- [inferloop-synthdata-audio](https://github.com/inferloop-ai/inferloop-synthdata-audio)
- [inferloop-synthdata-codeapidsl](https://github.com/inferloop-ai/inferloop-synthdata-codeapidsl)
- [inferloop-synthdata-image](https://github.com/inferloop-ai/inferloop-synthdata-image)
- [inferloop-synthdata-tabular](https://github.com/inferloop-ai/inferloop-synthdata-tabular)
- [inferloop-synthdata-textnlp](https://github.com/inferloop-ai/inferloop-synthdata-textnlp)
- [inferloop-synthdata-tsiot](https://github.com/inferloop-ai/inferloop-synthdata-tsiot)
- [inferloop-synthdata-video](https://github.com/inferloop-ai/inferloop-synthdata-video)
- [inferloop-synthdata-structured-docs](https://github.com/inferloop-ai/inferloop-synthdata-structured-docs)

This repository is now archived. Please use the individual repositories above.
```

### 2. Archive or Delete Original
Options:
- Archive the monorepo on GitHub (recommended)
- Convert to a meta-repository with links
- Delete after confirming all splits are successful

### 3. Update External References
- Update any documentation linking to the monorepo
- Update CI/CD pipelines
- Update package references

## Benefits of This Approach

1. **Preserves History**: Git subtree split maintains full commit history
2. **Clean Split**: No remnants of other modules in each repository
3. **Independent Development**: Each team can work without affecting others
4. **Smaller Repository Size**: Faster clones and operations
5. **Granular Access Control**: Different permissions per repository
6. **Independent Versioning**: Each module can have its own version scheme

## Timeline

- **Day 1**: Test split process on a fork
- **Day 2**: Execute splits and create repositories
- **Day 3**: Update documentation and external references
- **Day 4**: Archive original repository

## Rollback Plan

If needed, the original monorepo remains intact until explicitly archived. All split operations are non-destructive to the source repository.
#\!/bin/bash
# Script to reorganize the repository structure

echo "Starting directory reorganization..."

# Create new directory structure
echo "Creating project-meta directories..."
mkdir -p project-meta/design-docs
mkdir -p project-meta/audit-reports
mkdir -p project-meta/setup-scripts
mkdir -p project-meta/test-artifacts

# Move structured-documents-synthetic-data to repository root
echo "Moving structured-documents-synthetic-data to repository root..."
if [ -d "docs/docs/structured-documents-synthetic-data" ]; then
    mv docs/docs/structured-documents-synthetic-data .
    echo "✓ Moved structured-documents-synthetic-data to root"
else
    echo "✗ structured-documents-synthetic-data directory not found in docs/docs/"
fi

# Move design and documentation files
echo "Moving design documents..."
for file in docs/*.md docs/*.ini docs/*.pdf; do
    if [ -f "$file" ]; then
        mv "$file" project-meta/design-docs/ 2>/dev/null
    fi
done

# Move audit files
echo "Moving audit reports..."
for file in docs/codebase_audit* docs/audit_* docs/*test_results.json; do
    if [ -f "$file" ]; then
        mv "$file" project-meta/audit-reports/ 2>/dev/null
    fi
done

# Move setup scripts
echo "Moving setup scripts..."
for file in docs/repo_setup*.sh; do
    if [ -f "$file" ]; then
        mv "$file" project-meta/setup-scripts/ 2>/dev/null
    fi
done

# Move test artifacts
echo "Moving test artifacts..."
for file in docs/test_*.py; do
    if [ -f "$file" ]; then
        mv "$file" project-meta/test-artifacts/ 2>/dev/null
    fi
done

# Move test directories
for dir in docs/test_*/; do
    if [ -d "$dir" ]; then
        mv "$dir" project-meta/test-artifacts/ 2>/dev/null
    fi
done

# Check if docs directory has any remaining important files
echo "Checking remaining files in docs..."
if [ -z "$(ls -A docs 2>/dev/null)" ]; then
    echo "docs directory is empty and can be removed"
else
    echo "docs directory still contains files/directories:"
    ls -la docs/
fi

echo "Reorganization complete\!"
echo ""
echo "New structure:"
echo "├── structured-documents-synthetic-data/  # Main project"
echo "├── project-meta/"
echo "│   ├── design-docs/     # Design documents and specs"
echo "│   ├── audit-reports/   # Code audits and test results"
echo "│   ├── setup-scripts/   # Repository setup scripts"
echo "│   └── test-artifacts/  # Test scripts and environments"
echo "└── docs/               # (check if empty)"
EOF < /dev/null
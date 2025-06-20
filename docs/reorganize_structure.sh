#!/bin/bash

# Script to reorganize the repository structure
# Run this from the repository root: /Users/dattamiruke/INFERLOOP/GitHub/inferloop-synthdata

echo "Starting directory reorganization..."

# Create new directory structure
echo "Creating project-meta directories..."
mkdir -p project-meta/{design-docs,audit-reports,setup-scripts,test-artifacts}

# Move structured-documents-synthetic-data to repository root
echo "Moving structured-documents-synthetic-data to repository root..."
if [ -d "docs/structured-documents-synthetic-data" ]; then
    mv docs/structured-documents-synthetic-data .
    echo "✓ Moved structured-documents-synthetic-data to root"
else
    echo "✗ structured-documents-synthetic-data directory not found"
fi

# Move design and documentation files
echo "Moving design documents..."
mv docs/*.md project-meta/design-docs/ 2>/dev/null
mv docs/*.ini project-meta/design-docs/ 2>/dev/null
mv docs/*.pdf project-meta/design-docs/ 2>/dev/null

# Move audit files
echo "Moving audit reports..."
mv docs/codebase_audit* project-meta/audit-reports/ 2>/dev/null
mv docs/audit_* project-meta/audit-reports/ 2>/dev/null
mv docs/*test_results.json project-meta/audit-reports/ 2>/dev/null

# Move setup scripts
echo "Moving setup scripts..."
mv docs/repo_setup*.sh project-meta/setup-scripts/ 2>/dev/null

# Move test artifacts
echo "Moving test artifacts..."
mv docs/test_*.py project-meta/test-artifacts/ 2>/dev/null
mv docs/test_*/ project-meta/test-artifacts/ 2>/dev/null

# Check if docs directory has any remaining important files
echo "Checking remaining files in docs..."
remaining_files=$(ls -la docs/ 2>/dev/null | grep -v "^total" | grep -v "^d" | wc -l)
remaining_dirs=$(ls -la docs/ | grep "^d" | grep -v "\.$" | wc -l)

if [ $remaining_files -eq 0 ] && [ $remaining_dirs -eq 0 ]; then
    echo "docs directory is empty and can be removed"
    # Uncomment the next line to actually remove it
    # rmdir docs
else
    echo "docs directory still contains files/directories:"
    ls -la docs/
fi

echo "Reorganization complete!"
echo ""
echo "New structure:"
echo "├── structured-documents-synthetic-data/  # Main project"
echo "├── project-meta/"
echo "│   ├── design-docs/     # Design documents and specs"
echo "│   ├── audit-reports/   # Code audits and test results"
echo "│   ├── setup-scripts/   # Repository setup scripts"
echo "│   └── test-artifacts/  # Test scripts and environments"
echo "└── docs/               # (check if empty)"
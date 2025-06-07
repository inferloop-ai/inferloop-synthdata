#!/bin/bash

# Canvas File Splitter
# Splits a canvas file based on delimiter patterns like "====== filename ======="
# 
# Usage: ./canvas_splitter.sh input_file.txt [output_directory]
#
# Patterns supported:
# - ====== filename.ext =======
# - # filename.ext
# - // filename.ext

set -euo pipefail

# Default values
INPUT_FILE=""
OUTPUT_DIR="output"
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] INPUT_FILE [OUTPUT_DIR]

Split a canvas file into separate files based on delimiter patterns.

Arguments:
    INPUT_FILE      Path to the input canvas file
    OUTPUT_DIR      Output directory (default: output)

Options:
    -v, --verbose   Enable verbose output
    -h, --help      Show this help message

Supported patterns:
    ====== filename.ext =======
    # filename.ext
    // filename.ext

Examples:
    $0 canvas.txt
    $0 canvas.txt ./split_files
    $0 -v canvas.txt output_folder
EOF
}

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                if [[ -z "$INPUT_FILE" ]]; then
                    INPUT_FILE="$1"
                elif [[ -z "$OUTPUT_DIR" || "$OUTPUT_DIR" == "output" ]]; then
                    OUTPUT_DIR="$1"
                else
                    log_error "Too many arguments"
                    usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$INPUT_FILE" ]]; then
        log_error "Input file is required"
        usage
        exit 1
    fi

    if [[ ! -f "$INPUT_FILE" ]]; then
        log_error "Input file does not exist: $INPUT_FILE"
        exit 1
    fi
}

# Clean filename for filesystem compatibility
clean_filename() {
    local filename="$1"
    
    # Remove leading/trailing whitespace and quotes
    filename=$(echo "$filename" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed 's/^["'\'']*//;s/["'\'']*$//')
    
    # Replace invalid characters with underscores
    filename=$(echo "$filename" | sed 's/[<>:"|?*]/_/g')
    
    # Remove multiple consecutive slashes
    filename=$(echo "$filename" | sed 's|/\+|/|g')
    
    echo "$filename"
}

# Check if filename should be skipped
should_skip_filename() {
    local filename="$1"
    
    # Skip if empty
    [[ -z "$filename" ]] && return 0
    
    # Skip common non-filename patterns
    if echo "$filename" | grep -qiE '^(example|usage|note|todo|fixme|main|function|class|method)'; then
        return 0
    fi
    
    # Skip numbered items like "1. Something"
    if echo "$filename" | grep -qE '^\d+\.'; then
        return 0
    fi
    
    # Skip all caps headers
    if echo "$filename" | grep -qE '^[A-Z][A-Z\s]*$'; then
        return 0
    fi
    
    return 1
}

# Main splitting function using awk
split_canvas_file() {
    local input_file="$1"
    local output_dir="$2"
    
    log_info "Splitting file: $input_file"
    log_info "Output directory: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Use awk to parse and split the file
    awk -v output_dir="$output_dir" -v verbose="$VERBOSE" '
    BEGIN {
        current_file = ""
        content = ""
        files_created = 0
        skipped = 0
    }
    
    # Function to clean filename
    function clean_filename(filename) {
        # Remove leading/trailing whitespace and quotes
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", filename)
        gsub(/^["'\'']+|["'\'']+$/, "", filename)
        
        # Replace invalid characters
        gsub(/[<>:"|?*]/, "_", filename)
        
        # Clean up paths
        gsub(/\/+/, "/", filename)
        
        return filename
    }
    
    # Function to check if filename should be skipped
    function should_skip(filename) {
        if (filename == "") return 1
        if (match(filename, /^(example|usage|note|todo|fixme|main|function|class|method)/)) return 1
        if (match(filename, /^[0-9]+\./)) return 1
        if (match(filename, /^[A-Z][A-Z[:space:]]*$/)) return 1
        return 0
    }
    
    # Function to save current content
    function save_current_file() {
        if (current_file != "" && content != "") {
            # Create directory if needed
            dir_cmd = "dirname \"" output_dir "/" current_file "\""
            dir_cmd | getline file_dir
            close(dir_cmd)
            
            system("mkdir -p \"" file_dir "\"")
            
            # Write content to file
            output_file = output_dir "/" current_file
            print content > output_file
            close(output_file)
            
            if (verbose == "true") {
                printf "[DEBUG] Saved %d lines to: %s\n", split(content, lines, "\n"), current_file
            }
            
            printf "[INFO] Created file: %s\n", current_file
            files_created++
            
            content = ""
        }
    }
    
    # Pattern 1: ====== filename =======
    /^# =+[[:space:]]*[^[:space:]]+.*[[:space:]]*=+$/ {
        # Save previous file
        save_current_file()
        
        # Extract filename
        match($0, /^=+[[:space:]]*([^[:space:]]+.*?)[[:space:]]*=+$/, arr)
        filename = clean_filename(arr[1])
        
        if (should_skip(filename)) {
            if (verbose == "true") {
                printf "[DEBUG] Skipping: %s\n", filename
            }
            current_file = ""
            skipped++
        } else {
            current_file = filename
            if (verbose == "true") {
                printf "[DEBUG] Found section: %s\n", filename
            }
        }
        next
    }
    
    # Pattern 2: # filename.ext
    /^#[[:space:]]+[^[:space:]]+\.(py|txt|yaml|yml|json|js|html|css|md|sh|dockerfile|sql|c|cpp|h|java)([[:space:]]|$)/ {
        # Save previous file
        save_current_file()
        
        # Extract filename
        match($0, /^#[[:space:]]+([^[:space:]]+\.[^[:space:]]+)/, arr)
        filename = clean_filename(arr[1])
        
        if (should_skip(filename)) {
            if (verbose == "true") {
                printf "[DEBUG] Skipping: %s\n", filename
            }
            current_file = ""
            skipped++
        } else {
            current_file = filename
            if (verbose == "true") {
                printf "[DEBUG] Found section: %s\n", filename
            }
        }
        next
    }
    
    # Pattern 3: // filename.ext
    /^\/\/[[:space:]]+[^[:space:]]+\.(js|cpp|c|h|java)([[:space:]]|$)/ {
        # Save previous file
        save_current_file()
        
        # Extract filename
        match($0, /^\/\/[[:space:]]+([^[:space:]]+\.[^[:space:]]+)/, arr)
        filename = clean_filename(arr[1])
        
        if (should_skip(filename)) {
            if (verbose == "true") {
                printf "[DEBUG] Skipping: %s\n", filename
            }
            current_file = ""
            skipped++
        } else {
            current_file = filename
            if (verbose == "true") {
                printf "[DEBUG] Found section: %s\n", filename
            }
        }
        next
    }
    
    # Collect content for current file
    {
        if (current_file != "") {
            if (content != "") {
                content = content "\n" $0
            } else {
                content = $0
            }
        }
    }
    
    END {
        # Save the last file
        save_current_file()
        
        printf "[INFO] Processing complete!\n"
        printf "[INFO] Files created: %d\n", files_created
        printf "[INFO] Sections skipped: %d\n", skipped
    }
    ' "$input_file"
}

# Main function
main() {
    parse_args "$@"
    
    log_info "Starting canvas file splitter..."
    log_debug "Input file: $INPUT_FILE"
    log_debug "Output directory: $OUTPUT_DIR"
    
    # Check if input file exists and is readable
    if [[ ! -r "$INPUT_FILE" ]]; then
        log_error "Cannot read input file: $INPUT_FILE"
        exit 1
    fi
    
    # Split the file
    split_canvas_file "$INPUT_FILE" "$OUTPUT_DIR"
    
    log_info "Canvas file splitting completed!"
    log_info "Check the '$OUTPUT_DIR' directory for the split files."
}

# Run main function with all arguments
main "$@"

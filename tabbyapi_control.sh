#!/bin/bash
# =============================================================================
# TabbyAPI Control Script
# =============================================================================
# Manage your TabbyAPI server: start, stop, restart, status, logs
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/gpu-scripts}"
TABBY_DIR="$PROJECT_DIR/tabbyAPI"
PID_FILE="$PROJECT_DIR/tabbyapi.pid"
LOG_FILE="$PROJECT_DIR/logs/tabbyapi.log"
ENV_FILE="$PROJECT_DIR/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Functions
# =============================================================================

get_pid() {
    if [[ -f "$PID_FILE" ]]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

is_running() {
    local pid=$(get_pid)
    if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

get_port() {
    if [[ -f "$ENV_FILE" ]]; then
        source "$ENV_FILE"
    fi
    echo "${TABBY_PORT:-5000}"
}

status_cmd() {
    local port=$(get_port)

    echo "TabbyAPI Server Status"
    echo "======================"

    if is_running; then
        local pid=$(get_pid)
        log_success "Running (PID: $pid)"

        # Get process info
        echo ""
        echo "Process Info:"
        ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | head -c 200
        echo ""

        # Check HTTP endpoint
        echo ""
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            log_success "HTTP endpoint healthy: http://localhost:$port"

            # Try to get model info
            if command -v python3 &> /dev/null; then
                MODEL_INFO=$(curl -s "http://localhost:$port/v1/models" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'])" 2>/dev/null || echo "")
                if [[ -n "$MODEL_INFO" ]]; then
                    log_info "Model loaded: $MODEL_INFO"
                fi
            fi
        else
            log_warn "HTTP endpoint not responding (may still be loading model)"
        fi

        echo ""
        echo "Endpoints:"
        echo "  Health:  http://localhost:$port/health"
        echo "  API:     http://localhost:$port/v1"
        echo "  Models:  http://localhost:$port/v1/models"
        echo "  Docs:    http://localhost:$port/docs"

    else
        log_warn "Not running"
        if [[ -f "$PID_FILE" ]]; then
            log_info "Stale PID file found, removing..."
            rm -f "$PID_FILE"
        fi
    fi

    echo ""
    log_info "Log file: $LOG_FILE"
}

start_cmd() {
    if is_running; then
        log_warn "TabbyAPI is already running (PID: $(get_pid))"
        return 1
    fi

    log_info "Starting TabbyAPI server..."

    if [[ ! -f "$PROJECT_DIR/setup_tabbyapi.sh" ]]; then
        log_error "Setup script not found: $PROJECT_DIR/setup_tabbyapi.sh"
        return 1
    fi

    bash "$PROJECT_DIR/setup_tabbyapi.sh"
}

stop_cmd() {
    if ! is_running; then
        log_warn "TabbyAPI is not running"
        if [[ -f "$PID_FILE" ]]; then
            rm -f "$PID_FILE"
        fi
        return 0
    fi

    local pid=$(get_pid)
    log_info "Stopping TabbyAPI server (PID: $pid)..."

    # Try graceful shutdown
    kill "$pid" 2>/dev/null || true

    # Wait up to 15 seconds (model unloading can take time)
    for i in {1..15}; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            log_success "TabbyAPI stopped"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    log_warn "Forcing shutdown..."
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    log_success "TabbyAPI stopped (forced)"
}

restart_cmd() {
    log_info "Restarting TabbyAPI server..."
    stop_cmd
    sleep 3
    start_cmd
}

logs_cmd() {
    if [[ ! -f "$LOG_FILE" ]]; then
        log_error "Log file not found: $LOG_FILE"
        return 1
    fi

    if [[ "${1:-}" == "-f" ]] || [[ "${1:-}" == "--follow" ]]; then
        log_info "Following logs (Ctrl+C to exit)..."
        tail -f "$LOG_FILE"
    else
        log_info "Showing last 50 lines of logs..."
        tail -n 50 "$LOG_FILE"
    fi
}

test_cmd() {
    if ! is_running; then
        log_error "TabbyAPI is not running. Start it first with: $0 start"
        return 1
    fi

    local port=$(get_port)

    # Check if server is responding
    if ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        log_warn "Server is running but not responding yet. May still be loading model."
        log_info "Check logs: $0 logs -f"
        return 1
    fi

    if [[ -f "$PROJECT_DIR/test_tabbyapi.py" ]]; then
        python3 "$PROJECT_DIR/test_tabbyapi.py"
    else
        log_error "Test script not found: $PROJECT_DIR/test_tabbyapi.py"
        return 1
    fi
}

download_cmd() {
    log_info "Starting model download..."

    if [[ ! -d "$TABBY_DIR" ]]; then
        log_error "TabbyAPI not installed. Run setup first: $PROJECT_DIR/setup_tabbyapi.sh"
        return 1
    fi

    cd "$TABBY_DIR"

    if [[ ! -d "venv" ]]; then
        log_error "Virtual environment not found. Run setup first."
        return 1
    fi

    source venv/bin/activate

    # Load config
    if [[ -f "$ENV_FILE" ]]; then
        source "$ENV_FILE"
    fi

    MODEL_REPO="${1:-${TABBY_MODEL_REPO:-bartowski/Qwen2.5-Coder-32B-Instruct-exl2}}"
    MODEL_REVISION="${2:-${TABBY_MODEL_REVISION:-4.0bpw}}"
    MODEL_DIR="${TABBY_MODEL_DIR:-/workspace/models}"

    log_info "Downloading: $MODEL_REPO (revision: $MODEL_REVISION)"
    log_info "To: $MODEL_DIR"

    HF_TOKEN_ARG=""
    if [[ -n "${HF_TOKEN:-}" ]]; then
        HF_TOKEN_ARG="--token $HF_TOKEN"
    fi

    huggingface-cli download \
        $HF_TOKEN_ARG \
        "$MODEL_REPO" \
        --revision "$MODEL_REVISION" \
        --local-dir "$MODEL_DIR/$MODEL_REPO" \
        --local-dir-use-symlinks False

    log_success "Model downloaded successfully"
    deactivate
}

gpu_cmd() {
    log_info "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU %s: %s | VRAM: %dMB/%dMB (%.1f%%) | Util: %d%% | Temp: %d°C\n", $1, $2, $3, $4, ($3/$4)*100, $5, $6}'
}

usage() {
    echo "TabbyAPI Control Script"
    echo ""
    echo "Usage: $0 {start|stop|restart|status|logs|test|download|gpu}"
    echo ""
    echo "Commands:"
    echo "  start      Start TabbyAPI server"
    echo "  stop       Stop TabbyAPI server"
    echo "  restart    Restart TabbyAPI server"
    echo "  status     Show server status"
    echo "  logs       Show logs (use -f to follow)"
    echo "  test       Run test script"
    echo "  download   Download a model [repo] [revision]"
    echo "  gpu        Show GPU status"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 logs -f"
    echo "  $0 download bartowski/Qwen2.5-Coder-32B-Instruct-exl2 4.0bpw"
    echo "  $0 gpu"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

case "${1:-}" in
    start)
        start_cmd
        ;;
    stop)
        stop_cmd
        ;;
    restart)
        restart_cmd
        ;;
    status)
        status_cmd
        ;;
    logs)
        logs_cmd "${2:-}"
        ;;
    test)
        test_cmd
        ;;
    download)
        download_cmd "${2:-}" "${3:-}"
        ;;
    gpu)
        gpu_cmd
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        usage
        exit 1
        ;;
esac

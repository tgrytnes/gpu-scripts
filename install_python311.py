#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11.9}"
PYTHON_PREFIX="${PYTHON_PREFIX:-/usr/local}"
POETRY_VERSION="${POETRY_VERSION:-1.8.3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}/..}"

# Prefer repo .env, fall back to /root/.env, allow ENV_FILE override.
DEFAULT_ENV_FILE="${REPO_DIR}/.env"
if [[ ! -f "${DEFAULT_ENV_FILE}" && -f "/root/gpu-scripts/.env" ]]; then
  DEFAULT_ENV_FILE="/root/gpu-scripts/.env"
fi
ENV_FILE="${ENV_FILE:-${DEFAULT_ENV_FILE}}"

REPO_URL="${REPO_URL:-https://github.com/tgrytnes/Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification.git}"

get_env_var() {
  local key="$1"
  python3 - "${ENV_FILE}" "${key}" <<'PY'
import sys

path, key = sys.argv[1], sys.argv[2]
try:
    lines = open(path, encoding="utf-8").read().splitlines()
except FileNotFoundError:
    sys.exit(0)

for line in lines:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    if k.strip() != key:
        continue
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    print(v)
    sys.exit(0)
PY
}

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root (or via sudo) so Python can be installed to ${PYTHON_PREFIX}."
  exit 1
fi

echo "Installing build dependencies..."
apt-get update
apt-get install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libncurses5-dev \
  libncursesw5-dev \
  libreadline-dev \
  libsqlite3-dev \
  libgdbm-dev \
  libdb5.3-dev \
  libbz2-dev \
  libexpat1-dev \
  liblzma-dev \
  tk-dev \
  libffi-dev \
  uuid-dev \
  wget \
  curl

PYTHON_BIN="${PYTHON_PREFIX}/bin/python3.11"
if [[ -x "${PYTHON_BIN}" ]]; then
  echo "Python 3.11 already installed at ${PYTHON_BIN}."
else
  echo "Downloading Python ${PYTHON_VERSION}..."
  cd /usr/src
  if [[ ! -f "Python-${PYTHON_VERSION}.tgz" ]]; then
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
  fi

  if [[ ! -d "Python-${PYTHON_VERSION}" ]]; then
    tar -xzf "Python-${PYTHON_VERSION}.tgz"
  fi

  echo "Building and installing Python ${PYTHON_VERSION}..."
  cd "/usr/src/Python-${PYTHON_VERSION}"
  ./configure --with-ensurepip=install
  make -j"$(nproc)"
  make altinstall
fi

echo "Installing Poetry ${POETRY_VERSION}..."
export POETRY_VERSION
curl -sSL https://install.python-poetry.org | "${PYTHON_BIN}" -

POETRY_BIN="${HOME}/.local/bin/poetry"
export PATH="${HOME}/.local/bin:${PATH}"

# Persist Poetry in PATH for future shells
if ! grep -q '/root/.local/bin' /root/.bashrc; then
  echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc
fi




GITHUB_NAME_ENV=""
GITHUB_EMAIL_ENV=""
GITHUB_USERNAME_ENV=""
GITHUB_TOKEN_ENV=""
GIT_USER_NAME_ENV=""
GIT_USER_EMAIL_ENV=""
WANDB_API_KEY_ENV=""
AZURE_STORAGE_CONNECTION_STRING_ENV=""
if [[ -f "${ENV_FILE}" ]]; then
  GITHUB_NAME_ENV="$(get_env_var GITHUB_NAME)"
  GITHUB_EMAIL_ENV="$(get_env_var GITHUB_EMAIL)"
  GITHUB_USERNAME_ENV="$(get_env_var GITHUB_USERNAME)"
  GITHUB_TOKEN_ENV="$(get_env_var GITHUB_TOKEN)"
  GIT_USER_NAME_ENV="$(get_env_var GIT_USER_NAME)"
  GIT_USER_EMAIL_ENV="$(get_env_var GIT_USER_EMAIL)"
  WANDB_API_KEY_ENV="$(get_env_var WANDB_API_KEY)"
  AZURE_STORAGE_CONNECTION_STRING_ENV="$(get_env_var AZURE_STORAGE_CONNECTION_STRING)"
fi

GIT_NAME="${GIT_USER_NAME_ENV:-${GITHUB_NAME_ENV:-${GIT_USER_NAME:-${GITHUB_NAME:-}}}}"
GIT_EMAIL="${GIT_USER_EMAIL_ENV:-${GITHUB_EMAIL_ENV:-${GIT_USER_EMAIL:-${GITHUB_EMAIL:-}}}}"
GIT_USERNAME="${GITHUB_USERNAME_ENV:-${GITHUB_USERNAME:-}}"
if [[ -z "${GIT_EMAIL}" && -n "${GIT_USERNAME}" ]]; then
  GIT_EMAIL="${GIT_USERNAME}@users.noreply.github.com"
fi

if [[ -n "${GIT_NAME}" && -n "${GIT_EMAIL}" ]]; then
  echo "Configuring git user.name and user.email..."
  git config --global user.name "${GIT_NAME}"
  git config --global user.email "${GIT_EMAIL}"
else
  echo "Skipping git config (set GIT_USER_NAME/GIT_USER_EMAIL or GITHUB_NAME/GITHUB_EMAIL in .env)."
fi

if [[ -n "${GITHUB_USERNAME_ENV}" && -n "${GITHUB_TOKEN_ENV}" ]]; then
  echo "Configuring git credential helper..."
  git config --global credential.helper store
  printf 'https://%s:%s@github.com\n' "${GITHUB_USERNAME_ENV}" "${GITHUB_TOKEN_ENV}" > "${HOME}/.git-credentials"
else
  echo "Skipping git credentials (set GITHUB_USERNAME and GITHUB_TOKEN in .env)."
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "Cloning repository into ${REPO_DIR}..."
  mkdir -p "${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

echo "Installing Python dependencies with Poetry..."
cd "${REPO_DIR}"
"${POETRY_BIN}" env use "${PYTHON_BIN}"
"${POETRY_BIN}" install

if [[ -n "${WANDB_API_KEY_ENV}" ]]; then
  echo "Logging into Weights & Biases..."
  "${POETRY_BIN}" run wandb login "${WANDB_API_KEY_ENV}"
else
  echo "Skipping W&B login (set WANDB_API_KEY in .env)."
fi

if [[ -n "${AZURE_STORAGE_CONNECTION_STRING_ENV}" ]]; then
  echo "Configuring DVC Azure remote..."
  "${POETRY_BIN}" run pip install dvc-azure

  # Remove any broken config and set fresh
  "${POETRY_BIN}" run dvc remote modify --local --unset azure-remote connection_string 2>/dev/null || true
  "${POETRY_BIN}" run dvc remote modify --local azure-remote connection_string "${AZURE_STORAGE_CONNECTION_STRING_ENV}"

  # Verify configuration was set correctly
  if "${POETRY_BIN}" run dvc remote list --local | grep -q azure-remote; then
    echo "DVC remote configured successfully."
  else
    echo "Warning: DVC remote configuration may not have persisted."
  fi

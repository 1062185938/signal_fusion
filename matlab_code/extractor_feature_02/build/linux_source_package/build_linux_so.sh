#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build libextractAllFeatures.so from MATLAB Coder generated C++ source.

Usage:
  ./build_linux_so.sh [options]

Options:
  --src-dir DIR          Generated source directory. Default: ./src, or ./build/linux_source_package/src.
  --native-api-dir DIR   C ABI wrapper directory. Default: ./native_api, or ./build/linux_source_package/native_api.
  --out-dir DIR          Output directory. Default: ./out, or ./build/linux_source_package/out.
  --install-dir DIR      Optional skill native/linux directory. Copies .so and iq_feature_c_api.h there.
  --clean                Remove object/output directory before build.
  -h, --help             Show this help.

Environment variables:
  CXX                    C++ compiler. Default: g++.
  CXXFLAGS               Compile flags. Default includes -O2 -std=c++17 -fPIC -fopenmp.
  LDFLAGS                Link flags. Default includes -shared -fopenmp and rpath=$ORIGIN.

Linux does not need MATLAB for this step. It only needs a C++ compiler and the packaged source files.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "$SCRIPT_DIR/src" ]]; then
  PACKAGE_DIR="$SCRIPT_DIR"
else
  PACKAGE_DIR="$SCRIPT_DIR/build/linux_source_package"
fi

SRC_DIR="$PACKAGE_DIR/src"
NATIVE_API_DIR="$PACKAGE_DIR/native_api"
OUT_DIR="$PACKAGE_DIR/out"
INSTALL_DIR=""
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-dir)
      SRC_DIR="$2"
      shift 2
      ;;
    --native-api-dir)
      NATIVE_API_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory not found: $SRC_DIR" >&2
  echo "Run package_codegen_source_for_linux.m on Windows MATLAB first, then copy the package to Linux." >&2
  exit 1
fi

if [[ ! -d "$NATIVE_API_DIR" ]]; then
  echo "Native API directory not found: $NATIVE_API_DIR" >&2
  exit 1
fi

for required in extractAllFeatures.cpp extractAllFeatures.h extractAllFeatures_initialize.h extractAllFeatures_terminate.h; do
  if [[ ! -f "$SRC_DIR/$required" ]]; then
    echo "Required generated file missing: $SRC_DIR/$required" >&2
    exit 1
  fi
done

for required in iq_feature_c_api.cpp iq_feature_c_api.h; do
  if [[ ! -f "$NATIVE_API_DIR/$required" ]]; then
    echo "Required native API file missing: $NATIVE_API_DIR/$required" >&2
    exit 1
  fi
done

CXX="${CXX:-g++}"
DEFAULT_CXXFLAGS="-O2 -std=c++17 -fPIC -fvisibility=hidden -fopenmp"
DEFAULT_LDFLAGS="-shared -fopenmp -Wl,-rpath,'\$ORIGIN'"
CXXFLAGS="${CXXFLAGS:-$DEFAULT_CXXFLAGS}"
LDFLAGS="${LDFLAGS:-$DEFAULT_LDFLAGS}"

OBJ_DIR="$OUT_DIR/obj"
OUTPUT_SO="$OUT_DIR/libextractAllFeatures.so"

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "$OUT_DIR"
fi

mkdir -p "$OBJ_DIR"

mapfile -t SOURCES < <(find "$SRC_DIR" -maxdepth 1 -type f \( -name '*.cpp' -o -name '*.c' \) | sort)
SOURCES+=("$NATIVE_API_DIR/iq_feature_c_api.cpp")

OBJECTS=()
for src in "${SOURCES[@]}"; do
  base="$(basename "$src")"
  obj="$OBJ_DIR/${base%.*}.o"
  echo "[compile] $base"
  "$CXX" $CXXFLAGS -I"$SRC_DIR" -I"$NATIVE_API_DIR" -c "$src" -o "$obj"
  OBJECTS+=("$obj")
done

echo "[link] $(basename "$OUTPUT_SO")"
# shellcheck disable=SC2086
"$CXX" ${OBJECTS[@]} $LDFLAGS -o "$OUTPUT_SO"

if command -v strip >/dev/null 2>&1; then
  strip --strip-unneeded "$OUTPUT_SO" || true
fi

echo "[done] $OUTPUT_SO"

if [[ -n "$INSTALL_DIR" ]]; then
  mkdir -p "$INSTALL_DIR"
  cp "$OUTPUT_SO" "$INSTALL_DIR/libextractAllFeatures.so"
  cp "$NATIVE_API_DIR/iq_feature_c_api.h" "$INSTALL_DIR/iq_feature_c_api.h"
  echo "[install] copied to $INSTALL_DIR"
fi

if command -v ldd >/dev/null 2>&1; then
  echo "[ldd] dependency check:"
  ldd "$OUTPUT_SO" || true
fi
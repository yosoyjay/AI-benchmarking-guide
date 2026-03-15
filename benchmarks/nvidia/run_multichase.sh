#!/bin/bash
set -euo pipefail
# Detect CPU count and NUMA nodes dynamically, then run multichase
# across representative CPUs and all NUMA memory nodes.
#
# Usage: run_multichase.sh [/path/to/multichase] [-s stride] [-m memory] [-n iterations]

multichase_bin="${1:-multichase}"
shift || true

# Defaults (match multichase upstream defaults used historically)
stride="512"
memory="1g"
iterations="120"

while getopts "s:m:n:" opt; do
    case "$opt" in
        s) stride="$OPTARG" ;;
        m) memory="$OPTARG" ;;
        n) iterations="$OPTARG" ;;
        *) ;;
    esac
done

total_cpus=$(nproc)
last_cpu=$((total_cpus - 1))
mid_cpu=$((total_cpus / 2))

# Build list of representative CPUs: first, middle, last
cpus=(0 "$mid_cpu" "$last_cpu")

# Detect NUMA nodes from sysfs
numa_nodes=()
for node_dir in /sys/devices/system/node/node[0-9]*; do
    numa_nodes+=("$(basename "${node_dir}" | sed 's/node//')")
done
# Fallback if sysfs detection fails
if [ ${#numa_nodes[@]} -eq 0 ]; then
    numa_nodes=(0 1)
fi

# Print header
printf "%4s" "CPU"
for numa in "${numa_nodes[@]}"; do
    printf " %7s" "NODE${numa}"
done
printf "\n"

for cpu in "${cpus[@]}"; do
    printf "%4s " "${cpu}"
    for numa in "${numa_nodes[@]}"; do
        result=$(numactl -C "${cpu}" -m "${numa}" "${multichase_bin}" -s "${stride}" -m "${memory}" -n "${iterations}")
        printf "%7.1f " "${result}"
    done
    printf "\n"
done

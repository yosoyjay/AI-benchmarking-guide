#!/bin/bash
# Detect CPU count and NUMA nodes dynamically
total_cpus=$(nproc)
last_cpu=$((total_cpus - 1))
mid_cpu=$((total_cpus / 2))

# Build list of representative CPUs: first, middle, last
cpus=(0 "$mid_cpu" "$last_cpu")

# Detect NUMA nodes from sysfs
numa_nodes=()
for node_dir in /sys/devices/system/node/node[0-9]*; do
    numa_nodes+=("$(basename "$node_dir" | sed 's/node//')")
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
printf "%4s " ${cpu}
for numa in "${numa_nodes[@]}"; do
result=$(numactl -C ${cpu} -m ${numa} ../../multichase/multichase -s 512 -m 1g -n 120)
printf "%7.1f " ${result}
done
printf "\n"
done

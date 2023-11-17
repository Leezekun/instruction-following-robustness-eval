#!/bin/bash

# 目标目录
target_dir="./__deprecated__/outputs-dev-500-1116/"

# 检查目标目录是否存在，如果不存在则创建
if [ ! -d "$target_dir" ]; then
    mkdir -p "$target_dir"
fi

# 移动所有以 -dev-500.json 结尾的文件到目标目录
for file in *-dev-500.json; do
    # 检查文件是否存在
    if [ -f "$file" ]; then
        mv "$file" "$target_dir"
        echo "Moved $file to $target_dir"
    fi
done

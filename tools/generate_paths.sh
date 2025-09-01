#!/bin/bash

# 设置基础路径
BASE_DIR="lmy_white_72mins/square/speech"
OUTPUT_FILE="lmy_white_72mins/lmy_white_72mins.txt"

# 删除旧的输出文件（如果存在）
> "$OUTPUT_FILE"

echo "正在生成路径文件..."

# 计数器
count=0

# 直接遍历所有4位数字文件夹
for folder in "$BASE_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        # 检查是否是4位数字格式（0001, 0002等）
        if echo "$folder_name" | grep -E '^[0-9]{5}$' > /dev/null; then
            echo "处理目录: $folder_name"
            # 生成相对路径并写入文件
            echo "$BASE_DIR/$folder_name" >> "$OUTPUT_FILE"
            count=$((count + 1))
        fi
    fi
done

echo "完成！共找到 $count 个数据文件夹"
echo "路径文件已保存为: $OUTPUT_FILE"
echo ""
echo "前10行内容："
head -10 "$OUTPUT_FILE"
echo ""
echo "最后10行内容："
tail -10 "$OUTPUT_FILE"

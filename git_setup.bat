@echo off
echo ========================================
echo Git初始化和GitHub上传脚本
echo ========================================

echo 1. 初始化Git仓库...
git init

echo 2. 添加所有文件到暂存区...
git add .

echo 3. 创建初始提交...
git commit -m "Initial commit: C-MAPSS RUL prediction project with TSMixer and Transformer models"

echo 4. 设置主分支名称...
git branch -M main

echo ========================================
echo 现在需要手动执行以下步骤：
echo ========================================
echo 1. 在GitHub上创建一个新的仓库
echo 2. 复制仓库的URL
echo 3. 运行以下命令（替换YOUR_GITHUB_REPO_URL）：
echo.
echo    git remote add origin YOUR_GITHUB_REPO_URL
echo    git push -u origin main
echo.
echo ========================================
echo 示例：
echo    git remote add origin https://github.com/yourusername/cmapss-rul-prediction.git
echo    git push -u origin main
echo ========================================

pause

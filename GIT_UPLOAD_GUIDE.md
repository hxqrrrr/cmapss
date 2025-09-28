# Git上传GitHub完整指南

## 方法一：使用自动化脚本（推荐）

### Windows用户
双击运行 `git_setup.bat` 文件，或在命令行中执行：
```cmd
git_setup.bat
```

### Linux/Mac用户
```bash
chmod +x git_setup.sh
./git_setup.sh
```

## 方法二：手动执行命令

### 1. 初始化Git仓库
```bash
git init
```

### 2. 添加所有文件到暂存区
```bash
git add .
```

### 3. 创建初始提交
```bash
git commit -m "Initial commit: C-MAPSS RUL prediction project with TSMixer and Transformer models"
```

### 4. 设置主分支名称
```bash
git branch -M main
```

### 5. 在GitHub上创建仓库
1. 登录GitHub
2. 点击右上角的"+"号，选择"New repository"
3. 输入仓库名称，如：`cmapss-rul-prediction`
4. 选择公开或私有
5. **不要**勾选"Initialize this repository with a README"
6. 点击"Create repository"

### 6. 连接远程仓库并推送
```bash
# 替换下面的URL为你的GitHub仓库URL
git remote add origin https://github.com/yourusername/cmapss-rul-prediction.git
git push -u origin main
```

## 后续更新代码的命令

### 添加新文件或修改后的文件
```bash
git add .
```

### 提交更改
```bash
git commit -m "描述你的更改内容"
```

### 推送到GitHub
```bash
git push
```

## 常用Git命令

### 查看文件状态
```bash
git status
```

### 查看提交历史
```bash
git log --oneline
```

### 查看远程仓库
```bash
git remote -v
```

### 创建新分支
```bash
git checkout -b feature-branch-name
```

### 切换分支
```bash
git checkout main
```

### 合并分支
```bash
git merge feature-branch-name
```

## 注意事项

1. **首次推送**：使用 `git push -u origin main` 设置上游分支
2. **后续推送**：直接使用 `git push`
3. **大文件处理**：如果有大于100MB的文件，考虑使用Git LFS
4. **敏感信息**：确保不要提交密码、API密钥等敏感信息
5. **.gitignore**：已经配置好，会自动排除不需要的文件

## 文件说明

项目中已创建的Git相关文件：
- `.gitignore`：定义哪些文件不需要上传到GitHub
- `README.md`：项目说明文档
- `git_setup.bat`：Windows自动化脚本
- `git_setup.sh`：Linux/Mac自动化脚本
- `GIT_UPLOAD_GUIDE.md`：本指南文档

## 建议的仓库名称

- `cmapss-rul-prediction`
- `aircraft-engine-rul-prediction`
- `tsmixer-rul-forecasting`
- `deep-learning-rul-prediction`

## 如果遇到问题

1. **权限问题**：确保你有GitHub仓库的写权限
2. **网络问题**：检查网络连接或使用VPN
3. **认证问题**：配置GitHub的SSH密钥或使用Personal Access Token
4. **文件太大**：检查是否有大文件需要用Git LFS处理

## SSH密钥配置（可选但推荐）

1. 生成SSH密钥：
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. 添加到SSH代理：
```bash
ssh-add ~/.ssh/id_ed25519
```

3. 复制公钥到GitHub：
```bash
cat ~/.ssh/id_ed25519.pub
```

4. 在GitHub设置中添加SSH密钥

5. 使用SSH URL：
```bash
git remote add origin git@github.com:yourusername/cmapss-rul-prediction.git
```

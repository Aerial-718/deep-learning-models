# 从本地电脑上传到 GitHub

服务器不会保存你的 GitHub 凭据，也不会执行远程推送。请下载生成的
`deep-learning-models.zip`，然后在自己的电脑上完成以下步骤。

## 1. 创建 GitHub 空仓库

登录 GitHub，创建名为 `deep-learning-models` 的新仓库。

为了避免第一次 push 出现历史冲突，创建时暂时不要勾选：

- Add a README file
- Add `.gitignore`
- Choose a license

仓库创建完成后，复制它的 HTTPS 或 SSH 地址。

## 2. 解压并检查

将下载的 ZIP 解压，进入仓库根目录：

```bash
cd deep-learning-models
```

确认当前目录中能够看到：

```text
README.md
UPLOAD_FROM_LOCAL.md
RNN_LSTM_GRU/
```

建议先验证第一个项目：

```bash
cd RNN_LSTM_GRU
conda env create -f environment.yml
conda activate rnn-lstm-gru
python -m pip install -e .
pytest -q
cd ..
```

如果本地已经存在同名 Conda 环境，可以跳过 `conda env create`。

## 3. 初始化 Git

```bash
git init
git branch -M main
git add .
git status
git commit -m "feat: add RNN, LSTM, and GRU implementations"
```

请在 commit 前查看 `git status`，确认没有数据集、模型权重或缓存文件被加入。

## 4. 连接并推送 GitHub

使用 HTTPS：

```bash
git remote add origin https://github.com/YOUR_USERNAME/deep-learning-models.git
git push -u origin main
```

或者使用 SSH：

```bash
git remote add origin git@github.com:YOUR_USERNAME/deep-learning-models.git
git push -u origin main
```

将 `YOUR_USERNAME` 替换为你的 GitHub 用户名。如果 GitHub 要求认证，请使用浏览器登录、
Personal Access Token 或已经配置好的 SSH key，不要把 token 写入仓库文件。

## 5. 后续新增模型

每个模型使用独立目录，例如：

```text
deep-learning-models/
├── RNN_LSTM_GRU/
├── CNN/
├── AUTOENCODER_VAE/
└── TRANSFORMER/
```

完成修改后使用常规 Git 流程：

```bash
git add .
git commit -m "feat: add <model-name> implementation"
git push
```

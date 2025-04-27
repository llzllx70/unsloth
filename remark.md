
1. 添加原仓库为上游远程仓库：
git remote add upstream https://github.com/unslothai/unsloth
git remote -v

2. 从上游仓库获取最新更改：
git fetch upstream

3. 合并到本地分支（例如main或master）：
git merge upstream/main

4. 进行自己的改进
git add grpo-qwen2.5-3b-instruct.py 
git commit -m 'add grpo-qwen2.5-3b-instruct.py'
git push origin main 


# 使用官方 Node.js 20 版本
FROM node:20-bookworm

# 设置工作目录为 /app
WORKDIR /app

# 将 package.json 和 package-lock.json 拷贝到容器中
COPY package*.json ./

# 安装依赖
RUN npm install

# 将其余源代码拷贝到容器中
COPY . .

# 暴露端口 5173 用于 Vue 项目访问
EXPOSE 5173

# 启动开发模式，指定监听地址
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
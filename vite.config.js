import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// [https://vitejs.dev/config/](https://vitejs.dev/config/)
export default defineConfig({
  plugins: [react()],
  // ⚠️ 注意：这里必须修改为您在 GitHub 上创建的仓库名称
  // 例如仓库名为 smart-grid-ai，则填写 '/smart-grid-ai/'
  base: '/smart-grid-ai/',
  build: {
    outDir: 'dist',
  }
})
